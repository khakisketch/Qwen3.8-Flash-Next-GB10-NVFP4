"""Experimental launch preparation for the NVIDIA Spark baseline.

Real model starts are disabled because application/CUDA OOM coverage is
incomplete. The two proposed enforcement layers below remain experimental:

1. IN-CONTAINER PID-1 watchdog (serve/inner_watchdog.py, bind-mounted
   read-only, ``--entrypoint python3``). It is designed to sample host
   metrics from ``/proc`` before the server starts, spawn the server once,
   SIGKILL it on an observed breach, and never restart it. It can continue
   monitoring if the host supervisor dies, but hangs or failures of both
   layers remain unverified.
   Any server exit ends the container.

2. HOST supervisor (this module). Preflight, exact-name+label scoped
   control, independent guard monitor from before ``docker run`` until
   exit, startup/ready/health supervision, and prompt ``docker kill``
   (not slow ``docker stop``) of the verified container on an observed
   violation. Mutating commands are scoped by name, label and id using
   ``docker inspect --format`` without fetching Config.Env. Late-start
   cleanup races need further review before real use.

What this module does NOT do:
  * never auto-restarts anything (no ``--restart``; any exit is failure);
  * never passes credentials (HF_TOKEN et al. never set; host env never
    inherited wholesale; test-only ``QWEN38_*`` knobs never enter);
  * never uses a shell (all children are fixed argv lists);
  * actual ``launch`` is disabled even with ``--confirm-host-risk``;
    the experimental architecture does not cover every application OOM.

Only the Python 3.12 standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .baseline import (
    EXPECTED_SHARD_COUNT,
    EXPECTED_SHARD_NAMES,
    HF_HUB_HOST_PATH,
    HF_REPO_DIRNAME,
    IMAGE_ENTRYPOINT_SCRIPT,
    IMAGE_PINNED,
    MODEL_ID,
    SNAPSHOT_REF,
    VLLM_CONTAINER_ENV,
    VLLM_SERVE_ARGV,
    WATCHDOG_CONTAINER_PATH,
)
from .guard import (
    POLL_INTERVAL_S,
    Guard,
    GuardError,
    GuardMonitor,
)

PROJECT_LABEL_VALUE = "qwen38-fp4-local"
CONTAINER_NAME = "qwen38-fp4-local-serve"
CONTAINER_HF_HOME = "/root/.cache/huggingface"
CONTAINER_REPO_TARGET = f"{CONTAINER_HF_HOME}/hub/{HF_REPO_DIRNAME}"
HOST_PORT = 8000
HOST_BIND = "127.0.0.1"

STARTUP_TIMEOUT_S = 1800.0
DOCKER_START_TIMEOUT_S = 120.0
START_POLL_SLICE_S = 0.2
SETTLE_GRACE_S = 10.0
DOCKER_OP_TIMEOUT_S = 30
STOP_GRACE_S = 30
READY_PROBE_TIMEOUT_S = 5.0
READY_REPROBE_S = 15.0
HEALTH_MAX_CONSECUTIVE_FAILURES = 3
INSPECT_MAX_CONSECUTIVE_FAILURES = 3

# Credential-free inspect: id, name, our label, state, exit code ONLY.
# Full ``docker inspect`` JSON (incl. Config.Env) is NEVER fetched, so a
# foreign same-name container's secrets cannot be read through us.
DOCKER_INSPECT_FORMAT = (
    "{{.Id}}|{{.Name}}|{{index .Config.Labels \"project\"}}"
    "|{{.State.Status}}|{{.State.ExitCode}}"
)

# Minimal env inherited into the docker CLI child. Everything else -- in
# particular any *TOKEN*, *SECRET*, *KEY*, *PASSWORD*, HF_TOKEN,
# HF_HUB_TOKEN -- is dropped by construction.
_SUBPROCESS_ENV_ALLOW = frozenset(
    {"PATH", "LANG", "LC_ALL", "LC_MESSAGES", "TZ", "TMPDIR"}
)

_FORBIDDEN_CONTAINER_ENV_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|QWEN38_)",
    re.IGNORECASE,
)


class LaunchError(Exception):
    """Preflight/build/supervision failure. Fail closed (do not start)."""


class Refusal(Exception):
    """The launcher refused to act (missing confirmation, foreign target)."""


class InspectError(LaunchError):
    """Container identity could not be verified. Never treat as absence."""


class RealStartDisabled(LaunchError):
    """All real model/container start paths are hard-disabled."""


# HARD DISABLE. No independently verified mechanism currently meets the
# agreed immediate-stop-on-OOM guard for application/CUDA OOM paths: the
# host oom_kill counter does not count them, and the pinned image only
# guarantees ONE observed path (vllm/v1/worker/gpu_model_runner.py
# catches torch.cuda.OutOfMemoryError, logs "Failed to load model - not
# enough GPU memory", and re-raises) -- not every app OOM. Until such a
# mechanism is independently verified, launch() raises before ANY
# preflight, subprocess, or container action. This constant is
# intentionally NOT wired to any flag, argument, or environment variable:
# only a reviewed code change adding a verified mechanism may re-enable
# it. The model is NOT ready for use.
_REAL_START_ENABLED = False


# ---------------------------------------------------------------------------
# Environment handling
# ---------------------------------------------------------------------------

def sanitized_subprocess_env() -> dict[str, str]:
    """Minimal env for docker CLI children. Never inherits credentials."""
    env = {k: v for k, v in os.environ.items() if k in _SUBPROCESS_ENV_ALLOW}
    for k in list(env):
        if _FORBIDDEN_CONTAINER_ENV_RE.search(k):
            del env[k]
    env.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    # Docker falls back to the OS account's home when HOME is unset. Point
    # it at a procfs directory that cannot contain config.json instead, and
    # explicitly select the observed local socket without using a context.
    env["DOCKER_CONFIG"] = "/proc/self/fd"
    env["DOCKER_HOST"] = "unix:///var/run/docker.sock"
    return env


def container_env() -> dict[str, str]:
    """Env passed into the serving container (``-e KEY=VAL`` each)."""
    env = dict(VLLM_CONTAINER_ENV)
    env["HF_HUB_OFFLINE"] = "1"  # cached weights only; no network fetch
    env["HF_HOME"] = CONTAINER_HF_HOME
    # The only credential intentionally forwarded is vLLM's API key. It is
    # read from the host service environment (never a project file) so the
    # LAN listener is not anonymous. Keep the fixed baseline free of secrets.
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        if api_key != api_key.strip() or "\n" in api_key or "\r" in api_key:
            raise LaunchError("VLLM_API_KEY must be a single-line value")
        env["VLLM_API_KEY"] = api_key
    for key in env:
        if _FORBIDDEN_CONTAINER_ENV_RE.search(key) and key != "VLLM_API_KEY":
            raise LaunchError(
                f"refusing to set container env var {key!r}: looks like a "
                "secret or a test-only override"
            )
    return env


# ---------------------------------------------------------------------------
# Cache resolution: exact 21-shard baseline, symlink-safe, read-only
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedCache:
    host_repo: Path      # repo dir, fully resolved, under hub root
    host_snapshot: Path  # snapshots/<ref>, fully resolved
    shard_count: int


def resolve_snapshot_cache(hub_root: str | Path = HF_HUB_HOST_PATH) -> ResolvedCache:
    """Locate and verify the cached snapshot. Raises on ANY deviation.

    * resolves repo dir and snapshot fully; both must stay under the hub
      root (fail closed on symlink escape); the caller mounts
      ``host_repo`` read-only (``:ro``) so relative ``../../blobs``
      symlinks keep working;
    * requires config.json, model.safetensors.index.json, and EXACTLY the
      21 expected ``model-*-of-00021.safetensors`` shard names -- missing,
      extra, broken, or escaping shard symlinks all refuse;
    * cross-checks the index: the set of files referenced by
      ``weight_map`` must equal the expected 21 (serves a complete,
      self-consistent snapshot or nothing).
    """
    root = Path(hub_root)
    if not root.is_dir():
        raise LaunchError(f"hub root not a directory: {root}")
    root_real = root.resolve()
    repo = root_real / HF_REPO_DIRNAME
    if not repo.is_dir():
        raise LaunchError(f"cached repo dir missing: {repo}")
    try:
        repo_real = repo.resolve(strict=True)
    except OSError as exc:
        raise LaunchError(f"cannot resolve repo dir: {exc}") from exc
    try:
        repo_real.relative_to(root_real)
    except ValueError as exc:
        raise LaunchError(
            f"repo dir resolves outside hub root (symlink escape?): "
            f"{repo_real} not under {root_real}"
        ) from exc
    snap = repo_real / "snapshots" / SNAPSHOT_REF
    if not snap.exists():
        raise LaunchError(f"cached snapshot missing: {snap}")
    try:
        snap_real = snap.resolve(strict=True)
    except OSError as exc:
        raise LaunchError(f"cannot resolve snapshot symlinks: {exc}") from exc
    try:
        snap_real.relative_to(root_real)
    except ValueError as exc:
        raise LaunchError(
            f"snapshot resolves outside hub root (symlink escape?): "
            f"{snap_real} not under {root_real}"
        ) from exc
    if not (snap_real / "config.json").is_file():
        raise LaunchError("snapshot incomplete: missing config.json")
    index_path = snap_real / "model.safetensors.index.json"
    if not index_path.is_file():
        raise LaunchError(
            "snapshot incomplete: missing model.safetensors.index.json"
        )
    tensors = snap_real / "tensors"
    if not tensors.is_dir():
        raise LaunchError("snapshot incomplete: missing tensors/")
    actual: set[str] = set()
    for shard in sorted(tensors.glob("model-*.safetensors")):
        try:
            real = shard.resolve(strict=True)
        except OSError as exc:
            raise LaunchError(
                f"shard symlink broken {shard.name}: {exc}"
            ) from exc
        try:
            real.relative_to(repo_real)
        except ValueError as exc:
            raise LaunchError(
                f"shard {shard.name} escapes repo root: {exc}"
            ) from exc
        if not real.is_file():
            raise LaunchError(f"shard {shard.name} is not a file")
        actual.add(shard.name)
    if actual != EXPECTED_SHARD_NAMES:
        missing = sorted(EXPECTED_SHARD_NAMES - actual)
        extra = sorted(actual - EXPECTED_SHARD_NAMES)
        raise LaunchError(
            f"snapshot shard set != exact {EXPECTED_SHARD_COUNT}-shard "
            f"baseline: missing={missing} extra={extra}"
        )
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LaunchError(f"cannot parse weight index: {exc}") from exc
    try:
        weight_map = index["weight_map"]
    except (KeyError, TypeError) as exc:
        raise LaunchError(f"weight index has no weight_map: {exc}") from exc
    referenced = {Path(v).name for v in weight_map.values()}
    if referenced != EXPECTED_SHARD_NAMES:
        raise LaunchError(
            "weight index references != exact 21-shard baseline: "
            f"missing={sorted(EXPECTED_SHARD_NAMES - referenced)} "
            f"extra={sorted(referenced - EXPECTED_SHARD_NAMES)}"
        )
    return ResolvedCache(
        host_repo=repo_real, host_snapshot=snap_real,
        shard_count=len(actual),
    )


# ---------------------------------------------------------------------------
# Command construction (fixed argv, no shell)
# ---------------------------------------------------------------------------

def _vllm_cmd_for_container() -> list[str]:
    """vLLM server command tail (after the watchdog + entrypoint script).

    The literal ``"VLLM"`` element is resolved to an explicit executable
    via PATH *inside the container* by the watchdog (fail-closed if
    missing) -- the image entrypoint is never relied on to conjure it.
    """
    argv = list(VLLM_SERVE_ARGV)
    # Insert (not replace) the explicit-binary placeholder: the watchdog
    # resolves "VLLM" via PATH in-container, keeping the "serve"
    # subcommand intact: [vllm, serve, MODEL, flags...].
    argv.insert(0, "VLLM")
    # D1: loopback only. With --network host the container loopback IS the
    # host loopback, so this is what keeps the API off the LAN.
    try:
        host_i = argv.index("--host")
        argv[host_i + 1] = HOST_BIND
    except (ValueError, IndexError) as exc:
        raise LaunchError(f"baseline argv has no --host value: {exc}") from exc
    if "0.0.0.0" in argv:
        raise LaunchError("refusing: 0.0.0.0 bind present in vLLM argv")
    return argv


def default_watchdog_host_path() -> Path:
    return Path(__file__).with_name("inner_watchdog.py")


def build_docker_argv(
    cache: ResolvedCache,
    watchdog_host: Path | str | None = None,
    image: str = IMAGE_PINNED,
    name: str = CONTAINER_NAME,
) -> list[str]:
    """Build the supervised ``docker run`` argv. Raises on policy breach."""
    if name != CONTAINER_NAME:
        raise LaunchError(f"refusing: unexpected container name {name!r}")
    if "@sha256:" not in image or ":latest" in image:
        raise LaunchError(
            f"refusing: image must be digest-pinned, got {image!r}"
        )
    watchdog_host = watchdog_host or default_watchdog_host_path()
    watchdog_host = Path(watchdog_host)
    if not watchdog_host.is_file():
        raise LaunchError(f"watchdog script missing: {watchdog_host}")
    argv: list[str] = [
        "docker", "run", "-d",
        "--name", name,
        "--label", f"project={PROJECT_LABEL_VALUE}",
        "--gpus", "all",
        # D2/D3/D4: user-approved host network/IPC + unconfined (io_uring
        # evidence in baseline.py). Locality is enforced by D1 instead.
        "--ipc", "host",
        "--network", "host",
        "--shm-size=16g",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--security-opt", "seccomp=unconfined",
        "--security-opt", "apparmor=unconfined",
        "--stop-timeout", str(STOP_GRACE_S),
        # D7: PID-1 watchdog. No --restart flag anywhere: any exit is
        # failure, never restarted (a crash-looping server under memory
        # pressure amplifies harm).
        "--entrypoint", "python3",
    ]
    for key, val in container_env().items():
        argv += ["-e", f"{key}={val}"]
    argv += ["-v", f"{watchdog_host}:{WATCHDOG_CONTAINER_PATH}:ro"]
    argv += ["-v", f"{cache.host_repo}:{CONTAINER_REPO_TARGET}:ro"]
    argv += [image]
    argv += [WATCHDOG_CONTAINER_PATH, IMAGE_ENTRYPOINT_SCRIPT]
    argv += _vllm_cmd_for_container()
    blob = "\n".join(argv)
    for banned in (
        "0.0.0.0", "--restart", "/bin/bash", "HF_TOKEN",
        "QWEN38_PROC_ROOT", "QWEN38_POLL_S",
    ):
        if banned in blob:
            raise LaunchError(f"refusing: {banned!r} present in docker argv")
    if f"project={PROJECT_LABEL_VALUE}" not in argv:
        raise LaunchError("refusing: project label missing")
    return argv


# ---------------------------------------------------------------------------
# Container identity: exact name + label, credential-free, fail-closed
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopedInfo:
    id: str
    name: str
    label: str
    status: str
    exit_code: int


def _run_capture(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=sanitized_subprocess_env(),
        shell=False,
    )


def inspect_exact(
    name: str = CONTAINER_NAME,
    run=_run_capture,  # injectable for tests
) -> ScopedInfo | None:
    """Identify our exact container or return None if absent.

    Fetches ONLY id/name/label/state/exit-code via ``--format``; full
    inspect JSON (incl. Config.Env) is never requested, so a foreign
    same-name container's secrets cannot be read through us. ANY docker
    or parse failure raises :class:`InspectError` -- never mistaken for
    absence. Absence is ONLY "No such container".
    """
    if name != CONTAINER_NAME:
        raise Refusal(f"refusing to inspect foreign container {name!r}")
    try:
        proc = run(
            ["docker", "inspect", "--format", DOCKER_INSPECT_FORMAT, name],
            DOCKER_OP_TIMEOUT_S,
        )
    except Exception as exc:
        raise InspectError(
            f"docker inspect did not complete for {name!r}: {exc}"
        ) from exc
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        # Absence is ONLY a no-such-object report. Docker versions differ
        # in wording ("No such container" vs "no such object"); match
        # both, case-insensitively, and nothing else.
        if "no such container" in (out + err).lower() or "no such object" in (
            out + err
        ).lower():
            return None
        raise InspectError(
            f"docker inspect failed for {name!r} (NOT absence): "
            f"{(out + ' ' + err).strip()[:300]}"
        )
    parts = out.split("|")
    if len(parts) != 5:
        raise InspectError(
            f"unparseable inspect output for {name!r}: {out[:200]!r}"
        )
    cid, cname, label, status, code = parts
    if not cid or not status:
        raise InspectError(f"incomplete inspect output for {name!r}")
    return ScopedInfo(
        id=cid, name=cname.lstrip("/"), label=label, status=status,
        exit_code=int(code) if code.lstrip("-").isdigit() else -1,
    )


def _assert_ours(info: ScopedInfo, cid: str | None = None) -> None:
    if info.name != CONTAINER_NAME or info.label != PROJECT_LABEL_VALUE:
        raise Refusal(
            f"refusing: container {info.name!r} is not ours "
            f"(name/label mismatch)"
        )
    if cid is not None and info.id != cid:
        raise Refusal(
            "refusing: container id changed under our name "
            f"(expected {cid[:12]}, found {info.id[:12]})"
        )


def stop_project_container(
    name: str = CONTAINER_NAME,
    run=_run_capture,  # injectable for tests
    immediate: bool = False,
    expected_id: str | None = None,
) -> dict:
    """Stop+remove ONLY our exact container. Raises Refusal otherwise.

    ``immediate=True`` (guard-trip path) uses ``docker kill``: prompt
    SIGKILL, because ``docker stop --time`` is not immediate on OOM.
    ``immediate=False`` (operator ``stop``) tries graceful stop first with
    SIGKILL fallback. Inspect failures raise without touching anything:
    an unverifiable target is never acted on.

    ``expected_id`` pins a previously identity-bound id through the final
    operation: if the re-inspect returns a different id under our name
    (replacement between verify and act), refuse instead of acting on the
    newcomer. All mutating operations address the id (immutable), never
    the name, so at worst a mid-flight removal surfaces as a docker
    error -- never an action on the wrong container.
    """
    if name != CONTAINER_NAME:
        raise Refusal(f"refusing to stop foreign container {name!r}")
    info = inspect_exact(name, run)
    if info is None:
        return {"name": name, "action": "absent"}
    _assert_ours(info)
    if expected_id is not None and info.id != expected_id:
        raise Refusal(
            "refusing: container id changed between verify and stop "
            f"(expected {expected_id[:12]}, found {info.id[:12]}): "
            "possible same-name replacement"
        )
    cid = info.id
    if immediate:
        proc = run(["docker", "kill", cid], DOCKER_OP_TIMEOUT_S)
        if proc.returncode != 0:
            raise LaunchError(
                f"could not kill container {name!r}: "
                f"{(proc.stderr or '').strip()[:200]}"
            )
    else:
        proc = run(
            ["docker", "stop", "--time", str(STOP_GRACE_S), cid],
            DOCKER_OP_TIMEOUT_S + STOP_GRACE_S,
        )
        if proc.returncode != 0:
            kproc = run(["docker", "kill", cid], DOCKER_OP_TIMEOUT_S)
            if kproc.returncode != 0:
                raise LaunchError(
                    f"could not stop container {name!r}: "
                    f"{(proc.stderr or '').strip()[:200]}"
                )
    rproc = run(["docker", "rm", cid], DOCKER_OP_TIMEOUT_S)
    if rproc.returncode != 0:
        raise LaunchError(
            f"stopped but could not remove {name!r}: "
            f"{(rproc.stderr or '').strip()[:200]}"
        )
    return {
        "name": name,
        "action": "killed+removed" if immediate else "stopped+removed",
    }


# ---------------------------------------------------------------------------
# Preflight: guard, port ownership, stale container
# ---------------------------------------------------------------------------

def check_port_free(port: int = HOST_PORT) -> None:
    """Refuse if 127.0.0.1:port already answers (foreign server?) or if the
    check itself is inconclusive. A successful connect means SOMETHING is
    there -- never probe it further, just refuse."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect((HOST_BIND, port))
    except ConnectionRefusedError:
        return  # free: nothing listening on host loopback
    except OSError as exc:
        raise LaunchError(
            f"port check inconclusive (fail closed): {exc}"
        ) from exc
    else:
        raise LaunchError(
            f"preflight refused: {HOST_BIND}:{port} already answers -- "
            "a foreign server may own the port; never start over it"
        )
    finally:
        sock.close()


def preflight(
    guard: Guard | None = None,
    run=_run_capture,  # injectable for tests
    port: int = HOST_PORT,
) -> dict:
    """Host checks that must pass BEFORE any start. Raises on any failure.

    * guard must report SAFE right now (metric failures fail closed);
    * captures the oom_kill baseline for the coming run;
    * 127.0.0.1:port must refuse connections (no foreign server);
    * no previous project container may exist -- explicit ``stop``
      required (never auto-removed, so a coordinator crash can never
      cascade into touching the wrong thing); an UNVERIFIABLE stale state
      (inspect error) also refuses.
    """
    guard = guard or Guard()
    verdict = guard.check_now()
    if not verdict.safe:
        raise LaunchError(
            "preflight refused: " + "; ".join(verdict.violations)
        )
    assert verdict.sample is not None
    guard.capture_oom_baseline()
    check_port_free(port)
    stale = inspect_exact(run=run)
    if stale is not None:
        raise LaunchError(
            f"preflight refused: previous container {CONTAINER_NAME!r} "
            f"(state={stale.status}) still exists; run `stop` explicitly "
            "before starting (never auto-removed)"
        )
    cache = resolve_snapshot_cache()
    return {
        "mem_available_gib": round(
            verdict.sample.mem_available_bytes / 1024**3, 2
        ),
        "psi_full_avg10": verdict.sample.psi_full_avg10,
        "oom_baseline": guard.baseline_oom,
        "snapshot": str(cache.host_snapshot),
        "shards": cache.shard_count,
    }


# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------

def _probe_urlopen(url: str, timeout_s: float):
    """Open a readiness URL, adding the vLLM bearer token when configured."""
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}
        )
        return urllib.request.urlopen(request, timeout=timeout_s)
    return urllib.request.urlopen(url, timeout=timeout_s)


def _probe_ready(timeout_s: float = READY_PROBE_TIMEOUT_S) -> bool:
    try:
        with _probe_urlopen(
            f"http://{HOST_BIND}:{HOST_PORT}/health", timeout_s
        ) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _probe_model_present(timeout_s: float = READY_PROBE_TIMEOUT_S) -> bool:
    """True iff /v1/models on host loopback lists exactly our MODEL_ID.

    Proves readiness belongs to OUR server, not a foreign process that
    happened to answer /health on the port (defense against readiness
    misattribution after the preflight port check).
    """
    try:
        with _probe_urlopen(
            f"http://{HOST_BIND}:{HOST_PORT}/v1/models", timeout_s
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        ids = {
            item.get("id")
            for item in payload.get("data", [])
            if isinstance(item, dict)
        }
        return MODEL_ID in ids
    except Exception:
        return False


def _spawn_detached(argv: list[str], popen) -> tuple[threading.Thread, dict]:
    """Start ``docker run -d`` in a thread so a hanging daemon can never
    wedge the supervisor past guard trips. Returns (thread, holder)."""
    holder: dict = {}

    def _target() -> None:
        try:
            proc = popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=sanitized_subprocess_env(),
                shell=False,
            )
            out, err = proc.communicate(timeout=DOCKER_START_TIMEOUT_S)
            holder["rc"] = proc.returncode
            holder["out"] = out
            holder["err"] = err
        except Exception as exc:  # hang/timeout/spawn failure
            holder["error"] = exc

    thread = threading.Thread(target=_target, name="docker-run", daemon=True)
    thread.start()
    return thread, holder


def _kill_ours_by_verified_id(
    cid: str, run, report: dict, why: str
) -> None:
    """Kill+remove by a previously identity-bound id. Records the action."""
    info = inspect_exact(run=run)  # re-verify name+label before acting
    if info is None:
        raise LaunchError(
            f"container {CONTAINER_NAME!r} vanished before kill ({why}); "
            "treating as failure, never restarting"
        )
    _assert_ours(info, cid)
    result = stop_project_container(run=run, immediate=True, expected_id=cid)
    report["stop"] = result
    report["stop_reason"] = why


def launch(
    confirm_host_risk: bool = False,
    startup_timeout_s: float = STARTUP_TIMEOUT_S,
    guard: Guard | None = None,
    popen=None,  # injectable; defaults to subprocess.Popen
    run=_run_capture,  # injectable for tests
    probe=_probe_ready,
    probe_models=_probe_model_present,
    monitor_poll_s: float | None = None,  # override for tests
    port: int = HOST_PORT,  # preflight port-ownership check
    settle_grace_s: float = SETTLE_GRACE_S,  # late-start recheck window
) -> dict:
    """Start the container under dual supervision; blocks until failure.

    HARD-DISABLED: raises RealStartDisabled unconditionally, before any
    preflight, subprocess, or container action -- even with
    confirm_host_risk=True. The flag records operator intent; it does not
    and cannot enable a start while the OOM-coverage requirement is
    unverified. The supervision machinery below it is retained (and
    tested) for a future independently verified mechanism, but no exposed
    path can reach it today.
    """
    if not _REAL_START_ENABLED:
        raise RealStartDisabled(
            "real model/container start is DISABLED: no independently "
            "verified mechanism meets the agreed immediate-stop-on-OOM "
            "guard for application/CUDA OOM paths (host oom_kill does not "
            "count them; only one vLLM log-and-reraise path is known). "
            "The model is NOT ready for use. "
            "confirm_host_risk=True does not enable a start."
        )
    if not confirm_host_risk:
        raise Refusal(
            "refusing to start the real container without --confirm-host-risk "
            "(operator must independently assess host risk first)"
        )
    guard = guard or Guard()
    report = preflight(guard, run=run, port=port)
    violation_event = threading.Event()
    monitor = GuardMonitor(
        guard=guard,
        poll_interval_s=(
            POLL_INTERVAL_S if monitor_poll_s is None else monitor_poll_s
        ),
        on_violation=lambda reasons: violation_event.set(),
    )
    cache = resolve_snapshot_cache()
    docker_argv = build_docker_argv(cache)
    monitor.start()  # coverage starts BEFORE docker start (boot surge)
    # An unsafe FIRST sample must refuse before docker ever runs: start()
    # returns after the first sample even when it fires.
    if monitor.fired:
        monitor.stop()
        raise LaunchError(
            f"preflight unsafe at monitor start, docker never ran: "
            f"{monitor.failure}"
        )
    popen = popen or subprocess.Popen
    started_at = time.monotonic()
    run_thread, holder = _spawn_detached(docker_argv, popen)
    cid: str | None = None
    verified_id: str | None = None
    try:
        # --- phase 1: obtain the container id without wedging on hangs ---
        deadline = started_at + DOCKER_START_TIMEOUT_S
        while run_thread.is_alive() and time.monotonic() < deadline:
            if violation_event.is_set() or monitor.fired:
                break
            run_thread.join(START_POLL_SLICE_S)
        if violation_event.is_set() or monitor.fired:
            # Breach DURING docker start: the container may already exist
            # (host watchdog absent), may appear LATE (after our check),
            # or may never have been created. Settle, then report.
            _settle_started_container(
                run, run_thread, holder, report, settle_grace_s,
                "guard trip during start",
            )
            exc = LaunchError(
                f"guard tripped during container start; stopped: "
                f"{monitor.failure or 'see guard verdict'}"
            )
            exc.report = report  # type: ignore[attr-defined]
            raise exc
        if run_thread.is_alive():
            _settle_started_container(
                run, run_thread, holder, report, settle_grace_s,
                "docker run hung",
            )
            exc = LaunchError(
                f"docker run hung > {DOCKER_START_TIMEOUT_S}s (fail closed)"
            )
            exc.report = report  # type: ignore[attr-defined]
            raise exc
        if holder.get("error") is not None:
            raise LaunchError(
                f"docker run failed to complete: {holder['error']}"
            )
        if holder.get("rc") != 0:
            raise LaunchError(
                f"docker run failed rc={holder.get('rc')}: "
                f"{str(holder.get('err') or '').strip()[:300]}"
            )
        out = (holder.get("out") or "").strip()
        cid = out.splitlines()[-1].strip() if out else None
        if not cid:
            raise LaunchError("docker run returned no container id")
        # Bind identity: the FIRST scoped inspect must show our exact
        # name+label AND the id docker just returned.
        info = inspect_exact(run=run)
        if info is None:
            raise LaunchError(
                "container id returned but no such container exists "
                "(fail closed)"
            )
        _assert_ours(info, cid)
        verified_id = info.id
        report["container_id"] = verified_id

        # --- phase 2: supervise until failure/interrupt (never returns) ---
        ready = False
        inspect_fails = 0
        health_fails = 0
        last_reprobe = 0.0
        while True:
            if violation_event.is_set() or monitor.fired:
                _kill_ours_by_verified_id(
                    verified_id, run, report, "guard trip"
                )
                raise LaunchError(
                    f"guard tripped; container killed: "
                    f"{monitor.failure or 'see guard verdict'}"
                )
            if not monitor.alive or monitor.heartbeat_stale():
                _kill_ours_by_verified_id(
                    verified_id, run, report, "monitor dead/stale"
                )
                raise LaunchError(
                    "guard monitor dead/stale (fail closed); killed"
                )
            try:
                info = inspect_exact(run=run)
                inspect_fails = 0
            except InspectError as exc:
                inspect_fails += 1
                if inspect_fails >= INSPECT_MAX_CONSECUTIVE_FAILURES:
                    # Identity unverifiable: do NOT blind-kill by name.
                    # The in-container watchdog still enforces the guard.
                    report["identity"] = f"unverifiable: {exc}"
                    raise LaunchError(
                        f"container identity unverifiable x{inspect_fails} "
                        f"(fail closed, no blind kill; in-container watchdog "
                        f"still guards id {verified_id[:12]}): {exc}"
                    ) from exc
                _sleep_sliced(1.0, violation_event)
                continue
            if info is None:
                raise LaunchError(
                    "container vanished (no-such-container) outside our "
                    "control; treating as failure, never restarting"
                )
            _assert_ours(info, verified_id)
            if info.status != "running":
                raise LaunchError(
                    f"container exited (status={info.status} "
                    f"rc={info.exit_code}); never restarting"
                )
            now = time.monotonic()
            if not ready:
                if probe():
                    if probe_models():
                        ready = True
                        report["ready_at_s"] = round(now - started_at, 1)
                    # /health ok but wrong model (or not yet listed):
                    # keep waiting, do NOT declare ready.
                if not ready and now - started_at > startup_timeout_s:
                    _kill_ours_by_verified_id(
                        verified_id, run, report, "startup timeout"
                    )
                    raise LaunchError(
                        f"startup timeout ({startup_timeout_s}s) before "
                        f"verified ready; killed"
                    )
            else:
                if now - last_reprobe >= READY_REPROBE_S:
                    last_reprobe = now
                    if probe() and probe_models():
                        health_fails = 0
                    else:
                        health_fails += 1
                        if health_fails >= HEALTH_MAX_CONSECUTIVE_FAILURES:
                            _kill_ours_by_verified_id(
                                verified_id, run, report, "health lost"
                            )
                            raise LaunchError(
                                "serving health lost "
                                f"(x{health_fails} consecutive, fail "
                                "closed); killed"
                            )
            _sleep_sliced(POLL_INTERVAL_S, violation_event)
    except (LaunchError, Refusal, KeyboardInterrupt):
        raise
    except Exception as exc:
        raise LaunchError(f"supervisor failure (fail closed): {exc}") from exc
    finally:
        monitor.stop()
        # Best-effort cleanup of OUR container only; never blind. A
        # container that already exited is reaped (rm), not killed.
        try:
            if verified_id is not None:
                try:
                    info = inspect_exact(run=run)
                except InspectError:
                    info = None
                if info is not None:
                    _assert_ours(info, verified_id)
                    if info.status == "running":
                        stop_project_container(run=run, immediate=True)
                        report.setdefault(
                            "stop", {"action": "killed+removed"}
                        )
                    else:
                        try:
                            rproc = run(
                                ["docker", "rm", verified_id],
                                DOCKER_OP_TIMEOUT_S,
                            )
                            if rproc.returncode == 0:
                                report.setdefault(
                                    "stop", {"action": "removed-exited"}
                                )
                        except Exception:
                            pass
            elif cid is None:
                _kill_started_or_named(run, report, "cleanup")
        except (LaunchError, Refusal):
            pass  # original exception takes precedence; report keeps why


def _note(report: dict, text: str) -> None:
    """Append-only report notes: cleanup/settle findings must survive."""
    prior = report.get("cleanup")
    report["cleanup"] = f"{prior}; {text}" if prior else text


def _kill_started_or_named(run, report: dict, why: str) -> None:
    """Best-effort kill when the container id was never bound.

    Only acts on a container that re-verifies as ours by exact name+label
    right now; an unverifiable state is REPORTED, not acted on. (The
    in-container watchdog, when the container exists, still guards it.)
    """
    try:
        info = inspect_exact(run=run)
    except InspectError as exc:
        _note(report, f"unverifiable, not touched ({why}): {exc}")
        return
    if info is None:
        _note(report, f"absent ({why})")
        return
    if info.name != CONTAINER_NAME or info.label != PROJECT_LABEL_VALUE:
        _note(report, f"foreign container present, not touched ({why})")
        return
    stop_project_container(run=run, immediate=True)
    _note(report, f"killed+removed ({why})")


def _settle_started_container(
    run, run_thread: threading.Thread, holder: dict, report: dict,
    grace_s: float, why: str,
) -> None:
    """Settle the phase-1 race: attempt an immediate verified kill, then
    allow a bounded grace for a LATE container creation and re-check.

    A ``docker run -d`` accepted by the daemon can still materialize AFTER
    an absent check. The grace re-check NARROWS that window; it cannot
    prevent post-grace materialization and makes no fail-closed claim
    about late starts -- any remainder is REPORTED (verify read-only,
    e.g. ``docker ps --filter name=qwen38-fp4-local-serve``). Any
    container that does exist carries the in-container watchdog, which
    guards it without the host. NOTE: this path is currently unreachable
    (launch() is hard-disabled); it is retained for a future verified
    mechanism, not as an executable claim today.
    """
    _kill_started_or_named(run, report, why)
    run_thread.join(timeout=max(0.0, grace_s))
    if run_thread.is_alive():
        _note(
            report,
            "start thread still hung after grace: a late container "
            "creation remains possible -- verify read-only "
            "(docker ps --filter name=qwen38-fp4-local-serve); any created "
            "container still carries the in-container watchdog",
        )
        return
    if holder.get("error") is None and holder.get("rc") == 0:
        out = (holder.get("out") or "").strip()
        late_cid = out.splitlines()[-1].strip() if out else None
        if late_cid:
            # The container materialized AFTER the first check: bind and
            # kill it now that its id is known.
            try:
                info = inspect_exact(run=run)
            except InspectError as exc:
                _note(
                    report,
                    f"late container {late_cid[:12]} unverifiable: {exc}",
                )
                return
            if info is None:
                return
            try:
                _assert_ours(info, late_cid)
            except Refusal as exc:
                _note(report, f"late container refused: {exc}")
                return
            stop_project_container(run=run, immediate=True)
            _note(report, "late container killed+removed")


def _sleep_sliced(seconds: float, event: threading.Event) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if event.is_set():
            return
        time.sleep(min(START_POLL_SLICE_S, deadline - time.monotonic()))


# ---------------------------------------------------------------------------
# CLI (inspection only; real launch remains disabled)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail-closed serving preparation for Qwen3.8-FP4-Local."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight", help="host checks only; starts nothing")
    sub.add_parser("build", help="print the exact docker argv as JSON")
    lp = sub.add_parser(
        "launch",
        help="DISABLED: always fails closed (no verified OOM mechanism)",
    )
    lp.add_argument(
        "--confirm-host-risk",
        action="store_true",
        help="records operator intent only; does NOT enable a start",
    )
    lp.add_argument(
        "--startup-timeout-s", type=float, default=STARTUP_TIMEOUT_S
    )
    args = ap.parse_args(argv)

    if args.cmd == "preflight":
        try:
            print(json.dumps(preflight(), indent=2))
            return 0
        except (LaunchError, GuardError, Refusal) as exc:
            print(f"PREFLIGHT REFUSED: {exc}", file=sys.stderr)
            return 1
    if args.cmd == "build":
        try:
            cache = resolve_snapshot_cache()
            print(json.dumps(build_docker_argv(cache), indent=2))
            return 0
        except (LaunchError, GuardError, Refusal) as exc:
            print(f"BUILD REFUSED: {exc}", file=sys.stderr)
            return 1
    if args.cmd == "launch":
        started = datetime.now(timezone.utc).isoformat()
        try:
            report = launch(
                confirm_host_risk=args.confirm_host_risk,
                startup_timeout_s=args.startup_timeout_s,
            )
            report["started_at"] = started
            print(json.dumps(report, indent=2))
            return 0
        except RealStartDisabled as exc:
            print(f"LAUNCH DISABLED (fail closed): {exc}", file=sys.stderr)
            return 1
        except (LaunchError, GuardError, Refusal) as exc:
            print(f"LAUNCH ABORTED (fail closed): {exc}", file=sys.stderr)
            return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
