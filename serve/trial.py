"""One-shot, two-threshold startup trial policy and supervised runner.

This path is deliberately separate from the normal launcher. It is useful
only for the explicitly approved single startup experiment; it does not
enable routine serving or auto-restart.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue

from . import launch
from .baseline import IMAGE_PINNED, MODEL_ID
from .guard import Guard, Sample

STARTUP_MIN_GIB = 6
OPERATING_MIN_GIB = 18
PSI_MAX_PCT = 10.0
TRIAL_CONTEXT = 32768
TRIAL_SEQUENCES = 1
TRIAL_STARTUP_TIMEOUT_S = 1800.0
START_PENDING_TIMEOUT_S = 15.0
TRIAL_POLL_S = 0.5
OOM_PATTERNS = (
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUDA(?: error)?:?\s*out of memory", re.IGNORECASE),
    re.compile(r"not enough GPU memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
)


class TrialFailure(RuntimeError):
    """A trial failure that must end the one-shot run without restart."""


def is_application_oom(line: str) -> bool:
    return any(pattern.search(line) for pattern in OOM_PATTERNS)


@dataclass
class TrialGuard:
    """Enforce 6 GiB only before readiness, then 18 GiB permanently."""

    oom_baseline: int
    ready_seen: bool = False
    startup_min_gib: int = STARTUP_MIN_GIB

    @property
    def minimum_gib(self) -> int:
        return OPERATING_MIN_GIB if self.ready_seen else self.startup_min_gib

    def check(self, sample: Sample | None, *, ready: bool) -> None:
        if sample is None:
            raise TrialFailure("host guard sample unavailable")
        # Readiness is a one-way transition. Evaluate the transition sample
        # under 18 GiB before the server can be reported ready to the caller.
        if ready:
            self.ready_seen = True
        problems: list[str] = []
        minimum = self.minimum_gib * 1024**3
        if sample.mem_available_bytes < minimum:
            problems.append(
                f"MemAvailable {sample.mem_available_bytes / 1024**3:.2f} GiB "
                f"< minimum {self.minimum_gib} GiB"
            )
        if sample.psi_full_avg10 >= PSI_MAX_PCT:
            problems.append(
                f"memory PSI full avg10 {sample.psi_full_avg10:.2f}% "
                f">= limit {PSI_MAX_PCT:.1f}%"
            )
        if sample.oom_kill != self.oom_baseline:
            problems.append(
                f"host kernel oom_kill changed {self.oom_baseline} -> "
                f"{sample.oom_kill}"
            )
        if problems:
            raise TrialFailure("; ".join(problems))


def _probe_urlopen(url: str, timeout_s: float):
    """Probe the local API, authenticating when VLLM_API_KEY is configured."""
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}
        )
        return urllib.request.urlopen(request, timeout=timeout_s)
    return urllib.request.urlopen(url, timeout=timeout_s)


def _probe_server(timeout_s: float = 1.0) -> tuple[bool, str | None, bool]:
    try:
        with _probe_urlopen(
            f"http://{launch.HOST_BIND}:{launch.HOST_PORT}/health", timeout_s
        ) as response:
            if response.status != 200:
                return False, None, False
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return False, None, False
    try:
        with _probe_urlopen(
            f"http://{launch.HOST_BIND}:{launch.HOST_PORT}/v1/models", timeout_s
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        models = body.get("data", [])
        if len(models) != 1 or not isinstance(models[0], dict):
            return True, None, False
        model_id = models[0].get("id")
        if not isinstance(model_id, str) or not model_id:
            return True, None, False
        return True, model_id, model_id != MODEL_ID
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return True, None, False


def _model_ready(timeout_s: float = 1.0) -> str | None:
    health_ready, model_id, mismatch = _probe_server(timeout_s)
    return model_id if health_ready and not mismatch else None


def _trial_argv(
    cache: launch.ResolvedCache, watchdog_path: Path, heartbeat_path: Path
) -> list[str]:
    argv = launch.build_docker_argv(cache, watchdog_host=watchdog_path)
    for flag, value in (
        ("--max-model-len", str(TRIAL_CONTEXT)),
        ("--max-num-seqs", str(TRIAL_SEQUENCES)),
    ):
        try:
            argv[argv.index(flag) + 1] = value
        except (ValueError, IndexError) as exc:
            raise TrialFailure(f"baseline missing {flag}: {exc}") from exc
    if "--served-model-name" in argv:
        argv[argv.index("--served-model-name") + 1] = MODEL_ID
    else:
        try:
            model_index = argv.index("--max-model-len")
        except ValueError as exc:
            raise TrialFailure("baseline missing --max-model-len") from exc
        argv[model_index:model_index] = ["--served-model-name", MODEL_ID]
    try:
        image_index = argv.index(IMAGE_PINNED)
    except ValueError as exc:
        raise TrialFailure("pinned image missing from docker argv") from exc
    argv[image_index:image_index] = [
        "-e", "QWEN38_TRIAL_HEARTBEAT=/watch/trial-heartbeat",
        "-v",
        f"{heartbeat_path.resolve()}:/watch/trial-heartbeat:ro",
    ]
    if argv.index(IMAGE_PINNED) < argv.index("-e", 2):
        raise TrialFailure("heartbeat environment must precede image")
    if "--restart" in argv:
        raise TrialFailure("restart policy is forbidden for one-shot trial")
    return argv


def _trial_docker_create(argv: list[str]) -> list[str]:
    """Convert the pinned build argv into `docker create`, preserving flags."""
    if len(argv) < 4 or argv[:2] != ["docker", "run"]:
        raise TrialFailure("unexpected docker argv shape")
    created = list(argv)
    created[1] = "create"
    if "-d" in created:
        created.remove("-d")
    return created


class TrialRunner:
    """Single-use Docker trial supervisor. Dependencies are injectable in tests."""

    def __init__(
        self,
        *,
        guard: Guard | None = None,
        run=None,
        popen=None,
        server_probe=_probe_server,
        evidence_root: str | Path = ".opencode/diagnostics",
        startup_timeout_s: float = TRIAL_STARTUP_TIMEOUT_S,
        poll_s: float = TRIAL_POLL_S,
        startup_min_gib: int = STARTUP_MIN_GIB,
        argv_builder=_trial_argv,
        watchdog_path: Path | None = None,
        outcome_metadata: dict | None = None,
    ):
        self.startup_min_gib = startup_min_gib
        self.argv_builder = argv_builder
        self.watchdog_path = watchdog_path
        self.outcome_metadata = dict(outcome_metadata or {})
        self.guard = guard or Guard(mem_min_bytes=startup_min_gib * 1024**3)
        self.run = run or launch._run_capture
        self.popen = popen or subprocess.Popen
        self.server_probe = server_probe
        self.evidence_root = Path(evidence_root)
        self.startup_timeout_s = startup_timeout_s
        self.poll_s = poll_s
        self._used = False

    def execute(self) -> dict:
        if self._used:
            raise TrialFailure("trial runner is one-shot and has already run")
        self._used = True
        stamp = datetime.now(timezone.utc).strftime("trial-%Y%m%d-%H%M%S")
        evidence = self.evidence_root / stamp
        evidence.mkdir(parents=True, mode=0o700, exist_ok=False)
        metrics_path = evidence / "metrics.log"
        outcome_path = evidence / "outcome.json"
        metrics = metrics_path.open("x", encoding="utf-8", buffering=1)
        cid: str | None = None
        proc = None
        logs: Queue[str | None] = Queue(maxsize=256)
        log_thread: threading.Thread | None = None
        stop = threading.Event()
        heartbeat = evidence / "heartbeat"
        heartbeat.touch(mode=0o600, exist_ok=False)
        heartbeat_thread: threading.Thread | None = None
        ready = False
        ready_announced = False
        served_model_id: str | None = None
        reason = "unknown"
        log_path = evidence / "container.log"
        log_file = log_path.open("xb", buffering=0)
        try:
            preflight = launch.preflight(self.guard, run=self.run)
            cache = launch.resolve_snapshot_cache()
            host_watch = self.watchdog_path or Path(__file__).with_name("trial_watchdog.py")
            argv = self.argv_builder(cache, host_watch, heartbeat)
            create_argv = _trial_docker_create(argv)
            if self.guard.baseline_oom is None:
                raise TrialFailure("preflight did not capture kernel OOM baseline")
            preflight["trial_oom_baseline"] = self.guard.baseline_oom
            trial_guard = TrialGuard(self.guard.baseline_oom,
                                     startup_min_gib=self.startup_min_gib)
            precreate_sample = self.guard.sample()
            self._write_sample(
                metrics, precreate_sample, ready=False, phase="precreate"
            )
            trial_guard.check(precreate_sample, ready=False)
            heartbeat_thread = threading.Thread(
                target=self._heartbeat, args=(heartbeat, stop),
                name="qwen38-trial-heartbeat", daemon=True,
            )
            heartbeat_thread.start()
            created = self.run(create_argv, launch.DOCKER_OP_TIMEOUT_S)
            if created.returncode != 0:
                raise TrialFailure(
                    f"docker create failed rc={created.returncode}: "
                    f"{created.stderr.strip()[:300]}"
                )
            cid = created.stdout.strip().splitlines()[-1]
            if not cid:
                raise TrialFailure("docker create returned no container id")
            info = launch.inspect_exact(run=self.run)
            if info is None:
                raise TrialFailure("created container identity not found")
            launch._assert_ours(info, cid)
            if info.status != "created":
                raise TrialFailure(
                    f"new container unexpectedly in state {info.status!r}"
                )
            # Start attached to a log sink only after identity is bound.
            proc = self.popen(
                ["docker", "start", "--attach", cid],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=launch.sanitized_subprocess_env(),
                bufsize=0,
            )
            log_thread = threading.Thread(
                target=self._copy_logs,
                args=(proc.stdout, log_file, logs, stop),
                name="qwen38-trial-logs",
                daemon=True,
            )
            log_thread.start()
            started = time.monotonic()
            while True:
                code = proc.poll()
                info = launch.inspect_exact(run=self.run)
                if info is None:
                    raise TrialFailure("container disappeared; no restart")
                launch._assert_ours(info, cid)
                for line in self._drain_logs(logs):
                    if is_application_oom(line):
                        raise TrialFailure(
                            f"detectable application/CUDA OOM log: {line[:240]}"
                        )
                if info.status == "created":
                    if code is not None:
                        raise TrialFailure(
                            f"docker attach exited rc={code} before container "
                            "entered running state"
                        )
                    if time.monotonic() - started > START_PENDING_TIMEOUT_S:
                        raise TrialFailure(
                            "container remained created past start timeout"
                        )
                    sample = self.guard.sample()
                    self._write_sample(
                        metrics, sample, ready=False, phase="start_pending"
                    )
                    trial_guard.check(sample, ready=False)
                    time.sleep(self.poll_s)
                    continue
                if info.status != "running":
                    raise TrialFailure(
                        f"container exited state={info.status} "
                        f"rc={info.exit_code}; no restart"
                    )
                if code is not None:
                    raise TrialFailure(
                        f"docker start/attach exited rc={code} while container "
                        "was running; no restart"
                    )
                health_ready, observed_model_id, mismatch = self.server_probe()
                if observed_model_id and served_model_id is None:
                    served_model_id = observed_model_id
                detected_ready = health_ready
                sample = self.guard.sample()
                self._write_sample(metrics, sample, detected_ready)
                trial_guard.check(sample, ready=detected_ready)
                ready = trial_guard.ready_seen
                if mismatch:
                    raise TrialFailure(
                        "server health is ready but returned unexpected model ID "
                        f"{observed_model_id!r}; expected {MODEL_ID!r}"
                    )
                if ready and served_model_id == MODEL_ID and not ready_announced:
                    readiness_path = evidence / "readiness.json"
                    payload = json.dumps(
                        {
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "model_id": served_model_id,
                            "threshold_gib": OPERATING_MIN_GIB,
                        },
                        indent=2,
                    ) + "\n"
                    with readiness_path.open("x", encoding="utf-8") as result:
                        result.write(payload)
                        result.flush()
                        os.fsync(result.fileno())
                    ready_announced = True
                    reason = "readiness_observed; supervised until interruption or failure"
                if (
                    not ready_announced
                    and time.monotonic() - started > self.startup_timeout_s
                ):
                    raise TrialFailure("model readiness timeout")
                time.sleep(self.poll_s)
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stop.set()
            if cid is not None:
                # The attach client can exit while its container keeps
                # running. Always re-inspect and stop only this bound ID.
                try:
                    current = launch.inspect_exact(run=self.run)
                    if current is not None:
                        launch._assert_ours(current, cid)
                        if current.status == "running":
                            launch.stop_project_container(
                                run=self.run, immediate=True, expected_id=cid
                            )
                        else:
                            removed = self.run(
                                ["docker", "rm", cid], launch.DOCKER_OP_TIMEOUT_S
                            )
                            if removed.returncode != 0:
                                raise TrialFailure(
                                    "could not remove verified exited trial "
                                    f"container {cid[:12]}: "
                                    f"{(removed.stderr or '').strip()[:200]}"
                                )
                except Exception as exc:
                    reason += f"; scoped_stop_error={exc}"
                try:
                    proc.wait(timeout=launch.DOCKER_OP_TIMEOUT_S)
                except Exception:
                    pass
            if log_thread is not None:
                log_thread.join(timeout=5)
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2)
            try:
                final_logs = self._drain_logs(logs, allow_eof=True)
                if any(is_application_oom(line) for line in final_logs):
                    reason += "; detectable application/CUDA OOM in final logs"
            except TrialFailure as exc:
                reason += f"; final log monitor failure: {exc}"
            self._record(metrics, self.guard, phase="exit", ready=ready)
            metrics.close()
            log_file.close()
            outcome = {
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "reason": reason,
                        "ready": ready,
                        "threshold_gib": (
                            OPERATING_MIN_GIB if ready else self.startup_min_gib
                        ),
                        "container_id": cid,
                        "restart": False,
                    }
            outcome.update(self.outcome_metadata)
            outcome_path.write_text(
                json.dumps(
                    outcome,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _heartbeat(path: Path, stop: threading.Event) -> None:
        while not stop.wait(0.5):
            os.utime(path, None)

    @staticmethod
    def _copy_logs(stream, destination, events: Queue, stop: threading.Event):
        oom_event_sent = False
        try:
            while True:
                chunk = stream.readline()
                if not chunk:
                    break
                # Persist the complete stream before publishing actionable
                # events. Normal diagnostics must never back-pressure the
                # log reader and lose the final evidence tail.
                destination.write(chunk)
                text = chunk.decode("utf-8", errors="replace").rstrip()
                if not oom_event_sent and is_application_oom(text):
                    events.put_nowait(text)
                    oom_event_sent = True
        except Exception as exc:
            events.put_nowait(f"[trial log monitor failure] {exc}")
        finally:
            events.put_nowait(None)

    @staticmethod
    def _drain_logs(
        events: Queue[str | None], *, allow_eof: bool = False
    ) -> list[str]:
        output: list[str] = []
        while True:
            try:
                line = events.get_nowait()
            except Empty:
                return output
            if line is None:
                if not allow_eof:
                    raise TrialFailure("container log stream ended unexpectedly")
            elif line.startswith("[trial log monitor failure]"):
                raise TrialFailure(line)
            else:
                output.append(line)

    @staticmethod
    def _record(metrics, guard: Guard, *, phase: str, ready: bool) -> None:
        try:
            sample = guard.sample()
        except Exception as exc:
            metrics.write(
                f"{time.time():.3f} phase={phase} ready={int(ready)} "
                f"sample_error={type(exc).__name__}:{exc}\n"
            )
            return
        TrialRunner._write_sample(metrics, sample, ready, phase=phase)

    @staticmethod
    def _write_sample(metrics, sample: Sample, ready: bool, phase="monitor"):
        metrics.write(
            f"{time.time():.3f} phase={phase} ready={int(ready)} "
            f"MemAvailable={sample.mem_available_bytes} "
            f"psi_full_avg10={sample.psi_full_avg10:.2f} "
            f"oom_kill={sample.oom_kill}\n"
        )


def main() -> int:
    try:
        TrialRunner().execute()
    except KeyboardInterrupt:
        print("one-shot model trial stopped by operator", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"one-shot model trial failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
