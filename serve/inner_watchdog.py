#!/usr/bin/env python3
"""PID-1 self-guard for the FP4-Local serving container.

This file is bind-mounted READ-ONLY into the container at
``/watch/qwen38-watchdog.py`` and runs as the container's PID 1
(``--entrypoint python3``, CMD starts with this path). It is intended to
monitor host metrics inside the container if the host supervisor dies.
Hangs or simultaneous failures of both monitoring layers are not ruled out:

  * ``/proc/meminfo``, ``/proc/pressure/memory`` and ``/proc/vmstat`` are
    NOT PID- or mount-namespaced by default Docker; inside the container
    they normally report HOST values. Unreadable or unparseable data
    fails closed (exit 2/3), and zero/constrained values trip the
    thresholds -- but masked values that merely LOOK healthy cannot be
    distinguished from inside. Real-boot validation must confirm
    host-global visibility (prerequisite evidence so far: the pinned
    image under a 2 GiB cgroup saw host-global meminfo/PSI/oom_kill and
    had memory.events; python3 runs as PID 1 with and without GPU; a
    30 s child of an exiting Python PID 1 stopped its container in
    0.24 s). Prerequisites only -- not OOM coverage.
  * Graceful SIGTERM (operator ``stop``) is forwarded to the server, and
    sampling continues during the grace window; an observed breach then
    SIGKILLs the child (exit 2). Sampling still has gaps and latency.
  * The watchdog spawns the server ONCE, supervises it, and on ANY guard
    breach SIGKILLs the child and exits. It NEVER restarts the server.
  * ANY server exit (including exit 0, including an application/CUDA OOM
    crash) ends the container with the child's code; the host supervisor
    treats every exit as failure and never restarts either.

Exit codes:
  2 -- guard breach or guard metric failure (child SIGKILLed first);
  3 -- configuration/preflight failure (server NEVER started);
  else -- the server child's own exit code (no restart in any case).

Server command convention: the watchdog's own argv after this script is
the server command, ``[entrypoint.sh, "VLLM", "serve", ...]``. The literal
element ``"VLLM"`` is resolved via PATH inside the container
(``shutil.which``); a missing binary fails closed with exit 3 instead of
relying on shell/entrypoint magic.

Limits mirrored here MUST equal serve/guard.py (enforced by tests):
  MEM_MIN_BYTES == guard.MEM_AVAILABLE_MIN_BYTES (18 GiB),
  PSI_MAX_PCT   == guard.PSI_FULL_AVG10_MAX_PCT (10.0).

``QWEN38_PROC_ROOT`` / ``QWEN38_POLL_S`` exist ONLY for automated tests
(synthetic /proc trees, faster polls). The launcher MUST NOT set them in
the container (its builder asserts their absence).

Only the Python 3.12 standard library is used. No imports from the
``serve`` package: this file must run standalone inside the image.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time

MEM_MIN_BYTES = 18 * 1024**3
PSI_MAX_PCT = 10.0
EXIT_BREACH = 2
EXIT_CONFIG = 3
TERM_GRACE_S = 25.0

PROC_ROOT = os.environ.get("QWEN38_PROC_ROOT", "/proc")
# Image entrypoint under test override. The launcher never sets this in
# the container (builder test asserts its absence), so in-container runs
# always use the real image path below.
ENTRYPOINT_SH = os.environ.get(
    "QWEN38_ENTRYPOINT_SH", "/opt/nvidia/nvidia_entrypoint.sh"
)
try:
    POLL_S = float(os.environ.get("QWEN38_POLL_S", "1.0"))
except ValueError:
    POLL_S = 1.0
if not POLL_S > 0:
    POLL_S = 1.0

VLLM_PLACEHOLDER = "VLLM"

_MEM_RE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB\s*$", re.MULTILINE)
_PSI_FULL_RE = re.compile(r"^full\s+avg10=([0-9]+(?:\.[0-9]+)?)\b", re.MULTILINE)
_OOM_RE = re.compile(r"^oom_kill\s+(\d+)\s*$", re.MULTILINE)


class WatchdogError(Exception):
    """Any preflight/metric/config failure. Always fail closed."""


def log(msg: str) -> None:
    print(f"[qwen38-watchdog] {msg}", file=sys.stderr, flush=True)


def _read(name: str) -> str:
    path = os.path.join(PROC_ROOT, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise WatchdogError(f"cannot read {path}: {exc}") from exc


def _parse_mem_kb(text: str) -> int:
    if not text:
        raise WatchdogError("empty meminfo")
    m = _MEM_RE.search(text)
    if not m:
        raise WatchdogError("MemAvailable line unparseable")
    return int(m.group(1)) * 1024


def _parse_psi(text: str) -> float:
    if not text:
        raise WatchdogError("empty pressure file")
    m = _PSI_FULL_RE.search(text)
    if not m:
        raise WatchdogError("memory PSI 'full avg10=' line unparseable")
    val = float(m.group(1))
    if not (0.0 <= val <= 100.0):
        raise WatchdogError(f"PSI avg10 out of range: {val!r}")
    return val


def _parse_oom(text: str) -> int:
    if not text:
        raise WatchdogError("empty vmstat")
    m = _OOM_RE.search(text)
    if not m:
        raise WatchdogError("oom_kill line unparseable")
    return int(m.group(1))


def check_now(baseline_oom: int) -> tuple[bool, tuple[str, ...]]:
    """Sample and evaluate. Raises WatchdogError on ANY metric failure."""
    mem = _parse_mem_kb(_read("meminfo"))
    psi = _parse_psi(_read("pressure/memory"))
    oom = _parse_oom(_read("vmstat"))
    problems: list[str] = []
    if mem < MEM_MIN_BYTES:
        problems.append(
            f"MemAvailable {mem / 1024**3:.2f} GiB "
            f"< minimum {MEM_MIN_BYTES / 1024**3:.0f} GiB"
        )
    if psi >= PSI_MAX_PCT:
        problems.append(
            f"memory PSI full avg10 {psi:.2f}% >= limit {PSI_MAX_PCT:.1f}%"
        )
    if oom != baseline_oom:
        # KERNEL oom_kill counter only; app/CUDA OOMs do not increment it.
        problems.append(
            f"host kernel oom_kill changed {baseline_oom} -> {oom}"
        )
    return (not problems, tuple(problems))


def resolve_server_argv(argv: list[str]) -> list[str]:
    """Validate the server command shape and resolve the vLLM binary."""
    if len(argv) < 3 or argv[0] != ENTRYPOINT_SH:
        raise WatchdogError(
            f"unexpected server argv shape (expected "
            f"[{ENTRYPOINT_SH!r}, 'VLLM', 'serve', ...]): {argv[:3]!r}"
        )
    if not (
        os.path.isfile(ENTRYPOINT_SH) and os.access(ENTRYPOINT_SH, os.X_OK)
    ):
        raise WatchdogError(
            f"image entrypoint {ENTRYPOINT_SH!r} missing or not executable"
        )
    out = list(argv)
    if VLLM_PLACEHOLDER in out:
        exe = shutil.which("vllm")
        if not exe:
            raise WatchdogError(
                "vllm executable not found on container PATH "
                "(refusing to rely on entrypoint magic)"
            )
        out[out.index(VLLM_PLACEHOLDER)] = exe
        log(f"resolved vllm executable: {exe}")
    return out


def main(argv: list[str]) -> int:
    start = time.monotonic()
    try:
        server_argv = resolve_server_argv(argv)
    except WatchdogError as exc:
        log(f"config failure (fail closed, server not started): {exc}")
        return EXIT_CONFIG
    try:
        baseline_oom = _parse_oom(_read("vmstat"))
    except WatchdogError as exc:
        log(f"oom baseline unreadable (fail closed): {exc}")
        return EXIT_CONFIG
    try:
        safe, problems = check_now(baseline_oom)
    except WatchdogError as exc:
        log(f"preflight metrics unreadable (fail closed): {exc}")
        return EXIT_CONFIG
    if not safe:
        log(f"host already unsafe, server not started: {'; '.join(problems)}")
        return EXIT_CONFIG

    try:
        child = subprocess.Popen(server_argv)
    except OSError as exc:
        log(f"server spawn failed (fail closed): {exc}")
        return EXIT_CONFIG
    log(f"server started as pid {child.pid}: {' '.join(server_argv)}")

    term_requested = False

    def _on_term(signum, frame):  # noqa: ARG001
        nonlocal term_requested
        term_requested = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    while True:
        if term_requested:
            log("SIGTERM received: forwarding to server; guard stays on")
            child.terminate()
            deadline = time.monotonic() + TERM_GRACE_S
            while child.poll() is None:
                # Sampling continues during the grace window: a
                # host supervisor may be absent exactly then, so a
                # violation here SIGKILLs immediately (exit 2) instead of
                # waiting out the grace period.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    gsafe, gproblems = check_now(baseline_oom)
                except WatchdogError as exc:
                    log(
                        "metric failure during SIGTERM grace "
                        f"(fail closed): {exc}; SIGKILLing server"
                    )
                    child.kill()
                    child.wait()
                    return EXIT_BREACH
                if not gsafe:
                    log(
                        "GUARD BREACH during SIGTERM grace: "
                        f"{'; '.join(gproblems)}; SIGKILLing server"
                    )
                    child.kill()
                    child.wait()
                    return EXIT_BREACH
                time.sleep(min(0.2, remaining))
            if child.poll() is None:
                child.kill()
            code = child.wait()
            log(f"server reaped rc={code} after SIGTERM")
            return 143
        rc = child.poll()
        if rc is not None:
            # ANY server exit ends the container. No restart, ever: a
            # crash-looping server under memory pressure amplifies harm.
            log(f"server exited rc={rc}; no restart (fail closed)")
            return rc
        try:
            safe, problems = check_now(baseline_oom)
        except WatchdogError as exc:
            log(f"metric failure (fail closed): {exc}; SIGKILLing server")
            child.kill()
            child.wait()
            return EXIT_BREACH
        if not safe:
            log(f"GUARD BREACH: {'; '.join(problems)}; SIGKILLing server")
            child.kill()
            child.wait()
            return EXIT_BREACH
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
