"""PID-1 watchdog for the single 6 GiB startup trial.

This file is mounted read-only and intentionally standalone for Python in
the pinned serving image. Any healthy HTTP response raises the floor to
18 GiB; only the expected model ID counts as verified readiness.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import urllib.error
import urllib.request

STARTUP_MIN_BYTES = 6 * 1024**3
OPERATING_MIN_BYTES = 18 * 1024**3
PSI_MAX_PCT = 10.0
POLL_S = 0.5
HEARTBEAT_PATH = os.environ.get(
    "QWEN38_TRIAL_HEARTBEAT", "/watch/trial-heartbeat"
)
HEARTBEAT_MAX_AGE_S = 4.0
EXPECTED_MODEL_ID = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
OOM_PATTERNS = (
    re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
    re.compile(r"CUDA(?: error)?:?\s*out of memory", re.IGNORECASE),
    re.compile(r"not enough GPU memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
)


class WatchdogError(RuntimeError):
    pass


def log(message: str) -> None:
    print(f"[qwen38-trial-watchdog] {message}", file=sys.stderr, flush=True)


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        raise WatchdogError(f"cannot read {path}: {exc}") from exc


def _sample(baseline_oom: int, ready: bool) -> tuple[bool, str]:
    mem_match = re.search(r"^MemAvailable:\s+(\d+)\s+kB\s*$", _read("/proc/meminfo"), re.M)
    psi_match = re.search(
        r"^full\s+avg10=([0-9]+(?:\.[0-9]+)?)\b",
        _read("/proc/pressure/memory"), re.M,
    )
    oom_match = re.search(r"^oom_kill\s+(\d+)\s*$", _read("/proc/vmstat"), re.M)
    if not mem_match or not psi_match or not oom_match:
        raise WatchdogError("host metric parse failed")
    mem = int(mem_match.group(1)) * 1024
    psi = float(psi_match.group(1))
    oom = int(oom_match.group(1))
    floor = OPERATING_MIN_BYTES if ready else STARTUP_MIN_BYTES
    errors = []
    if mem < floor:
        errors.append(f"MemAvailable {mem / 1024**3:.2f} GiB < {floor / 1024**3:.0f} GiB")
    if not 0 <= psi <= 100:
        errors.append(f"PSI invalid {psi}")
    elif psi >= PSI_MAX_PCT:
        errors.append(f"memory PSI full avg10 {psi:.2f}% >= {PSI_MAX_PCT:.1f}%")
    if oom != baseline_oom:
        errors.append(f"host kernel oom_kill changed {baseline_oom} -> {oom}")
    if errors:
        return False, "; ".join(errors)
    return True, f"mem={mem / 1024**3:.2f}GiB psi_full={psi:.2f}% oom={oom} floor={floor / 1024**3:.0f}GiB"


def _probe_urlopen(url: str, timeout_s: float):
    """Use the API key that vLLM received, if the operator configured one."""
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {api_key}"}
        )
        return urllib.request.urlopen(request, timeout=timeout_s)
    return urllib.request.urlopen(url, timeout=timeout_s)


def _probe_server() -> tuple[bool, str | None, bool]:
    try:
        with _probe_urlopen("http://127.0.0.1:8000/health", 0.75) as response:
            if response.status != 200:
                return False, None, False
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return False, None, False
    try:
        with _probe_urlopen("http://127.0.0.1:8000/v1/models", 0.75) as response:
            result = json.loads(response.read().decode("utf-8"))
        models = result.get("data", [])
        if len(models) != 1 or not isinstance(models[0], dict):
            return True, None, False
        model_id = models[0].get("id")
        if not isinstance(model_id, str) or not model_id:
            return True, None, False
        return True, model_id, model_id != EXPECTED_MODEL_ID
    except (OSError, ValueError, TypeError, urllib.error.URLError):
        return True, None, False


def _oom_log(line: str) -> bool:
    return any(pattern.search(line) for pattern in OOM_PATTERNS)


def _heartbeat_alive(now: float | None = None) -> bool:
    try:
        age = (time.time() if now is None else now) - Path(HEARTBEAT_PATH).stat().st_mtime
    except OSError:
        return False
    return 0 <= age <= HEARTBEAT_MAX_AGE_S


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[0] != "/opt/nvidia/nvidia_entrypoint.sh":
        log("invalid server command; server was not started")
        return 3
    if not (os.path.isfile(argv[0]) and os.access(argv[0], os.X_OK)):
        log("image entrypoint missing; server was not started")
        return 3
    command = list(argv)
    if "VLLM" in command:
        executable = shutil.which("vllm")
        if not executable:
            log("vllm executable missing; server was not started")
            return 3
        command[command.index("VLLM")] = executable
    if not _heartbeat_alive():
        log("host supervisor heartbeat missing/stale; server was not started")
        return 3
    try:
        baseline_match = re.search(r"^oom_kill\s+(\d+)\s*$", _read("/proc/vmstat"), re.M)
        if not baseline_match:
            raise WatchdogError("kernel OOM baseline missing")
        baseline_oom = int(baseline_match.group(1))
        safe, message = _sample(baseline_oom, False)
        if not safe:
            raise WatchdogError(message)
    except WatchdogError as exc:
        log(f"preflight failed closed: {exc}")
        return 3

    try:
        child = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        log(f"server start failed: {exc}")
        return 3
    oom_seen = threading.Event()
    log_error: list[str] = []

    def pump() -> None:
        try:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                if _oom_log(line):
                    oom_seen.set()
        except Exception as exc:
            log_error.append(str(exc))

    reader = threading.Thread(target=pump, name="trial-vllm-log-reader", daemon=True)
    reader.start()
    term = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: term.set())
    signal.signal(signal.SIGINT, lambda *_: term.set())
    served_model_id: str | None = None
    health_ready = False
    log(f"server pid={child.pid}; one-shot; no restart")
    while True:
        if not _heartbeat_alive():
            log("host supervisor heartbeat missing/stale; killing server")
            child.kill()
            child.wait()
            return 2
        if term.is_set():
            # Do not wait through an unmonitored grace period. The operator
            # asked to stop the one-shot trial; terminate promptly.
            child.kill()
            child.wait()
            return 143
        if oom_seen.is_set():
            log("detectable application/CUDA OOM log; killing server, no restart")
            child.kill()
            child.wait()
            return 2
        if log_error:
            log(f"log monitor failed closed: {log_error[0]}")
            child.kill()
            child.wait()
            return 2
        rc = child.poll()
        if rc is not None:
            log(f"server exited rc={rc}; no restart")
            return rc
        try:
            found_health, found_id, mismatch = _probe_server()
            health_ready = health_ready or found_health
            served_model_id = served_model_id or found_id
            ready = health_ready
            safe, message = _sample(baseline_oom, ready)
            log(message)
            if not safe:
                log(f"guard breach; killing server: {message}")
                child.kill()
                child.wait()
                return 2
            if mismatch:
                log(
                    "health is ready but model ID mismatches; "
                    f"got {found_id!r}, expected {EXPECTED_MODEL_ID!r}"
                )
                child.kill()
                child.wait()
                return 2
            if served_model_id == EXPECTED_MODEL_ID:
                log(f"verified served model ID: {served_model_id}")
        except WatchdogError as exc:
            log(f"monitor failure; killing server: {exc}")
            child.kill()
            child.wait()
            return 2
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
