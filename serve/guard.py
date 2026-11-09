"""Fail-closed host safety guard for the FP4-Local serving preparation.

Agreed limits (MUST NOT be weakened):
  * host MemAvailable >= 18 GiB (below trips; exactly 18 is still safe);
  * memory PSI ``full`` avg10 >= 10.0 trips;
  * any change of the host KERNEL ``oom_kill`` counter vs the pre-start
    baseline -> immediate stop.

OOM coverage (do NOT over-claim): ``oom_kill`` counts only kills by the
kernel OOM killer. Application/CUDA OOM errors (e.g. out-of-memory raised
inside the process, NCCL/CUDA errors) do NOT increment it. Those paths
are covered only indirectly: real exhaustion usually moves
MemAvailable/PSI first (trip), a crashed server exits (the watchdog and
supervisor treat ANY server exit as failure and never restart), and lost
observability trips. A silent GPU-side exhaustion that moves none of
these is NOT covered; see serve/README.md.

Fail-closed contract:
  * Every read/parse failure raises :class:`GuardError` and is treated as
    a violation (never as "safe"). Callers MUST stop/shut-down on
    ``GuardError`` exactly as on a threshold breach.
  * Boundary rule: MemAvailable below 18 GiB trips (exactly 18 GiB is
    still safe); PSI full avg10 at or above 10.0 trips.

PSI sampling disclosure (read before relying on this guard):
  * The kernel reports PSI over fixed windows: avg10 / avg60 / avg300
    (10 s / 60 s / 300 s exponential averages). We evaluate ``full``
    ``avg10`` sampled every POLL_INTERVAL_S seconds (default 1.0 s).
  * ``avg10`` smooths over ~10 s: a sharp stall spike inside the window
    is under-reported, and a 1 s poll cannot observe sub-second events.
  * Sampling CANNOT guarantee catching memory exhaustion before the OOM
    killer acts, and CANNOT guarantee the 18 GiB minimum holds between
    polls or under all conditions. Observed on this host: MemAvailable
    fell 29.43 -> 14.39 GiB between two consecutive 10-second samples --
    a ~15 GiB change between samples; its timing inside the interval is
    unknown. This guard is a mitigation intended to stop OUR container
    when the host looks unsafe, not a reservation or prevention mechanism.
  * Rationale for avg10 (not avg60/avg300): avg60/avg300 react far too
    slowly for a startup surge; avg10 is the fastest window the kernel
    provides via /proc/pressure/*.

Scope of effect: the guard only produces verdicts. Two experimental
enforcers are designed to act on them: the in-container PID-1 watchdog
(serve/inner_watchdog.py) and the host supervisor (serve/launch.py). Their
combined behavior is not verified for the real model, and model starts are
disabled. This module never touches processes, containers, or services.

Only the Python 3.12 standard library is used.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

MEM_AVAILABLE_MIN_BYTES = 18 * 1024**3
PSI_FULL_AVG10_MAX_PCT = 10.0
POLL_INTERVAL_S = 1.0
HEARTBEAT_STALE_AFTER_S = 3 * POLL_INTERVAL_S

PSI_INTERVAL_CHOICE = "full avg10 (10-second kernel window)"
POLL_INTERVAL_DISCLOSURE = (
    "Guard evaluates memory PSI 'full avg10' sampled every "
    f"{POLL_INTERVAL_S:.1f}s. avg10 smooths over ~10s, so brief spikes are "
    "under-reported and sub-second events are invisible. Sampling cannot "
    "guarantee OOM is caught first nor that MemAvailable >= 18 GiB holds "
    "between polls or under all conditions; the guard mitigates by "
    "stopping this project's container promptly, it does not prevent "
    "host exhaustion."
)

DEFAULT_MEMINFO_PATH = "/proc/meminfo"
DEFAULT_PRESSURE_PATH = "/proc/pressure/memory"
DEFAULT_VMSTAT_PATH = "/proc/vmstat"


class GuardError(Exception):
    """Any guard input/monitor failure. ALWAYS fail closed (unsafe)."""


@dataclass(frozen=True)
class Sample:
    mem_available_bytes: int
    psi_full_avg10: float
    oom_kill: int


@dataclass(frozen=True)
class Verdict:
    safe: bool
    violations: tuple[str, ...] = ()
    sample: Sample | None = None


_MEM_RE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB\s*$", re.MULTILINE)
_PSI_FULL_RE = re.compile(r"^full\s+avg10=([0-9]+(?:\.[0-9]+)?)\b", re.MULTILINE)
_OOM_RE = re.compile(r"^oom_kill\s+(\d+)\s*$", re.MULTILINE)


def parse_memavailable_bytes(text: str) -> int:
    """Parse MemAvailable from /proc/meminfo text. Raises GuardError."""
    if not text:
        raise GuardError("empty meminfo")
    m = _MEM_RE.search(text)
    if not m:
        raise GuardError("MemAvailable line not found/unparseable in meminfo")
    try:
        kb = int(m.group(1))
    except ValueError as exc:
        raise GuardError(f"MemAvailable value not an integer: {exc}") from exc
    if kb < 0:
        raise GuardError("MemAvailable negative")
    return kb * 1024


def parse_psi_full_avg10(text: str) -> float:
    """Parse the ``full avg10=`` value from /proc/pressure/memory text."""
    if not text:
        raise GuardError("empty pressure file")
    m = _PSI_FULL_RE.search(text)
    if not m:
        raise GuardError("memory PSI 'full avg10=' line not found/unparseable")
    try:
        val = float(m.group(1))
    except ValueError as exc:
        raise GuardError(f"PSI avg10 not a float: {exc}") from exc
    if not (0.0 <= val <= 100.0):
        raise GuardError(f"PSI avg10 out of range 0..100: {val!r}")
    return val


def parse_oom_kill(text: str) -> int:
    """Parse the ``oom_kill`` counter from /proc/vmstat text."""
    if not text:
        raise GuardError("empty vmstat")
    m = _OOM_RE.search(text)
    if not m:
        raise GuardError("oom_kill line not found/unparseable in vmstat")
    try:
        val = int(m.group(1))
    except ValueError as exc:
        raise GuardError(f"oom_kill value not an integer: {exc}") from exc
    if val < 0:
        raise GuardError("oom_kill negative")
    return val


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise GuardError(f"cannot read {path}: {exc}") from exc


@dataclass
class Guard:
    """Threshold evaluator with injectable metric sources (for tests)."""

    meminfo_path: str = DEFAULT_MEMINFO_PATH
    pressure_path: str = DEFAULT_PRESSURE_PATH
    vmstat_path: str = DEFAULT_VMSTAT_PATH
    mem_min_bytes: int = MEM_AVAILABLE_MIN_BYTES
    psi_max_pct: float = PSI_FULL_AVG10_MAX_PCT
    baseline_oom: int | None = None

    def capture_oom_baseline(self) -> int:
        """Record the current oom_kill counter. Raises GuardError on failure."""
        base = parse_oom_kill(_read_text(self.vmstat_path))
        self.baseline_oom = base
        return base

    def sample(self) -> Sample:
        """Take one host sample. Raises GuardError on ANY failure."""
        return Sample(
            mem_available_bytes=parse_memavailable_bytes(
                _read_text(self.meminfo_path)
            ),
            psi_full_avg10=parse_psi_full_avg10(_read_text(self.pressure_path)),
            oom_kill=parse_oom_kill(_read_text(self.vmstat_path)),
        )

    def check(self, sample: Sample) -> Verdict:
        """Evaluate a sample. Pure; never raises on values (only types)."""
        violations: list[str] = []
        if sample.mem_available_bytes < self.mem_min_bytes:
            violations.append(
                "MemAvailable "
                f"{sample.mem_available_bytes / 1024**3:.2f} GiB "
                f"< minimum {self.mem_min_bytes / 1024**3:.0f} GiB"
            )
        if sample.psi_full_avg10 >= self.psi_max_pct:
            violations.append(
                f"memory PSI full avg10 {sample.psi_full_avg10:.2f}% "
                f">= limit {self.psi_max_pct:.1f}%"
            )
        if self.baseline_oom is not None and sample.oom_kill != self.baseline_oom:
            # Any change (increase expected; a decrease would mean counter
            # reset/reboot -- equally reason to stop and re-assess). This
            # is the KERNEL oom_kill counter only; application/CUDA OOM
            # errors do not increment it (see module docstring).
            violations.append(
                f"host kernel oom_kill changed {self.baseline_oom} -> "
                f"{sample.oom_kill}: immediate stop"
            )
        return Verdict(
            safe=not violations, violations=tuple(violations), sample=sample
        )

    def check_now(self) -> Verdict:
        """Sample and evaluate. Metric failures -> unsafe Verdict (fail closed).

        Unlike check(), this never raises for metric problems: the
        GuardError is converted into a violation so callers that only look
        at ``safe`` still stop.
        """
        try:
            return self.check(self.sample())
        except GuardError as exc:
            return Verdict(
                safe=False,
                violations=(f"guard metric failure (fail closed): {exc}",),
                sample=None,
            )


@dataclass
class GuardMonitor:
    """Background sampler covering startup AND steady state.

    The monitor MUST be started BEFORE the container starts so the memory
    surge of startup itself is covered, not just post-ready serving.

    Any threshold breach OR any internal/metric failure invokes
    ``on_violation`` exactly once with the list of reasons. A dead monitor
    thread or stale heartbeat means "unknown" -> the owner (launch.py
    watchdog) must treat that as failure and stop the container: a crashed
    monitor never implies safety.

    ``on_violation`` is called from the monitor thread; it must be quick
    and thread-safe (launch.py only sets an event there).
    """

    guard: Guard
    poll_interval_s: float = POLL_INTERVAL_S
    on_violation: callable = field(default=lambda reasons: None)  # type: ignore[assignment]

    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _fired: threading.Event = field(default_factory=threading.Event, init=False)
    _first_sample: threading.Event = field(
        default_factory=threading.Event, init=False
    )
    last_sample_at: float = field(default=0.0, init=False)
    last_verdict: Verdict | None = field(default=None, init=False)
    internal_error: str | None = field(default=None, init=False)

    def start(self, first_sample_timeout_s: float = 10.0) -> None:
        """Start sampling and block until the first sample lands.

        This removes the startup race where a supervisor could judge the
        monitor "stale" before it ever sampled. Raises GuardError
        (fail closed) if no first sample arrives in time.
        """
        if self._thread is not None:
            raise GuardError("monitor already started")
        self._stop.clear()
        self._first_sample.clear()
        self._thread = threading.Thread(
            target=self._run, name="guard-monitor", daemon=True
        )
        self._thread.start()
        if not self._first_sample.wait(timeout=first_sample_timeout_s):
            self.stop()
            raise GuardError(
                "guard monitor produced no first sample in time (fail closed)"
            )

    def _fire_once(self, reasons: tuple[str, ...]) -> None:
        if not self._fired.is_set():
            self._fired.set()
            try:
                self.on_violation(list(reasons))
            except Exception as exc:  # callback must not kill the signal
                self.internal_error = f"on_violation callback raised: {exc}"

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                verdict = self.guard.check_now()
                self.last_verdict = verdict
                self.last_sample_at = time.monotonic()
                self._first_sample.set()
                if not verdict.safe:
                    self._fire_once(verdict.violations)
                    return
                self._stop.wait(self.poll_interval_s)
        except Exception as exc:  # fail closed on monitor-internal failure
            self.internal_error = f"monitor crashed: {exc}"
            self._fire_once((f"guard monitor failure (fail closed): {exc}",))

    def stop(self) -> None:
        self._stop.set()
        self._first_sample.set()  # unblock a start() still waiting
        if self._thread is not None:
            # The thread waits on _stop, so it wakes promptly; the timeout
            # is only a deadlock backstop, independent of the poll interval.
            self._thread.join(timeout=10.0)

    @property
    def fired(self) -> bool:
        return self._fired.is_set()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def heartbeat_stale(self, now: float | None = None) -> bool:
        """True when no sample has been recorded within the stale horizon."""
        if self.last_sample_at == 0.0:
            return True
        now = time.monotonic() if now is None else now
        return (now - self.last_sample_at) > HEARTBEAT_STALE_AFTER_S

    @property
    def failure(self) -> str | None:
        """Human-readable failure if tripped or internally broken."""
        if self.last_verdict is not None and not self.last_verdict.safe:
            return "; ".join(self.last_verdict.violations)
        return self.internal_error
