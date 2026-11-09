"""Unit tests for serve.guard. Stdlib only; temp files under /tmp/opencode."""

from __future__ import annotations

import os
import tempfile
import time
import unittest

from serve.guard import (
    HEARTBEAT_STALE_AFTER_S,
    Guard,
    GuardError,
    GuardMonitor,
    Sample,
    parse_memavailable_bytes,
    parse_oom_kill,
    parse_psi_full_avg10,
)

GIB = 1024**3


def _tmpdir() -> str:
    os.makedirs("/tmp/opencode", exist_ok=True)
    return tempfile.mkdtemp(prefix="qwen38-guard-", dir="/tmp/opencode")


def _write(path: str, text: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _meminfo(mem_gib: float) -> str:
    return (
        "MemTotal:       127598836 kB\n"
        f"MemAvailable:    {int(mem_gib * GIB / 1024)} kB\n"
    )


def _pressure(full_avg10: float) -> str:
    return (
        f"some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
        f"full avg10={full_avg10:.2f} avg60=0.00 avg300=0.00 total=1\n"
    )


def _vmstat(oom: int) -> str:
    return f"nr_free_pages 1\noom_kill {oom}\n"


class ParseTests(unittest.TestCase):
    def test_memavailable_ok(self):
        self.assertEqual(parse_memavailable_bytes(_meminfo(20.0)), 20 * GIB)

    def test_memavailable_missing_is_error(self):
        with self.assertRaises(GuardError):
            parse_memavailable_bytes("MemTotal: 1 kB\n")

    def test_memavailable_empty_is_error(self):
        with self.assertRaises(GuardError):
            parse_memavailable_bytes("")

    def test_memavailable_garbage_is_error(self):
        with self.assertRaises(GuardError):
            parse_memavailable_bytes("MemAvailable: lots kB\n")

    def test_psi_ok(self):
        self.assertAlmostEqual(parse_psi_full_avg10(_pressure(3.25)), 3.25)

    def test_psi_missing_full_line_is_error(self):
        with self.assertRaises(GuardError):
            parse_psi_full_avg10("some avg10=0.00 avg60=0 avg300=0 total=1\n")

    def test_psi_out_of_range_is_error(self):
        with self.assertRaises(GuardError):
            parse_psi_full_avg10(_pressure(101.0))

    def test_oom_ok(self):
        self.assertEqual(parse_oom_kill(_vmstat(7)), 7)

    def test_oom_missing_is_error(self):
        with self.assertRaises(GuardError):
            parse_oom_kill("nr_free_pages 1\n")

    def test_oom_empty_is_error(self):
        with self.assertRaises(GuardError):
            parse_oom_kill("")


def _guard_in(tmp: str, mem_gib=64.0, psi=0.0, oom=0) -> Guard:
    mem = _write(os.path.join(tmp, "meminfo"), _meminfo(mem_gib))
    prs = _write(os.path.join(tmp, "pressure"), _pressure(psi))
    vm = _write(os.path.join(tmp, "vmstat"), _vmstat(oom))
    return Guard(meminfo_path=mem, pressure_path=prs, vmstat_path=vm)


class CheckTests(unittest.TestCase):
    def test_safe_sample(self):
        g = _guard_in(_tmpdir())
        g.capture_oom_baseline()
        v = g.check_now()
        self.assertTrue(v.safe, v.violations)
        self.assertEqual(v.violations, ())

    def test_mem_below_18_trips(self):
        g = _guard_in(_tmpdir(), mem_gib=17.9)
        v = g.check_now()
        self.assertFalse(v.safe)
        self.assertTrue(any("MemAvailable" in x for x in v.violations))

    def test_mem_exactly_18_is_safe(self):
        g = _guard_in(_tmpdir(), mem_gib=18.0)
        v = g.check_now()
        self.assertTrue(v.safe, v.violations)

    def test_prior_20k_low_would_trip(self):
        # The observed ~6.12 GiB low from the 20K/MTP3 run must trip.
        g = _guard_in(_tmpdir(), mem_gib=6.12)
        v = g.check_now()
        self.assertFalse(v.safe)

    def test_psi_at_10_trips(self):
        g = _guard_in(_tmpdir(), psi=10.0)
        v = g.check_now()
        self.assertFalse(v.safe)
        self.assertTrue(any("PSI" in x for x in v.violations))

    def test_psi_below_10_safe(self):
        g = _guard_in(_tmpdir(), psi=9.99)
        v = g.check_now()
        self.assertTrue(v.safe, v.violations)

    def test_oom_increase_trips(self):
        tmp = _tmpdir()
        g = _guard_in(tmp, oom=3)
        g.capture_oom_baseline()
        _write(g.vmstat_path, _vmstat(4))
        v = g.check_now()
        self.assertFalse(v.safe)
        self.assertTrue(any("oom_kill" in x for x in v.violations))

    def test_oom_decrease_also_trips(self):
        tmp = _tmpdir()
        g = _guard_in(tmp, oom=5)
        g.capture_oom_baseline()
        _write(g.vmstat_path, _vmstat(2))  # counter reset/reboot: re-assess
        v = g.check_now()
        self.assertFalse(v.safe)

    def test_metric_failure_is_unsafe_not_error(self):
        g = Guard(
            meminfo_path="/tmp/opencode/does-not-exist-meminfo",
            pressure_path="/tmp/opencode/does-not-exist-pressure",
            vmstat_path="/tmp/opencode/does-not-exist-vmstat",
        )
        v = g.check_now()  # must not raise; must be unsafe (fail closed)
        self.assertFalse(v.safe)
        self.assertTrue(any("fail closed" in x for x in v.violations))
        self.assertIsNone(v.sample)

    def test_sample_raises_on_missing(self):
        g = Guard(meminfo_path="/tmp/opencode/does-not-exist-meminfo")
        with self.assertRaises(GuardError):
            g.sample()

    def test_baseline_capture_failure_raises(self):
        g = Guard(vmstat_path="/tmp/opencode/does-not-exist-vmstat")
        with self.assertRaises(GuardError):
            g.capture_oom_baseline()


class MonitorTests(unittest.TestCase):
    def test_monitor_covers_and_fires_on_violation(self):
        tmp = _tmpdir()
        g = _guard_in(tmp)
        fired: list = []
        mon = GuardMonitor(
            guard=g, poll_interval_s=0.05, on_violation=fired.append
        )
        mon.start()
        try:
            time.sleep(0.2)
            self.assertFalse(mon.fired)
            self.assertTrue(mon.alive)
            self.assertFalse(mon.heartbeat_stale())
            # Host degrades: MemAvailable collapses below the minimum.
            _write(g.meminfo_path, _meminfo(4.0))
            deadline = time.monotonic() + 5.0
            while not mon.fired and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(mon.fired, "monitor did not fire on violation")
            self.assertTrue(any("MemAvailable" in r for r in fired[0]))
            self.assertIsNotNone(mon.failure)
        finally:
            mon.stop()

    def test_monitor_callback_exception_does_not_lose_signal(self):
        tmp = _tmpdir()
        g = _guard_in(tmp, mem_gib=1.0)  # already violating

        def bad_cb(reasons):
            raise RuntimeError("callback boom")

        mon = GuardMonitor(
            guard=g, poll_interval_s=0.05, on_violation=bad_cb
        )
        mon.start()
        try:
            deadline = time.monotonic() + 5.0
            while not mon.fired and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertTrue(mon.fired)
            self.assertIsNotNone(mon.internal_error)
        finally:
            mon.stop()

    def test_heartbeat_stale_before_first_sample(self):
        g = _guard_in(_tmpdir())
        mon = GuardMonitor(guard=g, poll_interval_s=60.0)
        self.assertTrue(mon.heartbeat_stale())  # never sampled: unknown
        self.assertGreater(HEARTBEAT_STALE_AFTER_S, 0)

    def test_start_without_first_sample_raises_fail_closed(self):
        g = Guard(
            meminfo_path="/tmp/opencode/does-not-exist-meminfo",
            pressure_path="/tmp/opencode/does-not-exist-pressure",
            vmstat_path="/tmp/opencode/does-not-exist-vmstat",
        )
        # Metric failure -> unsafe verdict, which fires immediately; start
        # must still return (signal recorded), not hang.
        fired: list = []
        mon = GuardMonitor(
            guard=g, poll_interval_s=0.05, on_violation=fired.append
        )
        mon.start()
        try:
            self.assertTrue(mon.fired)
            self.assertTrue(fired)
        finally:
            mon.stop()

    def test_start_blocks_until_first_sample(self):
        tmp = _tmpdir()
        g = _guard_in(tmp)
        mon = GuardMonitor(guard=g, poll_interval_s=60.0)
        mon.start(first_sample_timeout_s=10.0)
        try:
            # After start() returns, a sample exists: not stale.
            self.assertFalse(mon.heartbeat_stale())
            self.assertTrue(mon.alive)
        finally:
            mon.stop()


if __name__ == "__main__":
    unittest.main()
