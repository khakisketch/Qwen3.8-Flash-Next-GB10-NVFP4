"""Unit tests for serve.inner_watchdog. Real watchdog subprocesses against
synthetic /proc trees and fake server children under /tmp/opencode.

These tests exercise watchdog behavior without a host supervisor using
synthetic processes. They do not prove the real container's orphan race
or complete OOM coverage.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest

from serve import inner_watchdog as w
from serve import guard as g

WATCHDOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "serve",
    "inner_watchdog.py",
)
WATCHDOG_PATH = os.path.normpath(WATCHDOG_PATH)


def _tmpdir() -> str:
    os.makedirs("/tmp/opencode", exist_ok=True)
    return tempfile.mkdtemp(prefix="qwen38-wd-", dir="/tmp/opencode")


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _proc_tree(tmp: str, mem_gib=64.0, psi=0.0, oom=0) -> str:
    root = os.path.join(tmp, "proc")
    _write(
        os.path.join(root, "meminfo"),
        "MemTotal:       127598836 kB\n"
        f"MemAvailable:    {int(mem_gib * 1024**3 / 1024)} kB\n",
    )
    _write(
        os.path.join(root, "pressure", "memory"),
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
        f"full avg10={psi:.2f} avg60=0.00 avg300=0.00 total=1\n",
    )
    _write(os.path.join(root, "vmstat"), f"nr_free_pages 1\noom_kill {oom}\n")
    return root


def _fake_entrypoint(tmp: str) -> str:
    """Executable stub standing in for the image entrypoint script."""
    path = os.path.join(tmp, "entrypoint.sh")
    _write(
        path,
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "os.execvp(sys.argv[1], sys.argv[1:])\n",
    )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


def _env(proc_root: str, extra: dict | None = None) -> dict:
    env = {
        "PATH": "/usr/bin:/bin",
        "QWEN38_PROC_ROOT": proc_root,
        "QWEN38_POLL_S": "0.05",
    }
    if extra:
        env.update(extra)
    return env


def _wait_path(path: str, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.05)
    return False


class MirrorTests(unittest.TestCase):
    def test_thresholds_mirror_guard(self):
        self.assertEqual(w.MEM_MIN_BYTES, g.MEM_AVAILABLE_MIN_BYTES)
        self.assertEqual(w.PSI_MAX_PCT, g.PSI_FULL_AVG10_MAX_PCT)

    def test_exit_codes_distinct(self):
        self.assertEqual({w.EXIT_BREACH, w.EXIT_CONFIG}, {2, 3})


class WatchdogBehaviorTests(unittest.TestCase):
    def test_breach_sigkills_child_and_exits_2(self):
        """Host collapses mid-run: child SIGKILLed, watchdog exits 2.

        No supervisor exists in this test -- the watchdog alone enforces.
        """
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        marker = os.path.join(tmp, "child-started")
        pidfile = os.path.join(tmp, "child.pid")
        entry = _fake_entrypoint(tmp)
        child_code = (
            "import os,time;"
            f"open({pidfile!r},'w').write(str(os.getpid()));"
            f"open({marker!r},'w').write('up');"
            "time.sleep(60)"
        )
        proc = subprocess.Popen(
            [sys.executable, WATCHDOG_PATH, entry,
             sys.executable, "-c", child_code],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            env=_env(root, {"QWEN38_ENTRYPOINT_SH": entry}),
        )
        try:
            # Sequencing: child must be UP before the breach is injected.
            self.assertTrue(
                _wait_path(marker), "server child never started"
            )
            with open(pidfile) as fh:
                child_pid = int(fh.read())
            # Host collapses: MemAvailable far below the minimum.
            _write(
                os.path.join(root, "meminfo"),
                "MemTotal:       127598836 kB\n"
                f"MemAvailable:    {int(2.0 * 1024**3 / 1024)} kB\n",
            )
            rc = proc.wait(timeout=15)
            err = proc.stderr.read()
            proc.stderr.close()
            self.assertEqual(rc, 2, f"stderr: {err[:500]}")
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)  # child is dead: SIGKILLed, reaped
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_unsafe_at_start_never_spawns_server(self):
        tmp = _tmpdir()
        root = _proc_tree(tmp, mem_gib=4.0)
        marker = os.path.join(tmp, "child-started")
        entry = _fake_entrypoint(tmp)
        proc = subprocess.run(
            [sys.executable, WATCHDOG_PATH, entry,
             sys.executable, "-c",
             f"open({marker!r},'w').write('up')"],
            capture_output=True, text=True, timeout=30,
            env=_env(root, {"QWEN38_ENTRYPOINT_SH": entry}),
        )
        self.assertEqual(proc.returncode, 3)
        self.assertFalse(
            os.path.exists(marker), "server must never spawn when unsafe"
        )
        self.assertIn("not started", proc.stderr)

    def test_child_crash_exits_with_code_no_restart(self):
        """Simulated application/CUDA OOM (child exits 1): no restart."""
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        log = os.path.join(tmp, "spawns.log")
        entry = _fake_entrypoint(tmp)
        child_code = (
            f"open({log!r},'a').write('spawn\\n');"
            "raise SystemExit(1)"
        )
        proc = subprocess.run(
            [sys.executable, WATCHDOG_PATH, entry,
             sys.executable, "-c", child_code],
            capture_output=True, text=True, timeout=30,
            env=_env(root, {"QWEN38_ENTRYPOINT_SH": entry}),
        )
        self.assertEqual(proc.returncode, 1)
        with open(log) as fh:
            self.assertEqual(
                fh.read().count("spawn"), 1,
                "server must be spawned exactly once (no restart)",
            )

    def test_metric_failure_midrun_kills_and_exits_2(self):
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        marker = os.path.join(tmp, "child-started")
        entry = _fake_entrypoint(tmp)
        proc = subprocess.Popen(
            [sys.executable, WATCHDOG_PATH, entry,
             sys.executable, "-c",
             f"open({marker!r},'w').write('up');"
             "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            env=_env(root, {"QWEN38_ENTRYPOINT_SH": entry}),
        )
        try:
            self.assertTrue(_wait_path(marker))
            # Metrics vanish mid-run (masked /proc): fail closed.
            os.remove(os.path.join(root, "pressure", "memory"))
            rc = proc.wait(timeout=15)
            proc.stderr.close()
            self.assertEqual(rc, 2)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_sigterm_forwards_gracefully(self):
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        marker = os.path.join(tmp, "child-started")
        pidfile = os.path.join(tmp, "child.pid")
        entry = _fake_entrypoint(tmp)
        proc = subprocess.Popen(
            [sys.executable, WATCHDOG_PATH, entry,
             sys.executable, "-c",
             f"import os,time;"
             f"open({pidfile!r},'w').write(str(os.getpid()));"
             f"open({marker!r},'w').write('up');"
             "time.sleep(60)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            env=_env(root, {"QWEN38_ENTRYPOINT_SH": entry}),
        )
        try:
            self.assertTrue(_wait_path(marker))
            with open(pidfile) as fh:
                child_pid = int(fh.read())
            proc.terminate()
            rc = proc.wait(timeout=40)
            proc.stderr.close()
            self.assertEqual(rc, 143)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_breach_during_sigterm_grace_kills_immediately(self):
        """A violation inside the SIGTERM grace window SIGKILLs at once.

        The child ignores SIGTERM (would otherwise outlive the 25 s
        grace); the host flips unsafe mid-grace. Enforcement must be
        continuous: exit 2 promptly, not exit 143 after the grace.
        """
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        marker = os.path.join(tmp, "child-started")
        pidfile = os.path.join(tmp, "child.pid")
        entry = _fake_entrypoint(tmp)
        child_code = (
            "import os,signal,time;"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
            f"open({pidfile!r},'w').write(str(os.getpid()));"
            f"open({marker!r},'w').write('up');"
            "time.sleep(60)"
        )
        proc = subprocess.Popen(
            [sys.executable, WATCHDOG_PATH, entry,
             sys.executable, "-c", child_code],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            env=_env(root, {"QWEN38_ENTRYPOINT_SH": entry}),
        )
        try:
            self.assertTrue(_wait_path(marker))
            with open(pidfile) as fh:
                child_pid = int(fh.read())
            proc.terminate()  # watchdog enters its grace window
            time.sleep(0.3)  # let it get there; host still safe
            _write(
                os.path.join(root, "meminfo"),
                "MemTotal:       127598836 kB\n"
                f"MemAvailable:    {int(2.0 * 1024**3 / 1024)} kB\n",
            )
            rc = proc.wait(timeout=15)
            err = proc.stderr.read()
            proc.stderr.close()
            self.assertEqual(rc, 2, f"stderr: {err[:500]}")
            self.assertIn("during SIGTERM grace", err)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_vllm_placeholder_resolves_explicitly(self):
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        bindir = os.path.join(tmp, "bin")
        os.makedirs(bindir)
        marker = os.path.join(tmp, "vllm-started")
        fake_vllm = os.path.join(bindir, "vllm")
        _write(
            fake_vllm,
            f"#!{sys.executable}\n"
            f"open({marker!r},'w').write('up')\n",
        )
        os.chmod(fake_vllm, os.stat(fake_vllm).st_mode | stat.S_IXUSR)
        entry = _fake_entrypoint(tmp)
        env = _env(
            root,
            {"QWEN38_ENTRYPOINT_SH": entry, "PATH": bindir},
        )
        proc = subprocess.run(
            [sys.executable, WATCHDOG_PATH, entry,
             "VLLM", "serve", "--help"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(os.path.exists(marker))
        self.assertIn(f"resolved vllm executable: {fake_vllm}", proc.stderr)

    def test_missing_vllm_binary_fails_closed(self):
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        bindir = os.path.join(tmp, "emptybin")
        os.makedirs(bindir)
        entry = _fake_entrypoint(tmp)
        env = _env(
            root,
            {"QWEN38_ENTRYPOINT_SH": entry, "PATH": bindir},
        )
        proc = subprocess.run(
            [sys.executable, WATCHDOG_PATH, entry,
             "VLLM", "serve", "model"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        self.assertEqual(proc.returncode, 3)
        self.assertIn("vllm executable not found", proc.stderr)

    def test_unexpected_argv_shape_fails_closed(self):
        tmp = _tmpdir()
        root = _proc_tree(tmp)
        proc = subprocess.run(
            [sys.executable, WATCHDOG_PATH, "whatever"],
            capture_output=True, text=True, timeout=30, env=_env(root),
        )
        self.assertEqual(proc.returncode, 3)


if __name__ == "__main__":
    unittest.main()
