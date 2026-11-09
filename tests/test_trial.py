from __future__ import annotations

import unittest
import json
import threading
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from serve.trial import (
    TrialGuard,
    TrialFailure,
    _trial_docker_create,
    _trial_argv,
    is_application_oom,
    _model_ready,
    _probe_server,
    TrialRunner,
)
from serve import trial_watchdog as watchdog
from serve import launch
from serve.guard import Sample
from serve.baseline import IMAGE_PINNED, MODEL_ID
from pathlib import Path


def sample(mem_gib: float, psi: float = 0.0, oom: int = 0) -> Sample:
    return Sample(
        mem_available_bytes=int(mem_gib * 1024**3),
        psi_full_avg10=psi,
        oom_kill=oom,
    )


class TrialGuardTests(unittest.TestCase):
    def test_startup_uses_six_gib_then_ready_transition_requires_eighteen(self):
        guard = TrialGuard(oom_baseline=4)
        self.assertEqual(guard.minimum_gib, 6)
        guard.check(sample(7.0, oom=4), ready=False)
        with self.assertRaisesRegex(TrialFailure, "18 GiB"):
            guard.check(sample(17.9, oom=4), ready=True)
        self.assertEqual(guard.minimum_gib, 18)
        with self.assertRaisesRegex(TrialFailure, "18 GiB"):
            guard.check(sample(17.9, oom=4), ready=False)

    def test_ready_transition_at_or_above_eighteen_is_sticky(self):
        guard = TrialGuard(oom_baseline=0)
        guard.check(sample(18.0, oom=0), ready=True)
        self.assertEqual(guard.minimum_gib, 18)
        guard.check(sample(18.0), ready=False)
        self.assertEqual(guard.minimum_gib, 18)

    def test_psi_and_kernel_oom_trip_during_startup(self):
        with self.assertRaisesRegex(TrialFailure, "PSI"):
            TrialGuard(0).check(sample(120, psi=10), ready=False)
        with self.assertRaisesRegex(TrialFailure, "oom_kill"):
            TrialGuard(4).check(sample(120, oom=5), ready=False)

    def test_metric_failure_is_fail_closed(self):
        with self.assertRaisesRegex(TrialFailure, "sample"):
            TrialGuard(0).check(None, ready=False)


class ApplicationOomTests(unittest.TestCase):
    def test_detects_known_cuda_oom_spellings(self):
        for line in (
            "torch.cuda.OutOfMemoryError: CUDA out of memory",
            "Failed to load model - not enough GPU memory",
            "RuntimeError: CUDA error: out of memory",
        ):
            with self.subTest(line=line):
                self.assertTrue(is_application_oom(line))

    def test_does_not_misclassify_unrelated_logs(self):
        self.assertFalse(is_application_oom("model loading started"))


class TrialCommandTests(unittest.TestCase):
    def test_trial_overrides_context_and_seq_and_mounts_before_image(self):
        argv = [
            "docker", "run", "-d", "--name", "qwen38-fp4-local-serve",
            "-e", "VLLM_PLE_TABLE_MEMORY=disk", IMAGE_PINNED,
            "/watch/qwen38-watchdog.py", "/opt/nvidia/nvidia_entrypoint.sh",
            "vllm", "serve", "model", "--max-model-len", "262144",
            "--max-num-seqs", "8",
        ]
        with patch("serve.trial.launch.build_docker_argv", return_value=argv):
            result = _trial_argv(object(), Path("watchdog.py"), Path("heartbeat"))
        image = result.index(IMAGE_PINNED)
        self.assertLess(result.index("QWEN38_TRIAL_HEARTBEAT=/watch/trial-heartbeat"), image)
        heartbeat_mount = next(x for x in result if x.endswith(":/watch/trial-heartbeat:ro"))
        self.assertLess(result.index(heartbeat_mount), image)
        self.assertEqual(result[result.index("--max-model-len") + 1], "32768")
        self.assertEqual(result[result.index("--max-num-seqs") + 1], "1")

    def test_create_command_is_single_shot_and_has_no_auto_restart(self):
        created = _trial_docker_create(
            ["docker", "run", "-d", "--name", "trial", "image", "cmd"]
        )
        self.assertEqual(created, ["docker", "create", "--name", "trial", "image", "cmd"])
        self.assertNotIn("--restart", created)

    def test_unexpected_docker_command_shape_is_refused(self):
        with self.assertRaises(TrialFailure):
            _trial_docker_create(["docker", "compose", "up"])

    def test_readiness_uses_the_model_id_returned_by_server(self):
        response = MagicMock()
        response.status = 200
        response.read.return_value = (
            b'{"data":[{"id":"nvidia/Qwen3.8-Flash-Next-FP4-Local"}]}'
        )
        response.__enter__.return_value = response
        with patch("serve.trial.urllib.request.urlopen", side_effect=[response, response]):
            self.assertEqual(
                _model_ready(), "nvidia/Qwen3.8-Flash-Next-FP4-Local"
            )

    def test_failed_model_list_probe_is_not_readiness(self):
        with patch(
            "serve.trial.urllib.request.urlopen",
            side_effect=[MagicMock(status=200), OSError("offline")],
        ):
            self.assertIsNone(_model_ready())

    def test_health_ready_transitions_even_without_model_id(self):
        health = MagicMock(status=200)
        health.__enter__.return_value = health
        listing = MagicMock(status=200)
        listing.read.side_effect = OSError("model list unavailable")
        listing.__enter__.return_value = listing
        with patch("serve.trial.urllib.request.urlopen", side_effect=[health, listing]):
            self.assertEqual(_probe_server(), (True, None, False))

    def test_health_ready_reports_mismatched_model_id(self):
        health = MagicMock(status=200)
        health.__enter__.return_value = health
        listing = MagicMock(status=200)
        listing.read.return_value = b'{"data":[{"id":"/cache/snapshot/path"}]}'
        listing.__enter__.return_value = listing
        with patch("serve.trial.urllib.request.urlopen", side_effect=[health, listing]):
            self.assertEqual(
                _probe_server(), (True, "/cache/snapshot/path", True)
            )

    def test_probe_sends_bearer_token_when_api_key_is_configured(self):
        health = MagicMock(status=200)
        health.__enter__.return_value = health
        listing = MagicMock(status=200)
        listing.read.return_value = (
            '{"data": [{"id": "%s"}]}' % MODEL_ID
        ).encode()
        listing.__enter__.return_value = listing
        seen = []

        def urlopen(request, timeout):
            seen.append(request)
            return health if len(seen) == 1 else listing

        with patch.dict("os.environ", {"VLLM_API_KEY": "unit-test-key"}, clear=False), \
             patch("serve.trial.urllib.request.urlopen", side_effect=urlopen):
            self.assertEqual(_probe_server(), (True, MODEL_ID, False))
        self.assertEqual(len(seen), 2)
        for request in seen:
            self.assertEqual(
                request.headers["Authorization"], "Bearer unit-test-key"
            )

    def test_trial_argv_pins_served_model_name(self):
        argv = [
            "docker", "run", IMAGE_PINNED, "watchdog", "entrypoint", "vllm",
            "serve", "model", "--max-model-len", "262144",
            "--max-num-seqs", "8",
        ]
        with patch("serve.trial.launch.build_docker_argv", return_value=argv):
            result = _trial_argv(object(), Path("watchdog.py"), Path("heartbeat"))
        index = result.index("--served-model-name")
        self.assertEqual(result[index + 1], "nvidia/Qwen3.8-Flash-Next-FP4-Local")


class FakeChild:
    def __init__(self, code=None):
        self.code = code
        self.stdout = []
        self.killed = False
        self.pid = 12345

    def poll(self):
        return self.code

    def kill(self):
        self.killed = True
        self.code = -9

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        self.code = -15


class FakeLiveLog:
    def __init__(self, child, *, eof=False):
        self.child = child
        self.eof = eof

    def readline(self):
        if self.eof or self.child.killed or self.child.code is not None:
            return b""
        threading.Event().wait(0.001)
        return b"watchdog status\n"


class ContainerWatchdogTests(unittest.TestCase):
    def test_watchdog_sample_uses_six_then_eighteen_gib(self):
        def read(path):
            if path.endswith("meminfo"):
                return f"MemAvailable: {12 * 1024 * 1024} kB\n"
            if path.endswith("memory"):
                return "full avg10=0.0 avg60=0 avg300=0 total=0\n"
            return "oom_kill 7\n"

        with patch.object(watchdog, "_read", side_effect=read):
            startup = watchdog._sample(7, False)
            serving = watchdog._sample(7, True)
        self.assertTrue(startup[0])
        self.assertFalse(serving[0])
        self.assertIn("18 GiB", serving[1])

    def _mock_preflight(self, sample):
        return (
            patch.object(watchdog, "_read", return_value="oom_kill 7\n"),
            patch.object(watchdog.os.path, "isfile", return_value=True),
            patch.object(watchdog.os, "access", return_value=True),
            patch.object(watchdog.shutil, "which", return_value="/usr/bin/vllm"),
            patch.object(watchdog, "_sample", side_effect=sample),
            patch.object(watchdog, "_heartbeat_alive", return_value=True),
        )

    def _run_context(self, sample, child, server_state):
        patches = self._mock_preflight(sample)
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(patch.stopall)
        popen = patch.object(watchdog.subprocess, "Popen", return_value=child)
        ready_patch = patch.object(
            watchdog, "_probe_server", return_value=server_state
        )
        popen_mock = popen.start()
        ready_patch.start()
        self.addCleanup(popen.stop)
        self.addCleanup(ready_patch.stop)
        return popen_mock

    def test_ready_switches_watchdog_to_eighteen_gib_before_success(self):
        calls = []

        def sample(oom, ready):
            calls.append(ready)
            return True, "safe"

        child = FakeChild()
        self._run_context(
            sample, child,
            (True, "nvidia/Qwen3.8-Flash-Next-FP4-Local", False),
        )
        with patch.object(watchdog.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                watchdog.main([
                    "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
                ])
        self.assertEqual(calls, [False, True])

    def test_ready_below_eighteen_kills_server(self):
        calls = []

        def sample(oom, ready):
            calls.append(ready)
            return (not ready, "below 18 GiB" if ready else "safe")

        child = FakeChild()
        self._run_context(
            sample, child,
            (True, "nvidia/Qwen3.8-Flash-Next-FP4-Local", False),
        )
        result = watchdog.main([
            "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
        ])
        self.assertEqual(result, 2)
        self.assertTrue(child.killed)

    def test_health_ready_without_model_id_uses_eighteen_floor(self):
        calls = []

        def sample(oom, ready):
            calls.append(ready)
            return True, "safe"

        child = FakeChild()
        self._run_context(sample, child, (True, None, False))
        with patch.object(watchdog.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                watchdog.main([
                    "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
                ])
        self.assertEqual(calls, [False, True])

    def test_health_ready_mismatch_kills_after_eighteen_sample(self):
        calls = []

        def sample(oom, ready):
            calls.append(ready)
            return True, "safe"

        child = FakeChild()
        self._run_context(sample, child, (True, "/cache/snapshot/path", True))
        result = watchdog.main([
            "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
        ])
        self.assertEqual(result, 2)
        self.assertEqual(calls, [False, True])
        self.assertTrue(child.killed)


class FakeTrialGuard:
    def __init__(self, samples=None):
        self.baseline_oom = None
        self.samples = 0
        self.baseline_captures = 0
        self.sample_values = list(samples or [])

    def capture_oom_baseline(self):
        self.baseline_captures += 1
        self.baseline_oom = 7
        return 7

    def sample(self):
        self.samples += 1
        if self.sample_values:
            return self.sample_values.pop(0)
        return sample(40, oom=7)


class TrialRunnerLifecycleTests(unittest.TestCase):
    def _run_trial(
        self, root, child, sleeps, *, ready_id=None, server_state=None,
        log_eof=False, samples=None, startup_timeout_s=1800.0,
    ):
        guard = FakeTrialGuard(samples=samples)
        child.stdout = FakeLiveLog(child, eof=log_eof)
        argv = [
            "docker", "run", "-d", "--name", launch.CONTAINER_NAME,
            "-e", "VLLM_PLE_TABLE_MEMORY=disk", IMAGE_PINNED,
            "/watch/qwen38-watchdog.py", "/opt/nvidia/nvidia_entrypoint.sh",
            "vllm", "serve", "model", "--max-model-len", "262144",
            "--max-num-seqs", "8",
        ]
        info = launch.ScopedInfo(
            "cid", launch.CONTAINER_NAME,
            launch.PROJECT_LABEL_VALUE, "created", 0,
        )
        running_info = launch.ScopedInfo(
            "cid", launch.CONTAINER_NAME,
            launch.PROJECT_LABEL_VALUE, "running", 0,
        )
        run = MagicMock(return_value=SimpleNamespace(
            returncode=0, stdout="cid\n", stderr=""
        ))

        def preflight(g, **kwargs):
            g.capture_oom_baseline()
            return {}

        with (
            patch("serve.trial.launch.preflight", side_effect=preflight),
            patch("serve.trial.launch.resolve_snapshot_cache", return_value=object()),
            patch("serve.trial.launch.build_docker_argv", return_value=argv),
            patch("serve.trial.launch.inspect_exact", side_effect=[info, running_info, running_info, running_info]),
            patch("serve.trial.launch.stop_project_container") as stop,
            patch("serve.trial.launch._run_capture", run),
            patch("serve.trial.subprocess.Popen", return_value=child),
            patch("serve.trial.time.sleep", side_effect=sleeps),
        ):
            runner = TrialRunner(
                guard=guard, evidence_root=root,
                server_probe=lambda: (
                    server_state if server_state is not None
                    else (ready_id is not None, ready_id, False)
                ),
                startup_timeout_s=startup_timeout_s,
            )
            stop.side_effect = lambda **kwargs: child.kill()
            with self.assertRaises((KeyboardInterrupt, TrialFailure)) as raised:
                runner.execute()
        return guard, stop, raised.exception, next(Path(root).glob("trial-*"))

    def test_precreate_guard_reuses_preflight_oom_baseline(self):
        with tempfile.TemporaryDirectory() as root:
            guard = FakeTrialGuard(samples=[sample(5.9, oom=7)])
            argv = [
                "docker", "run", "-d", "--name", launch.CONTAINER_NAME,
                IMAGE_PINNED, "watchdog", "entrypoint",
                "--max-model-len", "262144", "--max-num-seqs", "8",
            ]
            run = MagicMock()

            def preflight(g, **kwargs):
                g.capture_oom_baseline()
                return {}

            with (
                patch("serve.trial.launch.preflight", side_effect=preflight),
                patch("serve.trial.launch.resolve_snapshot_cache", return_value=object()),
                patch("serve.trial.launch.build_docker_argv", return_value=argv),
                patch("serve.trial.launch._run_capture", run),
            ):
                runner = TrialRunner(guard=guard, evidence_root=root)
                with self.assertRaises(TrialFailure):
                    runner.execute()
            run.assert_not_called()
            self.assertEqual(guard.baseline_captures, 1)

    def test_diagnostic_precreate_failure_artifact_reports_eighteen_gib(self):
        with tempfile.TemporaryDirectory() as root:
            guard = FakeTrialGuard(samples=[sample(17.9, oom=7)])
            argv = [
                "docker", "run", "-d", "--name", launch.CONTAINER_NAME,
                IMAGE_PINNED, "watchdog", "entrypoint",
                "--max-model-len", "32768", "--max-num-seqs", "1",
            ]
            run = MagicMock()

            def preflight(g, **kwargs):
                g.capture_oom_baseline()
                return {}

            with (
                patch("serve.trial.launch.preflight", side_effect=preflight),
                patch("serve.trial.launch.resolve_snapshot_cache", return_value=object()),
                patch("serve.trial.launch.build_docker_argv", return_value=argv),
                patch("serve.trial.launch._run_capture", run),
            ):
                runner = TrialRunner(guard=guard, evidence_root=root, startup_min_gib=18)
                with self.assertRaises(TrialFailure):
                    runner.execute()
            outcome = json.loads(next(Path(root).glob("trial-*/outcome.json")).read_text())
            self.assertEqual(outcome["threshold_gib"], 18)


    def test_stays_supervised_after_readiness_until_interrupted(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild()
            guard, stop, raised, evidence = self._run_trial(
                root, child, [None, KeyboardInterrupt],
                ready_id="nvidia/Qwen3.8-Flash-Next-FP4-Local",
            )
            self.assertIsInstance(raised, KeyboardInterrupt)
            readiness = json.loads((evidence / "readiness.json").read_text())
            outcome = json.loads((evidence / "outcome.json").read_text())
            self.assertEqual(readiness["model_id"], "nvidia/Qwen3.8-Flash-Next-FP4-Local")
            self.assertTrue(outcome["ready"])
            self.assertFalse(outcome["restart"])
            self.assertGreaterEqual(guard.samples, 4)
            self.assertEqual(guard.baseline_captures, 1)
            stop.assert_called_once()
            self.assertTrue((evidence / "container.log").exists())
            self.assertTrue((evidence / "metrics.log").exists())

    def test_log_eof_while_container_running_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild()
            real_sleep = __import__("time").sleep
            _, stop, raised, _ = self._run_trial(
                root, child, [lambda _: real_sleep(0.02)], ready_id=None,
                log_eof=True,
            )
            self.assertIsInstance(raised, TrialFailure)
            self.assertIn("log", str(raised).lower())
            stop.assert_called_once()

    def test_high_volume_normal_logs_preserve_tail_and_eof_without_queue_block(self):
        import io
        from queue import Queue

        raw = b"".join(f"diag-{i:04d}\n".encode() for i in range(400))
        events = Queue(maxsize=256)
        stop = threading.Event()
        with tempfile.TemporaryFile("w+b") as destination:
            reader = threading.Thread(
                target=TrialRunner._copy_logs,
                args=(io.BytesIO(raw), destination, events, stop),
            )
            reader.start()
            reader.join(timeout=1)
            self.assertFalse(reader.is_alive(), "normal telemetry filled event queue")
            destination.seek(0)
            saved = destination.read()
        self.assertEqual(saved, raw)
        self.assertEqual(events.qsize(), 1)
        with self.assertRaisesRegex(TrialFailure, "ended unexpectedly"):
            TrialRunner._drain_logs(events)

    def test_high_volume_logs_still_queue_oom_for_action(self):
        import io
        from queue import Queue

        lines = [f"progress {i}\n" for i in range(400)]
        lines.insert(399, "torch.cuda.OutOfMemoryError: CUDA out of memory\n")
        raw = "".join(lines).encode()
        events = Queue(maxsize=256)
        with tempfile.TemporaryFile("w+b") as destination:
            reader = threading.Thread(
                target=TrialRunner._copy_logs,
                args=(io.BytesIO(raw), destination, events, threading.Event()),
            )
            reader.start()
            reader.join(timeout=1)
            self.assertFalse(reader.is_alive(), "OOM telemetry blocked log persistence")
            destination.seek(0)
            self.assertEqual(destination.read(), raw)
        queued = TrialRunner._drain_logs(events, allow_eof=True)
        self.assertEqual(len(queued), 1)
        self.assertTrue(is_application_oom(queued[0]))

    def test_health_ready_without_id_uses_eighteen_without_announcement(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild()
            guard, stop, raised, evidence = self._run_trial(
                root, child, [None, KeyboardInterrupt],
                server_state=(True, None, False),
            )
            self.assertIsInstance(raised, KeyboardInterrupt)
            self.assertTrue(json.loads((evidence / "outcome.json").read_text())["ready"])
            self.assertFalse((evidence / "readiness.json").exists())
            self.assertGreaterEqual(guard.samples, 4)
            stop.assert_called_once()

    def test_health_without_verified_model_id_still_times_out(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild()
            _, stop, raised, evidence = self._run_trial(
                root, child, [], server_state=(True, None, False),
                startup_timeout_s=0,
            )
            self.assertIsInstance(raised, TrialFailure)
            self.assertIn("readiness timeout", str(raised))
            self.assertIn("ready=1", (evidence / "metrics.log").read_text())
            self.assertFalse((evidence / "readiness.json").exists())
            stop.assert_called_once()

    def test_health_ready_with_mismatched_id_fails_closed_at_eighteen(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild()
            _, stop, raised, evidence = self._run_trial(
                root, child, [], server_state=(True, "/cache/snapshot", True),
            )
            self.assertIsInstance(raised, TrialFailure)
            self.assertIn("unexpected model ID", str(raised))
            metrics = (evidence / "metrics.log").read_text()
            self.assertIn("ready=1", metrics)
            self.assertFalse((evidence / "readiness.json").exists())
            stop.assert_called_once()

    def test_attach_exit_still_stops_verified_running_container(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild(code=1)
            _, stop, raised, _ = self._run_trial(root, child, [])
            self.assertIsInstance(raised, TrialFailure)
            stop.assert_called_once()
            self.assertEqual(stop.call_args.kwargs["expected_id"], "cid")

    def test_created_container_state_is_bounded_while_metrics_continue(self):
        with tempfile.TemporaryDirectory() as root:
            child = FakeChild()
            guard = FakeTrialGuard()
            created = launch.ScopedInfo(
                "cid", launch.CONTAINER_NAME, launch.PROJECT_LABEL_VALUE,
                "created", 0,
            )
            running = launch.ScopedInfo(
                "cid", launch.CONTAINER_NAME, launch.PROJECT_LABEL_VALUE,
                "running", 0,
            )
            # The first inspect binds the ID. The next sees Docker's brief
            # created state after start, then running; both iterations sample.
            info = [created, created, running, running, running]
            child.stdout = FakeLiveLog(child)
            argv = [
                "docker", "run", "-d", "--name", launch.CONTAINER_NAME,
                "-e", "VLLM_PLE_TABLE_MEMORY=disk", IMAGE_PINNED,
                "/watch/qwen38-watchdog.py", "/opt/nvidia/nvidia_entrypoint.sh",
                "vllm", "serve", "model", "--max-model-len", "262144",
                "--max-num-seqs", "8",
            ]
            run = MagicMock(return_value=SimpleNamespace(returncode=0, stdout="cid\n", stderr=""))

            def preflight(g, **kwargs):
                g.capture_oom_baseline()
                return {}

            with (
                patch("serve.trial.launch.preflight", side_effect=preflight),
                patch("serve.trial.launch.resolve_snapshot_cache", return_value=object()),
                patch("serve.trial.launch.build_docker_argv", return_value=argv),
                patch("serve.trial.launch.inspect_exact", side_effect=lambda **_: info.pop(0)),
                patch("serve.trial.launch.stop_project_container") as stop,
                patch("serve.trial.launch._run_capture", run),
                patch("serve.trial.subprocess.Popen", return_value=child),
                patch("serve.trial.time.sleep", side_effect=[None, None, KeyboardInterrupt]),
            ):
                runner = TrialRunner(
                    guard=guard, evidence_root=root,
                    server_probe=lambda: (False, None, False),
                )
                stop.side_effect = lambda **kwargs: child.kill()
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute()
            self.assertGreaterEqual(guard.samples, 4)
            stop.assert_called_once()


class ContainerWatchdogFailureTests(ContainerWatchdogTests):
    def test_sigterm_kills_without_unmonitored_grace(self):
        child = FakeChild()
        self._run_context(lambda *_: (True, "safe"), child, (False, None, False))

        def register(signum, handler):
            if signum == watchdog.signal.SIGTERM:
                handler()

        with patch.object(watchdog.signal, "signal", side_effect=register):
            result = watchdog.main([
                "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
            ])
        self.assertEqual(result, 143)
        self.assertTrue(child.killed)

    def test_monitor_failure_kills_server(self):
        calls = 0

        def sample(oom, ready):
            nonlocal calls
            calls += 1
            if calls == 1:
                return True, "safe"
            raise watchdog.WatchdogError("sample lost")

        child = FakeChild()
        self._run_context(sample, child, (False, None, False))
        self.assertEqual(
            watchdog.main([
                "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
            ]),
            2,
        )
        self.assertTrue(child.killed)

    def test_host_heartbeat_loss_kills_server(self):
        child = FakeChild()
        self._run_context(lambda *_: (True, "safe"), child, (False, None, False))
        with patch.object(watchdog, "_heartbeat_alive", side_effect=[True, False]):
            result = watchdog.main([
                "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
            ])
        self.assertEqual(result, 2)
        self.assertTrue(child.killed)

    def test_stale_heartbeat_prevents_server_spawn(self):
        child = FakeChild()
        popen = self._run_context(lambda *_: (True, "safe"), child, (False, None, False))
        with patch.object(watchdog, "_heartbeat_alive", return_value=False):
            result = watchdog.main([
                "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
            ])
        self.assertEqual(result, 3)
        popen.assert_not_called()

    def test_child_failure_is_not_restarted(self):
        child = FakeChild(code=1)
        popen = self._run_context(lambda *_: (True, "safe"), child, (False, None, False))
        result = watchdog.main([
            "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
        ])
        self.assertEqual(result, 1)
        self.assertEqual(popen.call_count, 1)

    def test_detectable_cuda_oom_log_kills_server(self):
        child = FakeChild()
        child.stdout = ["CUDA out of memory\n"]
        self._run_context(lambda *_: (True, "safe"), child, (False, None, False))
        original_sleep = watchdog.time.sleep
        with patch.object(
            watchdog.time, "sleep", side_effect=lambda _: original_sleep(0.01)
        ):
            result = watchdog.main([
                "/opt/nvidia/nvidia_entrypoint.sh", "VLLM", "serve"
            ])
        self.assertEqual(result, 2)
        self.assertTrue(child.killed)


if __name__ == "__main__":
    unittest.main()
