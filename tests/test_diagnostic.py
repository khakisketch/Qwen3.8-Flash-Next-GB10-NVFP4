from __future__ import annotations

import unittest
import tempfile
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from serve import diagnostic, launch
from serve.baseline import IMAGE_PINNED, MODEL_ID
from serve.guard import Sample


class DiagnosticPolicyTests(unittest.TestCase):
    def test_diagnostic_argv_mounts_readonly_hook_and_env_before_image(self):
        base = ["docker", "run", "-d", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1", "--max-model-len", "262144",
                "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            hooks = Path(temp)
            (hooks / "sitecustomize.py").write_text("# fake")
            with patch("serve.trial.launch.build_docker_argv", side_effect=lambda *_, **__: list(base)):
                argv = diagnostic.diagnostic_argv(object(), Path("watchdog"), hooks, Path("heartbeat"))
        image = argv.index(IMAGE_PINNED)
        self.assertLess(argv.index("QWEN38_MEMORY_DIAGNOSTICS=1"), image)
        hook_mount = next(v for v in argv if ":/watch/diag:ro" in v)
        self.assertLess(argv.index(hook_mount), image)
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "32768")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "1")
        self.assertNotIn("QWEN38_RECLAIM_CSF_CACHE=1", argv)
        self.assertNotIn("--kv-cache-memory-bytes", argv)
        self.assertFalse(any(":/root/.cache/b12x:rw" in value for value in argv))

    def test_listen_all_is_opt_in_and_changes_only_vllm_bind(self):
        base = ["docker", "run", "-d", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1",
                "--max-model-len", "262144", "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            hooks = Path(temp)
            (hooks / "sitecustomize.py").write_text("# fake")
            with patch("serve.trial.launch.build_docker_argv", side_effect=lambda *_, **__: list(base)):
                local_argv = diagnostic.diagnostic_argv(
                    object(), Path("watchdog"), hooks, Path("heartbeat"),
                )
                lan_argv = diagnostic.diagnostic_argv(
                    object(), Path("watchdog"), hooks, Path("heartbeat"),
                    listen_all=True,
                )

        self.assertEqual(local_argv[local_argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(lan_argv[lan_argv.index("--host") + 1], "0.0.0.0")
        self.assertGreater(lan_argv.index("--host"), lan_argv.index(MODEL_ID))
        self.assertEqual(diagnostic.build_parser().parse_args([]).listen_all, False)
        self.assertTrue(diagnostic.build_parser().parse_args(["--listen-all"]).listen_all)

    def test_listen_all_does_not_change_local_health_or_model_probe_addresses(self):
        from serve import trial

        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return ('{"data": [{"id": "%s"}]}' % MODEL_ID).encode()

        urls = []
        def urlopen(url, timeout):
            urls.append(url)
            return Response()

        with patch("serve.trial.urllib.request.urlopen", side_effect=urlopen):
            self.assertEqual(trial._probe_server(), (True, MODEL_ID, False))
        self.assertEqual(urls, [
            f"http://{launch.HOST_BIND}:{launch.HOST_PORT}/health",
            f"http://{launch.HOST_BIND}:{launch.HOST_PORT}/v1/models",
        ])
        self.assertEqual(launch.HOST_BIND, "127.0.0.1")

    def test_reclaim_option_is_opt_in_and_precedes_image(self):
        base = ["docker", "run", "-d", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1", "--max-model-len", "262144",
                "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            hooks = Path(temp)
            (hooks / "sitecustomize.py").write_text("# fake")
            with patch("serve.trial.launch.build_docker_argv", return_value=base):
                argv = diagnostic.diagnostic_argv(
                    object(), Path("watchdog"), hooks, Path("heartbeat"),
                    reclaim_csf_cache=True,
                )
        self.assertLess(argv.index("QWEN38_RECLAIM_CSF_CACHE=1"), argv.index(IMAGE_PINNED))

    def test_diagnostic_cli_reclaim_flag_defaults_off(self):
        parser = diagnostic.build_parser()
        self.assertFalse(parser.parse_args([]).reclaim_csf_cache)
        self.assertTrue(parser.parse_args(["--reclaim-csf-cache"]).reclaim_csf_cache)

    def test_diagnostic_cli_listen_all_defaults_off(self):
        parser = diagnostic.build_parser()
        self.assertFalse(parser.parse_args([]).listen_all)
        self.assertTrue(parser.parse_args(["--listen-all"]).listen_all)

    def test_diagnostic_cli_cache_options_are_opt_in_and_kv_bytes_positive(self):
        parser = diagnostic.build_parser()
        defaults = parser.parse_args([])
        self.assertIsNone(defaults.kv_cache_memory_bytes)
        self.assertIsNone(defaults.kernel_cache_dir)
        configured = parser.parse_args([
            "--kv-cache-memory-bytes", "4294967296",
            "--kernel-cache-dir", "/tmp/generated-kernel-cache",
        ])
        self.assertEqual(configured.kv_cache_memory_bytes, 4294967296)
        self.assertEqual(configured.kernel_cache_dir, "/tmp/generated-kernel-cache")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--kv-cache-memory-bytes", "0"])

    def test_diagnostic_argv_places_kv_limit_and_explicit_cache_mounts_before_image(self):
        base = ["docker", "run", "-d", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1", "--gpu-memory-utilization", "0.8",
                "--max-model-len", "262144", "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            hooks = Path(temp) / "hooks"
            hooks.mkdir()
            (hooks / "sitecustomize.py").write_text("# fake")
            cache = Path(temp) / "generated-kernels"
            (cache / "b12x").mkdir(parents=True)
            (cache / "triton").mkdir()
            with patch("serve.trial.launch.build_docker_argv", return_value=base):
                argv = diagnostic.diagnostic_argv(
                    object(), Path("watchdog"), hooks, Path("heartbeat"),
                    kv_cache_memory_bytes=4294967296, kernel_cache_dir=cache,
                )
        image = argv.index(IMAGE_PINNED)
        kv_flag = argv.index("--kv-cache-memory-bytes")
        self.assertEqual(argv[kv_flag + 1], "4294967296")
        self.assertGreater(kv_flag, argv.index(MODEL_ID))
        self.assertGreater(kv_flag, image)
        self.assertLess(kv_flag, argv.index("--gpu-memory-utilization"))
        mounts = [value for value in argv if value.endswith(":rw")]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(any(value.endswith("/b12x:/root/.cache/b12x:rw") for value in mounts))
        self.assertTrue(any(value.endswith("/triton:/root/.triton:rw") for value in mounts))
        self.assertTrue(all(argv.index(value) < image for value in mounts))
        self.assertEqual(argv[argv.index("--gpu-memory-utilization") + 1], "0.8")

    def test_cache_mount_rejects_missing_expected_subdirectory(self):
        base = ["docker", "run", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1", "--max-model-len", "262144",
                "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            hooks = Path(temp) / "hooks"
            hooks.mkdir()
            (hooks / "sitecustomize.py").write_text("# fake")
            cache = Path(temp) / "cache"
            (cache / "b12x").mkdir(parents=True)
            with patch("serve.trial.launch.build_docker_argv", return_value=base):
                with self.assertRaises(diagnostic.DiagnosticFailure):
                    diagnostic.diagnostic_argv(
                        object(), Path("watchdog"), hooks, Path("heartbeat"),
                        kernel_cache_dir=cache,
                    )

    def test_cache_mount_rejects_subdirectory_symlink_escape(self):
        base = ["docker", "run", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1", "--max-model-len", "262144",
                "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hooks = root / "hooks"
            hooks.mkdir()
            (hooks / "sitecustomize.py").write_text("# fake")
            outside = root / "outside"
            (outside / "b12x").mkdir(parents=True)
            (outside / "triton").mkdir()
            cache = root / "cache"
            cache.mkdir()
            (cache / "b12x").symlink_to(outside / "b12x", target_is_directory=True)
            (cache / "triton").mkdir()
            with patch("serve.trial.launch.build_docker_argv", return_value=base):
                with self.assertRaises(diagnostic.DiagnosticFailure):
                    diagnostic.diagnostic_argv(
                        object(), Path("watchdog"), hooks, Path("heartbeat"),
                        kernel_cache_dir=cache,
                    )

    def test_cache_reclaim_skips_small_cache_and_reports_reclaimed_cache(self):
        from serve.diag_hooks import layer_diagnostics as ld

        class Cuda:
            def __init__(self, reserved):
                self.reserved = reserved
                self.empty_calls = 0
            def memory_allocated(self): return 100
            def memory_reserved(self): return self.reserved
            def empty_cache(self): self.empty_calls += 1
        class Torch:
            def __init__(self, cuda): self.cuda = cuda

        small = Cuda(100 + 512 * 1024**2 - 1)
        self.assertFalse(ld.maybe_reclaim_csf_cache(
            "before-test", active=True, enabled=True, torch_module=Torch(small),
            memory=lambda: {"torch_allocated_bytes": 100,
                            "torch_reserved_bytes": small.reserved},
        ))
        self.assertEqual(small.empty_calls, 0)

        cuda = Cuda(100 + 512 * 1024**2)
        events = []
        self.assertTrue(ld.maybe_reclaim_csf_cache(
            "before-test", active=True, enabled=True, torch_module=Torch(cuda),
            emit=events.append,
            memory=lambda: {"host_memavailable_bytes": 20,
                            "torch_allocated_bytes": 100,
                            "torch_reserved_bytes": cuda.reserved},
        ))
        self.assertEqual(cuda.empty_calls, 1)
        self.assertEqual([e["event"] for e in events],
                         ["CACHE_RECLAIM_BEFORE", "CACHE_RECLAIM_AFTER"])

    def test_reclaim_helper_preserves_live_payload_and_original_error(self):
        from serve.diag_hooks import layer_diagnostics as ld

        class Cuda:
            def __init__(self): self.empty_calls = 0
            def memory_allocated(self): return 1
            def memory_reserved(self): return 512 * 1024**2 + 1
            def empty_cache(self): self.empty_calls += 1
        class Torch:
            def __init__(self): self.cuda = Cuda()
        torch = Torch()
        payload = object()
        events = []
        real_reclaim = ld.maybe_reclaim_csf_cache
        def reclaim(boundary, **kwargs):
            return real_reclaim(
                boundary, active=True, enabled=True, torch_module=torch,
                emit=events.append,
                memory=lambda: {"torch_allocated_bytes": 1,
                                "torch_reserved_bytes": torch.cuda.memory_reserved()},
            )
        with patch.object(ld, "maybe_reclaim_csf_cache", side_effect=reclaim):
            wrapped = ld.wrap_helper(
                lambda value: value,
                "b12x.moe._shared.kernels.w4a16.prepare._prepare_w4a16_packed_weights",
                emit=events.append, memory=lambda: {"process_rss_bytes": 1},
            )
            self.assertIs(wrapped(payload), payload)
            original_error = ValueError("packing failure")
            def fail(_): raise original_error
            failed = ld.wrap_helper(
                fail,
                "b12x.moe._shared.kernels.w4a16.prepare._prepare_w4a16_packed_weights",
                emit=events.append, memory=lambda: {"process_rss_bytes": 1},
            )
            with self.assertRaises(ValueError) as raised:
                failed(payload)
        self.assertIs(raised.exception, original_error)
        self.assertEqual(torch.cuda.empty_calls, 2)

    def test_cache_reclaim_defaults_off_without_opt_in_environment(self):
        from serve.diag_hooks import layer_diagnostics as ld

        class Cuda:
            def __init__(self): self.empty_calls = 0
            def empty_cache(self): self.empty_calls += 1
        class Torch:
            def __init__(self): self.cuda = Cuda()
        torch = Torch()
        with patch.dict(os.environ, {}, clear=True):
            reclaimed = ld.maybe_reclaim_csf_cache(
                "default-off", active=True, torch_module=torch,
                memory=lambda: {"torch_allocated_bytes": 1,
                                "torch_reserved_bytes": 1024**3},
            )
        self.assertFalse(reclaimed)
        self.assertEqual(torch.cuda.empty_calls, 0)

    def test_cache_reclaim_scope_is_only_nvfp4_csf_postload(self):
        from serve.diag_hooks import layer_diagnostics as ld

        class Nvfp4CsfMoEMethod:
            def process_weights_after_loading(self, layer):
                return ld.csf_postload_active()
        class OtherQuantMethod:
            def process_weights_after_loading(self, layer):
                return ld.csf_postload_active()
        wrapped = ld.wrap_quant_method(Nvfp4CsfMoEMethod.process_weights_after_loading)
        other = ld.wrap_quant_method(OtherQuantMethod.process_weights_after_loading)
        self.assertFalse(other(OtherQuantMethod(), object()))
        self.assertTrue(wrapped(Nvfp4CsfMoEMethod(), object()))
        self.assertFalse(ld.csf_postload_active())

    def test_scale_factor_probe_reclaims_after_without_enter_exit_spam(self):
        from serve.diag_hooks import layer_diagnostics as ld

        events = []
        def reclaim(boundary, **kwargs):
            events.append({"event": "CACHE_RECLAIM_BEFORE", "boundary": boundary})
            events.append({"event": "CACHE_RECLAIM_AFTER", "boundary": boundary})
            return True
        with patch.object(ld, "maybe_reclaim_csf_cache", side_effect=reclaim):
            helper = ld.wrap_helper(
                lambda value: value * 2,
                "b12x.moe._shared.kernels.w4a16.prepare._nvfp4_compute_scale_factor",
                emit=events.append, memory=lambda: {"process_rss_bytes": 1},
            )
            self.assertEqual(helper(4), 8)
        self.assertEqual([event["event"] for event in events],
                         ["CACHE_RECLAIM_BEFORE", "CACHE_RECLAIM_AFTER"])

    def test_diagnostic_argv_supports_measured_64k_two_sequence_profile(self):
        base = ["docker", "run", "-d", IMAGE_PINNED, "watchdog", "entrypoint",
                "vllm", "serve", MODEL_ID, "--host", "127.0.0.1",
                "--max-model-len", "262144", "--max-num-seqs", "8"]
        with tempfile.TemporaryDirectory() as temp:
            hooks = Path(temp)
            (hooks / "sitecustomize.py").write_text("# fake")
            with patch("serve.trial.launch.build_docker_argv", return_value=base):
                argv = diagnostic.diagnostic_argv(
                    object(), Path("watchdog"), hooks, Path("heartbeat"),
                    context=65536, sequences=2,
                )
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "65536")
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "2")
        with self.assertRaises(diagnostic.DiagnosticFailure):
            diagnostic.diagnostic_argv(
                object(), Path("watchdog"), hooks, Path("heartbeat"),
                context=131072, sequences=2,
            )

    def test_diagnostic_guard_is_eighteen_gib_from_start(self):
        self.assertEqual(diagnostic.DIAGNOSTIC_MIN_GIB, 18)
        policy = diagnostic.DiagnosticGuard(oom_baseline=5)
        with self.assertRaises(diagnostic.DiagnosticFailure):
            policy.check(Sample(int(17.9 * 1024**3), 0, 5))
        policy.check(Sample(18 * 1024**3, 0, 5))
        with self.assertRaises(diagnostic.DiagnosticFailure):
            policy.check(Sample(30 * 1024**3, 10, 5))

    def test_pid1_watchdog_source_remains_fixed_at_eighteen(self):
        text = Path(launch.default_watchdog_host_path()).read_text()
        self.assertIn("MEM_MIN_BYTES = 18 * 1024**3", text)

    def test_runner_host_preflight_and_watchdog_use_eighteen(self):
        runner = diagnostic.DiagnosticRunner(evidence_root="unused")
        self.assertEqual(runner.guard.mem_min_bytes, 18 * 1024**3)
        self.assertEqual(runner.startup_min_gib, 18)
        self.assertEqual(runner.watchdog_path.name, "diagnostic_watchdog.py")
        self.assertFalse(runner.reclaim_csf_cache)
        enabled = diagnostic.DiagnosticRunner(evidence_root="unused", reclaim_csf_cache=True)
        self.assertTrue(enabled.reclaim_csf_cache)
        self.assertTrue(enabled.outcome_metadata["diagnostic_options"]["reclaim_csf_cache"])
        configured = diagnostic.DiagnosticRunner(
            evidence_root="unused", kv_cache_memory_bytes=4294967296,
            kernel_cache_dir="/tmp/kernels",
        )
        self.assertEqual(configured.outcome_metadata["diagnostic_options"], {
            "reclaim_csf_cache": False,
            "kv_cache_memory_bytes": 4294967296,
            "kernel_cache_dir": "/tmp/kernels",
            "listen_all": False,
            "context": 32768,
            "sequences": 1,
        })
        lan = diagnostic.DiagnosticRunner(evidence_root="unused", listen_all=True)
        self.assertTrue(lan.outcome_metadata["diagnostic_options"]["listen_all"])

    def test_pid1_diagnostic_watchdog_uses_eighteen_before_and_after_ready(self):
        from serve import diagnostic_watchdog as watchdog

        self.assertEqual(watchdog.STARTUP_MIN_BYTES, 18 * 1024**3)
        self.assertEqual(watchdog.OPERATING_MIN_BYTES, 18 * 1024**3)
        mem = [f"MemAvailable: {int(17.9 * 1024**2)} kB\n"]
        files = {
            "/proc/meminfo": mem[0],
            "/proc/pressure/memory": "some avg10=0 avg60=0 avg300=0 total=0\nfull avg10=0 avg60=0 avg300=0 total=0\n",
            "/proc/vmstat": "oom_kill 7\n",
        }
        with patch.object(watchdog, "_read", side_effect=lambda path: files[path]):
            self.assertFalse(watchdog._sample(7, False)[0])
            self.assertFalse(watchdog._sample(7, True)[0])
            files["/proc/meminfo"] = f"MemAvailable: {18 * 1024**2} kB\n"
            self.assertTrue(watchdog._sample(7, False)[0])
            self.assertTrue(watchdog._sample(7, True)[0])

    def test_pid1_diagnostic_watchdog_probe_uses_configured_api_key(self):
        from serve import diagnostic_watchdog as watchdog

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

        with patch.dict(os.environ, {"VLLM_API_KEY": "unit-test-key"}, clear=False), \
             patch.object(watchdog.urllib.request, "urlopen", side_effect=urlopen):
            self.assertEqual(watchdog._probe_server(), (True, MODEL_ID, False))
        for request in seen:
            self.assertEqual(
                request.headers["Authorization"], "Bearer unit-test-key"
            )

    def test_pid1_diagnostic_watchdog_keeps_heartbeat_and_oom_log_checks(self):
        from serve import diagnostic_watchdog as watchdog

        self.assertTrue(watchdog._oom_log("torch.cuda.OutOfMemoryError"))
        with tempfile.TemporaryDirectory() as temp:
            heartbeat = Path(temp) / "heartbeat"
            heartbeat.touch()
            with patch.object(watchdog, "HEARTBEAT_PATH", str(heartbeat)):
                self.assertTrue(watchdog._heartbeat_alive())

    def test_pid1_rss_parser_reports_bytes(self):
        from serve.diag_hooks.layer_diagnostics import process_rss_bytes

        with tempfile.TemporaryDirectory() as temp:
            status = Path(temp) / "status"
            status.write_text("Name:\tpython\nVmRSS:\t1234 kB\n")
            self.assertEqual(process_rss_bytes(status), 1234 * 1024)

    def test_memory_snapshot_contains_host_and_cpu_rss(self):
        from serve.diag_hooks import layer_diagnostics as ld

        with (patch.object(ld, "_host_available", return_value=456),
              patch.object(ld, "process_rss_bytes", return_value=789)):
            snapshot = ld.memory_snapshot()
        self.assertEqual(snapshot["host_memavailable_bytes"], 456)
        self.assertEqual(snapshot["process_rss_bytes"], 789)

    def test_probe_wrapper_marks_enter_and_exit_without_transforming_call(self):
        from serve.diag_hooks import layer_diagnostics as ld

        events = []
        original = lambda value, *, option: (value, option)
        wrapped = ld.wrap_helper(
            original, "fake.module.prepare", emit=events.append,
            memory=lambda: {"process_rss_bytes": 77},
        )
        self.assertEqual(wrapped("x", option=3), ("x", 3))
        self.assertEqual([e["event"] for e in events], ["HELPER_ENTER", "HELPER_EXIT"])
        self.assertEqual(events[0]["helper"], "fake.module.prepare")
        self.assertEqual(events[1]["process_rss_bytes"], 77)

    def test_probe_wrapper_records_error_from_helper(self):
        from serve.diag_hooks import layer_diagnostics as ld

        events = []
        def fail(): raise ValueError("CSF preparation failed")
        wrapped = ld.wrap_helper(fail, "fake.module.prepare", emit=events.append,
                                 memory=lambda: {"process_rss_bytes": 77})
        with self.assertRaisesRegex(ValueError, "CSF preparation failed"):
            wrapped()
        self.assertEqual([e["event"] for e in events], ["HELPER_ENTER", "HELPER_ERROR"])

    def test_probe_telemetry_failures_do_not_prevent_successful_helper(self):
        from serve.diag_hooks import layer_diagnostics as ld

        calls = []
        def broken_memory(): raise RuntimeError("snapshot unavailable")
        def broken_emit(_): raise RuntimeError("stderr sink unavailable")
        wrapped = ld.wrap_helper(
            lambda: calls.append("called") or "result", "fake.prepare",
            emit=broken_emit, memory=broken_memory,
        )
        self.assertEqual(wrapped(), "result")
        self.assertEqual(calls, ["called"])

    def test_probe_telemetry_failures_do_not_mask_helper_exception(self):
        from serve.diag_hooks import layer_diagnostics as ld

        original = ValueError("original helper failure")
        def fail(): raise original
        def broken_memory(): raise RuntimeError("snapshot unavailable")
        def broken_emit(_): raise RuntimeError("stderr sink unavailable")
        wrapped = ld.wrap_helper(
            fail, "fake.prepare", emit=broken_emit, memory=broken_memory,
        )
        with self.assertRaises(ValueError) as raised:
            wrapped()
        self.assertIs(raised.exception, original)

    def test_requested_csf_and_w4a16_helpers_are_probed(self):
        from serve.diag_hooks.layer_diagnostics import _PROBED_HELPERS

        self.assertIn("prepare_nvfp4_csf_weights", _PROBED_HELPERS[
            "vllm.model_executor.layers.quantization.utils.nvfp4_csf_utils"])
        self.assertIn("prepare_weights", _PROBED_HELPERS["b12x.moe.fused_moe.planning"])
        self.assertEqual(set(_PROBED_HELPERS["b12x._lib.quant._csf_prepare"]), {
            "validate_source", "expand_source_scales", "prepare_csf_payload",
        })
        self.assertIn("prepare_csf_block_weights", _PROBED_HELPERS[
            "b12x.moe.fused_moe._csf_preparation"])
        self.assertNotIn("_process_nvfp4_packed_scales", _PROBED_HELPERS[
            "b12x.moe._shared.kernels.w4a16.prepare"])
        self.assertIn("_nvfp4_compute_scale_factor", _PROBED_HELPERS[
            "b12x.moe._shared.kernels.w4a16.prepare"])

    def test_rss_sampler_emits_inflight_pid_samples_and_stops(self):
        from serve.diag_hooks import layer_diagnostics as ld

        events = []
        stop = threading.Event()
        thread = ld.start_rss_sampler(
            interval=0.01, stop_event=stop, emit=events.append,
            sampler=lambda: 123,
        )
        time.sleep(0.04)
        stop.set()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any(e["event"] == "RSS_SAMPLE" and
                            e["process_rss_bytes"] == 123 for e in events))


class InstrumentationTests(unittest.TestCase):
    def test_sitecustomize_registers_child_safe_hook_for_dash_c_process(self):
        hooks = Path(__file__).parents[1] / "serve" / "diag_hooks"
        env = dict(os.environ)
        env["QWEN38_MEMORY_DIAGNOSTICS"] = "1"
        env["PYTHONPATH"] = str(hooks)
        proc = subprocess.run(
            [sys.executable, "-c", "import sys; print(sys.argv[0]); print('layer_diagnostics' in sys.modules); print('torch' in sys.modules); print(any(type(f).__name__ == 'VllmDiagnosticFinder' for f in sys.meta_path))"],
            capture_output=True, text=True, env=env, check=True,
        )
        self.assertEqual(proc.stdout.splitlines(), ["-c", "False", "False", "True"])

    def test_spawned_child_installs_before_loader_alias_and_logs_pid(self):
        hooks = Path(__file__).parents[1] / "serve" / "diag_hooks"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "vllm" / "model_executor" / "model_loader"
            package.mkdir(parents=True)
            for parent in (root / "vllm", root / "vllm" / "model_executor", package):
                (parent / "__init__.py").write_text("")
            (package / "utils.py").write_text(
                "from contextlib import nullcontext\n"
                "class QuantizeMethodBase:\n"
                " def process_weights_after_loading(self, layer): return None\n"
                "def maybe_retie_word_embeddings(model, config): return None\n"
                "def device_loading_context(module, device): return nullcontext(module)\n"
                "def process_weights_after_loading(model, config, device): return None\n"
            )
            (package / "base_loader.py").write_text(
                "from .utils import process_weights_after_loading\n"
            )
            (package / "quant_impl.py").write_text(
                "from .utils import QuantizeMethodBase\n"
                "class RealLikeQuant(QuantizeMethodBase):\n"
                " def process_weights_after_loading(self, layer): return None\n"
            )
            b12x = root / "b12x"
            csf_prepare = b12x / "_lib" / "quant"
            csf_consumer = b12x / "moe" / "fused_moe"
            for parent in (b12x, b12x / "_lib", csf_prepare,
                           b12x / "moe", b12x / "moe" / "fused_moe"):
                parent.mkdir(parents=True, exist_ok=True)
                (parent / "__init__.py").write_text("")
            (csf_prepare / "_csf_prepare.py").write_text(
                "def validate_source(source): return source\n"
                "def expand_source_scales(source, output, *, swizzled): return output\n"
                "def prepare_csf_payload(source, **kwargs): return source\n"
            )
            (csf_consumer / "_csf_preparation.py").write_text(
                "from b12x._lib.quant._csf_prepare import validate_source\n"
                "def prepare_csf_block_weights(plan, weights): return validate_source(weights)\n"
            )
            env = dict(os.environ)
            env["QWEN38_MEMORY_DIAGNOSTICS"] = "1"
            env["PYTHONPATH"] = os.pathsep.join((str(hooks), str(root)))
            code = (
                "import os; from vllm.model_executor.model_loader import utils,base_loader,quant_impl; "
                "print(base_loader.process_weights_after_loading is utils.process_weights_after_loading); "
                "print(os.getpid()); "
                "m=quant_impl.RealLikeQuant(); m.process_weights_after_loading(object()); "
                "from b12x.moe.fused_moe import _csf_preparation as cp; "
                "from b12x._lib.quant import _csf_prepare as hp; "
                "print(cp.validate_source is hp.validate_source); "
                "print(cp.prepare_csf_block_weights(None,'probe'))"
            )
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                  text=True, env=env, check=True)
            stdout = proc.stdout.splitlines()
            self.assertEqual(stdout[0], "True")
            self.assertIn('"event": "HOOK_INSTALLED"', proc.stderr)
            self.assertIn('"event": "LOADER_ALIAS_PATCHED"', proc.stderr)
            self.assertIn('"event": "ENTER"', proc.stderr)
            self.assertIn('"event": "HELPERS_INSTALLED"', proc.stderr)
            self.assertEqual(stdout[2:], ["True", "probe"])
            self.assertIn('"helper": "b12x._lib.quant._csf_prepare.validate_source"', proc.stderr)
            self.assertIn('"helper": "b12x.moe.fused_moe._csf_preparation.prepare_csf_block_weights"', proc.stderr)
            marker = next(line for line in proc.stderr.splitlines() if '"event": "HOOK_INSTALLED"' in line)
            self.assertIn(f'"pid": {stdout[1]}', marker)

    def test_loader_alias_is_rewired_to_instrumented_utils_function(self):
        from types import SimpleNamespace
        from serve.diag_hooks import layer_diagnostics as ld

        raw = lambda *args: None
        utils = SimpleNamespace(process_weights_after_loading=lambda *args: None)
        loader = SimpleNamespace(process_weights_after_loading=raw)
        ld.rewire_base_loader_alias(loader, utils)
        self.assertIs(loader.process_weights_after_loading,
                      utils.process_weights_after_loading)

    def test_install_marker_includes_pid(self):
        from serve.diag_hooks import layer_diagnostics as ld
        import os

        events = []
        ld.emit_install_marker(emit=events.append)
        self.assertEqual(events[0]["event"], "HOOK_INSTALLED")
        self.assertEqual(events[0]["pid"], os.getpid())

    def test_fake_quant_hook_emits_enter_and_exit_with_memory_fields(self):
        from serve.diag_hooks import layer_diagnostics as ld

        events = []
        class Parameter:
            def numel(self): return 3
            def element_size(self): return 4
        class Layer:
            prefix = "model.layers.2.mlp"
            def named_parameters(self, recurse=True): return [("weight", Parameter())]
        class Method:
            def process_weights_after_loading(self, layer): layer.done = True

        wrapper = ld.wrap_quant_method(Method.process_weights_after_loading,
                                       emit=events.append,
                                       memory=lambda: {"host_memavailable_bytes": 99,
                                                       "torch_allocated_bytes": 1,
                                                       "torch_reserved_bytes": 2})
        layer = Layer()
        wrapper(Method(), layer)
        self.assertEqual([e["event"] for e in events], ["ENTER", "EXIT"])
        self.assertEqual(events[0]["layer"], "model.layers.2.mlp")
        self.assertEqual(events[0]["parameter_bytes"], 12)
        self.assertEqual(events[1]["torch_reserved_bytes"], 2)

    def test_fake_model_hook_emits_failure_and_flushable_enter(self):
        from serve.diag_hooks import layer_diagnostics as ld

        events = []
        def fail(model, config, device): raise RuntimeError("postload fault")
        wrapper = ld.wrap_model_postload(fail, emit=events.append,
                                         memory=lambda: {"host_memavailable_bytes": 7})
        class Model:
            def named_modules(self): return [("", self)]
        with self.assertRaisesRegex(RuntimeError, "postload fault"):
            wrapper(Model(), object(), "cpu")
        self.assertEqual([e["event"] for e in events], ["POSTLOAD_ENTER", "POSTLOAD_ERROR"])
        self.assertEqual(events[-1]["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
