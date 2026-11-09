"""Unit tests for serve.launch. Stdlib only; temp files under /tmp/opencode.

Never starts a real container: docker is always a fake ``run``/``popen``,
the cache is a synthetic tree, sockets are host-loopback only. Real-host
checks use only the read-only ``preflight``/``build`` CLI paths.
"""

from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from serve import launch
from serve.baseline import (
    EXPECTED_SHARD_NAMES,
    HF_REPO_DIRNAME,
    IMAGE_PINNED,
    SNAPSHOT_REF,
)
from serve.guard import Guard


def _tmpdir() -> str:
    os.makedirs("/tmp/opencode", exist_ok=True)
    return tempfile.mkdtemp(prefix="qwen38-launch-", dir="/tmp/opencode")


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _meminfo(mem_gib: float) -> str:
    return (
        "MemTotal:       127598836 kB\n"
        f"MemAvailable:    {int(mem_gib * 1024**3 / 1024)} kB\n"
    )


def _pressure(full_avg10: float = 0.0) -> str:
    return (
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
        f"full avg10={full_avg10:.2f} avg60=0.00 avg300=0.00 total=1\n"
    )


def _vmstat(oom: int = 0) -> str:
    return f"nr_free_pages 1\noom_kill {oom}\n"


def _safe_guard(tmp: str) -> Guard:
    mem = os.path.join(tmp, "meminfo")
    prs = os.path.join(tmp, "pressure")
    vm = os.path.join(tmp, "vmstat")
    _write(mem, _meminfo(64.0))
    _write(prs, _pressure())
    _write(vm, _vmstat())
    return Guard(meminfo_path=mem, pressure_path=prs, vmstat_path=vm)


def _fake_cache_root(tmp: str) -> str:
    """Synthetic hub root with the exact 21-shard baseline + valid index."""
    snap = os.path.join(tmp, "hub", HF_REPO_DIRNAME, "snapshots", SNAPSHOT_REF)
    os.makedirs(os.path.join(snap, "tensors"), exist_ok=True)
    _write(os.path.join(snap, "config.json"), "{}")
    weight_map = {
        f"layer.{i}.weight": f"tensors/{name}"
        for i, name in enumerate(sorted(EXPECTED_SHARD_NAMES))
    }
    _write(
        os.path.join(snap, "model.safetensors.index.json"),
        json.dumps({"metadata": {}, "weight_map": weight_map}),
    )
    for name in EXPECTED_SHARD_NAMES:
        _write(os.path.join(snap, "tensors", name), "")
    return os.path.join(tmp, "hub")


def _free_port() -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.close()
    return port


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _ours_line(
    cid="cid-ours",
    status="running",
    code=0,
    label=launch.PROJECT_LABEL_VALUE,
) -> str:
    return f"{cid}|/{launch.CONTAINER_NAME}|{label}|{status}|{code}"


def _recording_run(script):
    """Fake ``run`` appending every argv; responses from script(argv)."""
    calls: list = []

    def run(argv, timeout):
        calls.append(argv)
        return script(argv, calls)

    run.calls = calls  # type: ignore[attr-defined]
    return run


class EnvTests(unittest.TestCase):
    def test_subprocess_env_drops_credentials(self):
        with patch.dict(
            os.environ,
            {
                "HF_TOKEN": "secret",
                "HF_HUB_TOKEN": "secret",
                "AWS_SECRET_ACCESS_KEY": "secret",
                "MY_PASSWORD": "secret",
                "PATH": "/usr/bin:/bin",
            },
            clear=False,
        ):
            env = launch.sanitized_subprocess_env()
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("HF_HUB_TOKEN", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertNotIn("MY_PASSWORD", env)
        self.assertIn("PATH", env)

    def test_subprocess_env_uses_local_docker_without_user_config(self):
        with patch.dict(
            os.environ,
            {
                "HOME": "/some/user/home",
                "DOCKER_CONFIG": "/some/user/docker-config",
                "DOCKER_HOST": "tcp://remote.example:2376",
            },
            clear=False,
        ):
            env = launch.sanitized_subprocess_env()
        self.assertNotIn("HOME", env)
        self.assertEqual(env["DOCKER_CONFIG"], "/proc/self/fd")
        self.assertEqual(env["DOCKER_HOST"], "unix:///var/run/docker.sock")

    def test_container_env_offline_no_tokens_or_test_knobs(self):
        with patch.dict(os.environ, {}, clear=True):
            env = launch.container_env()
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        for k in env:
            self.assertNotRegex(
                k, r"(?i)token|secret|password|credential|QWEN38_"
            )
        self.assertEqual(env["VLLM_PLE_TABLE_MEMORY"], "disk")
        self.assertEqual(env["SAFETENSORS_FAST_GPU"], "1")

    def test_container_env_forwards_only_the_explicit_vllm_api_key(self):
        with patch.dict(
            os.environ,
            {"VLLM_API_KEY": "unit-test-key", "OPENAI_API_KEY": "must-not-leak"},
            clear=False,
        ):
            env = launch.container_env()
        self.assertEqual(env["VLLM_API_KEY"], "unit-test-key")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")


class CacheTests(unittest.TestCase):
    def test_resolve_ok_exact_21(self):
        root = _fake_cache_root(_tmpdir())
        cache = launch.resolve_snapshot_cache(root)
        self.assertEqual(cache.shard_count, 21)

    def test_resolve_20_shards_refused(self):
        tmp = _tmpdir()
        root = _fake_cache_root(tmp)
        victim = os.path.join(
            root, HF_REPO_DIRNAME, "snapshots", SNAPSHOT_REF,
            "tensors", sorted(EXPECTED_SHARD_NAMES)[0],
        )
        os.remove(victim)
        with self.assertRaises(launch.LaunchError) as ctx:
            launch.resolve_snapshot_cache(root)
        self.assertIn("missing", str(ctx.exception))

    def test_resolve_22nd_extra_shard_refused(self):
        tmp = _tmpdir()
        root = _fake_cache_root(tmp)
        _write(
            os.path.join(
                root, HF_REPO_DIRNAME, "snapshots", SNAPSHOT_REF,
                "tensors", "model-99999-of-00021.safetensors",
            ),
            "",
        )
        with self.assertRaises(launch.LaunchError) as ctx:
            launch.resolve_snapshot_cache(root)
        self.assertIn("extra", str(ctx.exception))

    def test_resolve_broken_shard_symlink_refused(self):
        tmp = _tmpdir()
        root = _fake_cache_root(tmp)
        victim = os.path.join(
            root, HF_REPO_DIRNAME, "snapshots", SNAPSHOT_REF,
            "tensors", sorted(EXPECTED_SHARD_NAMES)[0],
        )
        os.remove(victim)
        os.symlink(
            os.path.join(tmp, "does-not-exist.safetensors"), victim
        )
        with self.assertRaises(launch.LaunchError) as ctx:
            launch.resolve_snapshot_cache(root)
        self.assertIn("broken", str(ctx.exception))

    def test_resolve_index_mismatch_refused(self):
        tmp = _tmpdir()
        root = _fake_cache_root(tmp)
        index = os.path.join(
            root, HF_REPO_DIRNAME, "snapshots", SNAPSHOT_REF,
            "model.safetensors.index.json",
        )
        from pathlib import Path
        payload = json.loads(Path(index).read_text(encoding="utf-8"))
        payload["weight_map"] = {
            k: v
            for k, v in list(payload["weight_map"].items())[:5]
        }
        _write(index, json.dumps(payload))
        with self.assertRaises(launch.LaunchError):
            launch.resolve_snapshot_cache(root)

    def test_resolve_index_unparseable_refused(self):
        tmp = _tmpdir()
        root = _fake_cache_root(tmp)
        _write(
            os.path.join(
                root, HF_REPO_DIRNAME, "snapshots", SNAPSHOT_REF,
                "model.safetensors.index.json",
            ),
            "{not json",
        )
        with self.assertRaises(launch.LaunchError):
            launch.resolve_snapshot_cache(root)

    def test_resolve_snapshot_symlink_escape_refused(self):
        tmp = _tmpdir()
        outside = os.path.join(tmp, "outside")
        os.makedirs(os.path.join(outside, "tensors"))
        _write(os.path.join(outside, "config.json"), "{}")
        _write(os.path.join(outside, "model.safetensors.index.json"), "{}")
        _write(
            os.path.join(outside, "tensors", "model-00001-of-00021.safetensors"),
            "",
        )
        root = os.path.join(tmp, "hub")
        link_parent = os.path.join(root, HF_REPO_DIRNAME, "snapshots")
        os.makedirs(link_parent)
        os.symlink(outside, os.path.join(link_parent, SNAPSHOT_REF))
        with self.assertRaises(launch.LaunchError):
            launch.resolve_snapshot_cache(root)

    def test_resolve_repo_symlink_escape_refused(self):
        tmp = _tmpdir()
        outside = os.path.join(tmp, "outside-repo")
        snap = os.path.join(outside, "snapshots", SNAPSHOT_REF)
        os.makedirs(os.path.join(snap, "tensors"))
        _write(os.path.join(snap, "config.json"), "{}")
        _write(os.path.join(snap, "model.safetensors.index.json"), "{}")
        _write(
            os.path.join(snap, "tensors", "model-00001-of-00021.safetensors"),
            "",
        )
        root = os.path.join(tmp, "hub")
        os.makedirs(root)
        os.symlink(outside, os.path.join(root, HF_REPO_DIRNAME))
        with self.assertRaises(launch.LaunchError):
            launch.resolve_snapshot_cache(root)


class BuildTests(unittest.TestCase):
    def _argv(self) -> list[str]:
        root = _fake_cache_root(_tmpdir())
        return launch.build_docker_argv(launch.resolve_snapshot_cache(root))

    def test_digest_pinned_never_latest(self):
        argv = self._argv()
        images = [a for a in argv if a.startswith("eugr/")]
        self.assertEqual(len(images), 1)
        self.assertIn("@sha256:", images[0])
        self.assertNotIn(":latest", " ".join(argv))
        self.assertEqual(images[0], IMAGE_PINNED)

    def test_user_scope_host_net_ipc_unconfined(self):
        argv = self._argv()
        self.assertIn("--network", argv)
        self.assertIn("host", argv)
        self.assertIn("--ipc", argv)
        blob = " ".join(argv)
        self.assertIn("seccomp=unconfined", blob)
        self.assertIn("apparmor=unconfined", blob)

    def test_loopback_api_no_publish_no_restart(self):
        argv = self._argv()
        self.assertNotIn("-p", argv)
        self.assertNotIn("--restart", argv)
        self.assertNotIn("0.0.0.0", argv)
        i = argv.index("--host")
        self.assertEqual(argv[i + 1], "127.0.0.1")

    def test_watchdog_entrypoint_and_cmd_shape(self):
        argv = self._argv()
        self.assertIn("--entrypoint", argv)
        self.assertEqual(argv[argv.index("--entrypoint") + 1], "python3")
        i = argv.index("/watch/qwen38-watchdog.py")
        tail = argv[i:]
        self.assertEqual(
            tail[0:5],
            [
                "/watch/qwen38-watchdog.py",
                "/opt/nvidia/nvidia_entrypoint.sh",
                "VLLM",
                "serve",
                "nvidia/Qwen3.8-Flash-Next-FP4-Local",
            ],
        )
        mounts = [a for a in argv if ":ro" in a]
        self.assertEqual(len(mounts), 2)  # watchdog + repo
        blob = " ".join(argv)
        self.assertNotIn("QWEN38_PROC_ROOT", blob)
        self.assertNotIn("QWEN38_POLL_S", blob)

    def test_baseline_model_flags_preserved(self):
        blob = " ".join(self._argv())
        for flag in (
            "--max-model-len", "262144", "num_speculative_tokens",
            "--kv-cache-dtype", "fp8", "qwen3_xml", "--reasoning-parser",
            "qwen3", "--gpu-memory-utilization", "0.8", "--block-size",
            "16", "--max-num-seqs", "8", "--load-format", "b12x",
        ):
            self.assertIn(flag, blob)

    def test_no_shell_no_tokens(self):
        argv = self._argv()
        self.assertTrue(all(isinstance(a, str) for a in argv))
        self.assertNotIn("HF_TOKEN", " ".join(argv))

    def test_foreign_name_rejected(self):
        root = _fake_cache_root(_tmpdir())
        cache = launch.resolve_snapshot_cache(root)
        with self.assertRaises(launch.LaunchError):
            launch.build_docker_argv(cache, name="ollama")

    def test_latest_image_rejected(self):
        root = _fake_cache_root(_tmpdir())
        cache = launch.resolve_snapshot_cache(root)
        with self.assertRaises(launch.LaunchError):
            launch.build_docker_argv(
                cache, image="eugr/spark-vllm-b12x:latest"
            )

    def test_missing_watchdog_refused(self):
        root = _fake_cache_root(_tmpdir())
        cache = launch.resolve_snapshot_cache(root)
        with self.assertRaises(launch.LaunchError):
            launch.build_docker_argv(
                cache, watchdog_host=os.path.join(_tmpdir(), "nope.py")
            )


class IdentityTests(unittest.TestCase):
    def test_inspect_uses_format_never_full_json(self):
        def script(argv, calls):
            self.assertEqual(argv[0:2], ["docker", "inspect"])
            self.assertIn("--format", argv)
            self.assertIn(launch.DOCKER_INSPECT_FORMAT, argv)
            self.assertNotIn("Config.Env", " ".join(argv))
            return FakeCompleted(stdout=_ours_line("cid-1"))

        run = _recording_run(script)
        info = launch.inspect_exact(run=run)
        self.assertEqual(info.id, "cid-1")
        self.assertEqual(info.name, launch.CONTAINER_NAME)
        self.assertEqual(info.status, "running")

    def test_inspect_absence_is_none(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container: x"
            )
        )
        self.assertIsNone(launch.inspect_exact(run=run))

    def test_inspect_absence_newer_docker_wording(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1,
                stderr="error: no such object: qwen38-fp4-local-serve",
            )
        )
        self.assertIsNone(launch.inspect_exact(run=run))

    def test_inspect_failure_is_error_not_absence(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: daemon boom"
            )
        )
        with self.assertRaises(launch.InspectError):
            launch.inspect_exact(run=run)

    def test_inspect_garbage_is_error(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(stdout="not|enough")
        )
        with self.assertRaises(launch.InspectError):
            launch.inspect_exact(run=run)

    def test_stop_absent_reports_absent(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container"
            )
        )
        report = launch.stop_project_container(run=run)
        self.assertEqual(report["action"], "absent")
        self.assertEqual(len(run.calls), 1)

    def test_stop_immediate_kills_verified_id(self):
        def script(argv, calls):
            if argv[1] == "inspect":
                return FakeCompleted(stdout=_ours_line("cid-k"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)
        report = launch.stop_project_container(run=run, immediate=True)
        self.assertEqual(report["action"], "killed+removed")
        kinds = [c[1] for c in run.calls]
        self.assertEqual(kinds, ["inspect", "kill", "rm"])
        self.assertIn("cid-k", run.calls[1])
        self.assertIn("cid-k", run.calls[2])
        self.assertNotIn("stop", kinds)  # kill, not slow stop

    def test_stop_graceful_uses_stop_with_timeout(self):
        def script(argv, calls):
            if argv[1] == "inspect":
                return FakeCompleted(stdout=_ours_line("cid-g"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)
        report = launch.stop_project_container(run=run, immediate=False)
        self.assertEqual(report["action"], "stopped+removed")
        stop_argv = run.calls[1]
        self.assertEqual(stop_argv[0:2], ["docker", "stop"])
        self.assertIn("--time", stop_argv)
        self.assertIn("cid-g", stop_argv)

    def test_stop_foreign_label_refused_without_mutation(self):
        def script(argv, calls):
            return FakeCompleted(
                stdout=_ours_line("cid-x", label="other-project")
            )

        run = _recording_run(script)
        with self.assertRaises(launch.Refusal):
            launch.stop_project_container(run=run, immediate=True)
        self.assertEqual(len(run.calls), 1)

    def test_stop_missing_label_refused(self):
        def script(argv, calls):
            return FakeCompleted(stdout=_ours_line("cid-x", label="<no value>"))

        run = _recording_run(script)
        with self.assertRaises(launch.Refusal):
            launch.stop_project_container(run=run)

    def test_stop_inspect_failure_touches_nothing(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: daemon boom"
            )
        )
        with self.assertRaises(launch.InspectError):
            launch.stop_project_container(run=run, immediate=True)
        self.assertEqual(len(run.calls), 1)

    def test_assert_id_change_refused(self):
        info = launch.ScopedInfo(
            id="cid-new", name=launch.CONTAINER_NAME,
            label=launch.PROJECT_LABEL_VALUE, status="running", exit_code=0,
        )
        with self.assertRaises(launch.Refusal):
            launch._assert_ours(info, "cid-old")

    def test_kill_by_verified_id_vanished_raises(self):
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container"
            )
        )
        with self.assertRaises(launch.LaunchError) as ctx:
            launch._kill_ours_by_verified_id("cid-z", run, {}, "trip")
        self.assertIn("vanished", str(ctx.exception))
        self.assertEqual(len(run.calls), 1)  # inspect only, no kill

    def test_stop_replacement_id_refused(self):
        """Same name, different id at act time: refuse, touch nothing.

        Verify saw cid-A; by the time stop_project_container re-inspects,
        a same-name replacement cid-B is there. The pinned expected id
        must refuse instead of acting on the newcomer.
        """
        def script(argv, calls):
            if argv[1] == "inspect" and len(calls) == 1:
                return FakeCompleted(stdout=_ours_line("cid-A"))
            if argv[1] == "inspect":
                return FakeCompleted(stdout=_ours_line("cid-B"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)
        with self.assertRaises(launch.Refusal) as ctx:
            launch._kill_ours_by_verified_id("cid-A", run, {}, "trip")
        self.assertIn("changed", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertNotIn("kill", kinds)
        self.assertNotIn("stop", kinds)
        self.assertNotIn("rm", kinds)


class PreflightTests(unittest.TestCase):
    def _run_absent(self):
        return _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container"
            )
        )

    def test_port_occupied_refuses(self):
        tmp = _tmpdir()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            port = srv.getsockname()[1]
            with patch.object(
                launch, "resolve_snapshot_cache",
                return_value=launch.resolve_snapshot_cache(
                    _fake_cache_root(tmp)
                ),
            ):
                with self.assertRaises(launch.LaunchError) as ctx:
                    launch.preflight(
                        _safe_guard(tmp), run=self._run_absent(), port=port
                    )
            self.assertIn("already answers", str(ctx.exception))
        finally:
            srv.close()

    def test_preflight_ok(self):
        tmp = _tmpdir()
        port = _free_port()
        with patch.object(
            launch, "resolve_snapshot_cache",
            return_value=launch.resolve_snapshot_cache(
                _fake_cache_root(tmp)
            ),
        ):
            report = launch.preflight(
                _safe_guard(tmp), run=self._run_absent(), port=port
            )
        self.assertEqual(report["shards"], 21)
        self.assertIsNotNone(report["oom_baseline"])

    def test_stale_container_refuses(self):
        tmp = _tmpdir()
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                stdout=_ours_line("cid-stale", status="exited", code=2)
            )
        )
        with patch.object(
            launch, "resolve_snapshot_cache",
            return_value=launch.resolve_snapshot_cache(
                _fake_cache_root(tmp)
            ),
        ):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.preflight(
                    _safe_guard(tmp), run=run, port=_free_port()
                )
        self.assertIn("still exists", str(ctx.exception))

    def test_unverifiable_stale_refuses(self):
        tmp = _tmpdir()
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: daemon boom"
            )
        )
        with patch.object(
            launch, "resolve_snapshot_cache",
            return_value=launch.resolve_snapshot_cache(
                _fake_cache_root(tmp)
            ),
        ):
            with self.assertRaises(launch.LaunchError):
                launch.preflight(
                    _safe_guard(tmp), run=run, port=_free_port()
                )
        self.assertEqual(len(run.calls), 1)  # inspect only, untouched


class _TripCounter(Guard):
    """Guard safe for the first `safe_samples`, then violating."""

    calls = 0

    def __init__(self, *a, safe_samples=1, **k):
        super().__init__(*a, **k)
        self.safe_samples = safe_samples
        type(self).calls = 0

    def sample(self):  # type: ignore[override]
        type(self).calls += 1
        base = super().sample()
        if type(self).calls <= self.safe_samples:
            return base
        from serve.guard import Sample as S

        return S(
            mem_available_bytes=1 * 1024**3,
            psi_full_avg10=base.psi_full_avg10,
            oom_kill=base.oom_kill,
        )


def _counter_guard(tmp: str, safe_samples: int) -> _TripCounter:
    mem = os.path.join(tmp, "meminfo")
    prs = os.path.join(tmp, "pressure")
    vm = os.path.join(tmp, "vmstat")
    _write(mem, _meminfo(64.0))
    _write(prs, _pressure())
    _write(vm, _vmstat())
    return _TripCounter(
        meminfo_path=mem, pressure_path=prs, vmstat_path=vm,
        safe_samples=safe_samples,
    )


class FakeProc:
    def __init__(self, rc=0, out="", hang=False):
        self.returncode = rc
        self._out = out
        self._hang = hang

    def communicate(self, timeout=None):
        if self._hang:
            time.sleep(30)
            return ("", "")
        return (self._out, "")


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        # The machinery below is HARD-DISABLED in production
        # (_REAL_START_ENABLED=False, no runtime override). These tests
        # re-enable it in-process ONLY, with fake docker and synthetic
        # /proc, to keep verifying the retained supervision logic for a
        # future independently verified mechanism. DisableTests (below)
        # prove the shipped default never starts anything.
        gate = patch.object(launch, "_REAL_START_ENABLED", True)
        gate.start()
        self.addCleanup(gate.stop)

    def _patched_cache(self, tmp):
        return patch.object(
            launch, "resolve_snapshot_cache",
            return_value=launch.resolve_snapshot_cache(
                _fake_cache_root(tmp)
            ),
        )

    def test_unsafe_first_sample_never_runs_docker(self):
        """Monitor fires on its very first sample -> docker never runs.

        Sequencing: preflight sample safe, monitor first sample unsafe.
        """
        tmp = _tmpdir()
        events: list = []
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container"
            )
        )

        def fake_popen(*a, **k):
            events.append("docker-run")
            return FakeProc(out="cid-x\n")

        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True,
                    guard=_counter_guard(tmp, safe_samples=1),
                    popen=fake_popen, run=run,
                    probe=lambda: True,
                    probe_models=lambda: True,
                    monitor_poll_s=0.02, port=_free_port(),
                )
        self.assertIn("docker never ran", str(ctx.exception))
        self.assertNotIn("docker-run", events)  # sequencing: never started
        for argv in run.calls:
            self.assertEqual(argv[1], "inspect")  # read-only only

    def test_trip_during_hung_start_kills_nothing_unverified(self):
        """Breach while `docker run` hangs: no blind kill, honest report.

        Sequencing: preflight + monitor-first-sample safe, THEN the
        breach lands while the docker thread is still hung.
        """
        tmp = _tmpdir()
        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container"
            )
        )

        def fake_popen(*a, **k):
            return FakeProc(hang=True)

        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True,
                    guard=_counter_guard(tmp, safe_samples=2),
                    popen=fake_popen, run=run,
                    probe=lambda: True,
                    probe_models=lambda: True,
                    monitor_poll_s=0.02, port=_free_port(),
                    settle_grace_s=0.5,
                )
        self.assertIn("during container start", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertNotIn("kill", kinds)
        self.assertNotIn("rm", kinds)  # absent: nothing to kill
        # The settle notes travel on the exception, not lost on print.
        self.assertIn("absent", ctx.exception.report["cleanup"])

    def test_late_container_after_absent_check_is_killed(self):
        """A container materializing AFTER the absent check still dies.

        Sequencing: breach fires while docker hangs; first settle check
        sees absent; the start then completes within the grace window;
        the late id is bound by a fresh verify and killed.
        """
        tmp = _tmpdir()
        gate = threading.Event()
        state = {"n": 0}

        def script(argv, calls):
            if argv[1] == "inspect":
                state["n"] += 1
                if state["n"] <= 2:  # preflight + first settle check
                    return FakeCompleted(
                        returncode=1, stderr="Error: No such container"
                    )
                return FakeCompleted(stdout=_ours_line("cid-late"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)

        class LateProc:
            returncode = 0

            def communicate(self, timeout=None):
                gate.wait(30)
                return ("cid-late\n", "")

        def fake_popen(*a, **k):
            return LateProc()

        timer = threading.Timer(0.5, gate.set)
        timer.start()
        try:
            with self._patched_cache(tmp):
                with self.assertRaises(launch.LaunchError) as ctx:
                    launch.launch(
                        confirm_host_risk=True,
                        guard=_counter_guard(tmp, safe_samples=2),
                        popen=fake_popen, run=run,
                        probe=lambda: True,
                        probe_models=lambda: True,
                        monitor_poll_s=0.02, port=_free_port(),
                        settle_grace_s=3.0,
                    )
        finally:
            timer.cancel()
            gate.set()
        self.assertIn("during container start", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertIn("kill", kinds)
        self.assertIn("rm", kinds)
        for c in run.calls:
            if c[1] in ("kill", "rm"):
                self.assertIn("cid-late", c)
        self.assertIn(
            "late container killed+removed", ctx.exception.report["cleanup"]
        )

    def test_trip_while_serving_kills_verified_id(self):
        """Violation after ready: kill+rm target the bound id, in order."""
        tmp = _tmpdir()
        state = {"started": False}

        def script(argv, calls):
            if argv[1] == "inspect":
                if not state["started"]:
                    return FakeCompleted(
                        returncode=1, stderr="Error: No such container"
                    )
                return FakeCompleted(stdout=_ours_line("cid-srv"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)

        def fake_popen(*a, **k):
            state["started"] = True
            return FakeProc(out="cid-srv\n")

        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True,
                    guard=_counter_guard(tmp, safe_samples=4),
                    popen=fake_popen, run=run,
                    probe=lambda: True,
                    probe_models=lambda: True,
                    monitor_poll_s=0.02, port=_free_port(),
                )
        self.assertIn("guard tripped; container killed", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        # Sequencing: identity bound by inspect BEFORE the first kill.
        first_inspect = kinds.index("inspect")
        first_kill = kinds.index("kill")
        self.assertLess(first_inspect, first_kill)
        self.assertLess(first_kill, kinds.index("rm"))
        for c in run.calls:
            if c[1] in ("kill", "rm"):
                self.assertIn("cid-srv", c)

    def test_interrupt_kills_ours(self):
        tmp = _tmpdir()
        state = {"started": False, "n": 0}

        def script(argv, calls):
            if argv[1] == "inspect":
                if not state["started"]:
                    return FakeCompleted(
                        returncode=1, stderr="Error: No such container"
                    )
                return FakeCompleted(stdout=_ours_line("cid-int"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)

        def fake_popen(*a, **k):
            state["started"] = True
            return FakeProc(out="cid-int\n")

        def fake_probe():
            state["n"] += 1
            if state["n"] >= 2:
                raise KeyboardInterrupt()
            return True

        with self._patched_cache(tmp):
            with self.assertRaises(KeyboardInterrupt):
                launch.launch(
                    confirm_host_risk=True, guard=_safe_guard(tmp),
                    popen=fake_popen, run=run, probe=fake_probe,
                    probe_models=lambda: True,
                    monitor_poll_s=0.02, port=_free_port(),
                )
        kinds = [c[1] for c in run.calls]
        self.assertIn("kill", kinds)
        self.assertIn("rm", kinds)

    def test_early_exit_never_restarts_reaps_only(self):
        """Server exits rc=3 at once: failure, rm but never kill."""
        tmp = _tmpdir()
        state = {"started": False}

        def script(argv, calls):
            if argv[1] == "inspect":
                if not state["started"]:
                    return FakeCompleted(
                        returncode=1, stderr="Error: No such container"
                    )
                return FakeCompleted(
                    stdout=_ours_line("cid-early", status="exited", code=3)
                )
            return FakeCompleted(stdout="")

        run = _recording_run(script)

        def fake_popen(*a, **k):
            state["started"] = True
            return FakeProc(out="cid-early\n")

        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True, guard=_safe_guard(tmp),
                    popen=fake_popen, run=run,
                    probe=lambda: False,
                    probe_models=lambda: False,
                    monitor_poll_s=0.02, port=_free_port(),
                )
        self.assertIn("never restarting", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertNotIn("kill", kinds)
        self.assertIn("rm", kinds)

    def test_docker_run_failure_aborts_cleanly(self):
        tmp = _tmpdir()

        def fake_popen(*a, **k):
            proc = FakeProc(rc=1)
            proc.communicate = lambda timeout=None: ("", "boom")
            return proc

        run = _recording_run(
            lambda argv, calls: FakeCompleted(
                returncode=1, stderr="Error: No such container"
            )
        )
        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True, guard=_safe_guard(tmp),
                    popen=fake_popen, run=run,
                    probe=lambda: False,
                    probe_models=lambda: False,
                    monitor_poll_s=0.02, port=_free_port(),
                )
        self.assertIn("docker run failed", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertNotIn("kill", kinds)

    def test_unverifiable_identity_aborts_without_blind_kill(self):
        tmp = _tmpdir()
        state = {"started": False, "n": 0}

        def script(argv, calls):
            if argv[1] == "inspect" and state["started"]:
                state["n"] += 1
                if state["n"] == 1:
                    # Identity binds once; every later check fails.
                    return FakeCompleted(stdout=_ours_line("cid-unk"))
                return FakeCompleted(
                    returncode=1, stderr="Error: daemon boom"
                )
            if argv[1] == "inspect":
                return FakeCompleted(
                    returncode=1, stderr="Error: No such container"
                )
            return FakeCompleted(stdout="")

        run = _recording_run(script)

        def fake_popen(*a, **k):
            state["started"] = True
            return FakeProc(out="cid-unk\n")

        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True, guard=_safe_guard(tmp),
                    popen=fake_popen, run=run,
                    probe=lambda: False,
                    probe_models=lambda: False,
                    monitor_poll_s=0.02, port=_free_port(),
                )
        self.assertIn("identity unverifiable", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertNotIn("kill", kinds)
        self.assertNotIn("rm", kinds)

    def test_startup_timeout_kills(self):
        tmp = _tmpdir()
        state = {"started": False}

        def script(argv, calls):
            if argv[1] == "inspect":
                if not state["started"]:
                    return FakeCompleted(
                        returncode=1, stderr="Error: No such container"
                    )
                return FakeCompleted(stdout=_ours_line("cid-to"))
            return FakeCompleted(stdout="")

        run = _recording_run(script)

        def fake_popen(*a, **k):
            state["started"] = True
            return FakeProc(out="cid-to\n")

        with self._patched_cache(tmp):
            with self.assertRaises(launch.LaunchError) as ctx:
                launch.launch(
                    confirm_host_risk=True, guard=_safe_guard(tmp),
                    popen=fake_popen, run=run,
                    probe=lambda: False,
                    probe_models=lambda: False,
                    monitor_poll_s=0.02, port=_free_port(),
                    startup_timeout_s=0.6,
                )
        self.assertIn("startup timeout", str(ctx.exception))
        kinds = [c[1] for c in run.calls]
        self.assertIn("kill", kinds)
        self.assertIn("rm", kinds)


class DisableTests(unittest.TestCase):
    """Prove the hard disable: the confirmation flag can never start the
    real model. No docker, no subprocess, not even guard preflight."""

    def _recorders(self, tmp):
        run_calls: list = []
        popen_calls: list = []

        def run(argv, timeout):
            run_calls.append(argv)
            raise AssertionError("docker must never be invoked")

        def popen(*a, **k):
            popen_calls.append((a, k))
            raise AssertionError("subprocess must never be invoked")

        guard = _safe_guard(tmp)
        guard_calls: list = []
        orig_sample = guard.sample
        orig_check = guard.check_now
        orig_base = guard.capture_oom_baseline

        def sample(*a, **k):
            guard_calls.append("sample")
            return orig_sample(*a, **k)

        def check_now(*a, **k):
            guard_calls.append("check_now")
            return orig_check(*a, **k)

        def baseline(*a, **k):
            guard_calls.append("baseline")
            return orig_base(*a, **k)

        guard.sample = sample  # type: ignore[method-assign]
        guard.check_now = check_now  # type: ignore[method-assign]
        guard.capture_oom_baseline = baseline  # type: ignore[method-assign]
        return guard, run, popen, run_calls, popen_calls, guard_calls

    def test_flag_supplied_still_disabled_zero_calls(self):
        tmp = _tmpdir()
        guard, run, popen, run_calls, popen_calls, guard_calls = (
            self._recorders(tmp)
        )
        with self.assertRaises(launch.RealStartDisabled):
            launch.launch(
                confirm_host_risk=True, guard=guard,
                popen=popen, run=run, port=_free_port(),
            )
        self.assertEqual(run_calls, [])
        self.assertEqual(popen_calls, [])
        self.assertEqual(guard_calls, [])  # fails BEFORE preflight
        self.assertTrue(
            issubclass(launch.RealStartDisabled, launch.LaunchError)
        )

    def test_flag_absent_also_disabled(self):
        tmp = _tmpdir()
        guard, run, popen, run_calls, popen_calls, guard_calls = (
            self._recorders(tmp)
        )
        with self.assertRaises(launch.RealStartDisabled):
            launch.launch(guard=guard, popen=popen, run=run)
        self.assertEqual(run_calls, [])
        self.assertEqual(popen_calls, [])
        self.assertEqual(guard_calls, [])

    def test_cli_launch_disabled_without_docker(self):
        with patch.object(launch, "preflight") as mp:
            rc = launch.main(["launch", "--confirm-host-risk"])
        self.assertEqual(rc, 1)
        mp.assert_not_called()

    def test_cli_has_no_stop_command(self):
        with patch.object(launch, "stop_project_container") as stop, \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as result:
                launch.main(["stop"])
        self.assertEqual(result.exception.code, 2)
        stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
