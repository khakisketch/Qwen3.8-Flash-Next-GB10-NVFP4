"""Opt-in, one-shot diagnostic runner under the normal 18 GiB guard."""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

from . import launch
from .baseline import IMAGE_PINNED, MODEL_ID
from .trial import TrialRunner, _trial_argv

DIAGNOSTIC_MIN_GIB = 18
CONTAINER_HOOK_DIR = "/watch/diag"
# Only profiles that have a measured host-memory envelope are accepted.
ALLOWED_CONTEXTS = (32768, 65536)
ALLOWED_SEQUENCES = (1, 2)


class DiagnosticFailure(RuntimeError):
    pass


class DiagnosticGuard:
    """Strict 18 GiB/10%/OOM policy used from preflight onward."""
    def __init__(self, oom_baseline: int):
        self.oom_baseline = oom_baseline

    def check(self, sample):
        if sample is None:
            raise DiagnosticFailure("host guard sample unavailable")
        issues = []
        if sample.mem_available_bytes < DIAGNOSTIC_MIN_GIB * 1024**3:
            issues.append("MemAvailable below 18 GiB")
        if sample.psi_full_avg10 >= 10:
            issues.append("memory PSI full avg10 at/above 10%")
        if sample.oom_kill != self.oom_baseline:
            issues.append("kernel oom_kill changed")
        if issues:
            raise DiagnosticFailure("; ".join(issues))


def diagnostic_argv(cache, watchdog_path: Path, hooks_path: Path,
                    heartbeat_path: Path, *, reclaim_csf_cache: bool = False,
                    kv_cache_memory_bytes: int | None = None,
                    kernel_cache_dir: str | Path | None = None,
                    listen_all: bool = False,
                    context: int = 32768,
                    sequences: int = 1) -> list[str]:
    """Build a guarded serving profile with the post-load instrumentation."""
    if context not in ALLOWED_CONTEXTS:
        raise DiagnosticFailure(
            f"context must be one of {ALLOWED_CONTEXTS}, got {context!r}"
        )
    if sequences not in ALLOWED_SEQUENCES:
        raise DiagnosticFailure(
            f"sequences must be one of {ALLOWED_SEQUENCES}, got {sequences!r}"
        )
    argv = _trial_argv(cache, watchdog_path, heartbeat_path)
    try:
        host_index = argv.index("--host")
        argv[host_index + 1] = "0.0.0.0" if listen_all else launch.HOST_BIND
        argv[argv.index("--max-model-len") + 1] = str(context)
        argv[argv.index("--max-num-seqs") + 1] = str(sequences)
    except (ValueError, IndexError) as exc:
        raise DiagnosticFailure(f"server argv missing a profile value: {exc}") from exc
    hooks_path = Path(hooks_path).resolve(strict=True)
    if not hooks_path.is_dir() or not (hooks_path / "sitecustomize.py").is_file():
        raise DiagnosticFailure("diagnostics hook directory/sitecustomize.py missing")
    try:
        image_index = argv.index(IMAGE_PINNED)
    except ValueError as exc:
        raise DiagnosticFailure("pinned image missing from docker argv") from exc
    extra = [
        "-e", "QWEN38_MEMORY_DIAGNOSTICS=1",
        "-e", f"PYTHONPATH={CONTAINER_HOOK_DIR}",
        "-v", f"{hooks_path}:{CONTAINER_HOOK_DIR}:ro",
    ]
    if reclaim_csf_cache:
        extra += ["-e", "QWEN38_RECLAIM_CSF_CACHE=1"]
    if kv_cache_memory_bytes is not None:
        if isinstance(kv_cache_memory_bytes, bool) or not isinstance(kv_cache_memory_bytes, int) or kv_cache_memory_bytes <= 0:
            raise DiagnosticFailure("--kv-cache-memory-bytes must be a positive integer")
    kernel_cache_mounts = []
    if kernel_cache_dir is not None:
        root = Path(kernel_cache_dir)
        try:
            root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise DiagnosticFailure(f"kernel cache directory is unavailable: {exc}") from exc
        if not root.is_dir():
            raise DiagnosticFailure("kernel cache path must be an existing directory")
        for child_name, target in (
            ("b12x", "/root/.cache/b12x"),
            ("triton", "/root/.triton"),
        ):
            child = root / child_name
            try:
                resolved_child = child.resolve(strict=True)
                resolved_child.relative_to(root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DiagnosticFailure(
                    f"kernel cache must contain an in-directory {child_name!r} subdirectory"
                ) from exc
            if not resolved_child.is_dir():
                raise DiagnosticFailure(f"kernel cache {child_name!r} path is not a directory")
            kernel_cache_mounts += ["-v", f"{resolved_child}:{target}:rw"]
    extra += kernel_cache_mounts
    argv[image_index:image_index] = extra
    if kv_cache_memory_bytes is not None:
        try:
            model_index = argv.index(MODEL_ID, image_index)
        except ValueError as exc:
            raise DiagnosticFailure("expected model ID missing from server argv") from exc
        argv[model_index + 1:model_index + 1] = [
            "--kv-cache-memory-bytes", str(kv_cache_memory_bytes),
        ]
    if argv.index(IMAGE_PINNED) < argv.index("QWEN38_MEMORY_DIAGNOSTICS=1"):
        raise DiagnosticFailure("diagnostic environment must precede image")
    if argv[argv.index("--max-model-len") + 1] != str(context):
        raise DiagnosticFailure("context flag was not applied")
    if argv[argv.index("--max-num-seqs") + 1] != str(sequences):
        raise DiagnosticFailure("sequence flag was not applied")
    return argv


class DiagnosticRunner(TrialRunner):
    """One-shot supervised diagnostic; inherits lifecycle, uses 18 GiB always."""
    def __init__(self, *, hooks_path: str | Path | None = None,
                 reclaim_csf_cache: bool = False,
                 kv_cache_memory_bytes: int | None = None,
                 kernel_cache_dir: str | Path | None = None,
                 listen_all: bool = False,
                 context: int = 32768,
                 sequences: int = 1,
                 **kwargs):
        self.hooks_path = Path(hooks_path or Path(__file__).with_name("diag_hooks"))
        self.reclaim_csf_cache = bool(reclaim_csf_cache)
        self.kv_cache_memory_bytes = kv_cache_memory_bytes
        self.kernel_cache_dir = None if kernel_cache_dir is None else str(kernel_cache_dir)
        self.listen_all = bool(listen_all)
        self.context = context
        self.sequences = sequences
        def builder(cache, watchdog, heartbeat):
            return diagnostic_argv(
                cache, watchdog, self.hooks_path, heartbeat,
                reclaim_csf_cache=self.reclaim_csf_cache,
                kv_cache_memory_bytes=self.kv_cache_memory_bytes,
                kernel_cache_dir=self.kernel_cache_dir,
                listen_all=self.listen_all,
                context=self.context,
                sequences=self.sequences,
            )
        super().__init__(startup_min_gib=DIAGNOSTIC_MIN_GIB,
                         argv_builder=builder,
                         watchdog_path=Path(__file__).with_name("diagnostic_watchdog.py"),
                         outcome_metadata={"diagnostic_options": {
                             "reclaim_csf_cache": self.reclaim_csf_cache,
                             "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
                             "kernel_cache_dir": self.kernel_cache_dir,
                             "listen_all": self.listen_all,
                             "context": self.context,
                             "sequences": self.sequences,
                         }},
                         **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a guarded FP4 post-load diagnostic")
    parser.add_argument(
        "--listen-all", action="store_true",
        help="listen on all network interfaces instead of loopback",
    )
    parser.add_argument(
        "--reclaim-csf-cache", action="store_true",
        help="experimentally release unused CUDA cache at NVFP4 CSF preparation boundaries",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes", type=positive_int, default=None,
        help="optional explicit vLLM KV cache byte budget",
    )
    parser.add_argument(
        "--kernel-cache-dir", default=None,
        help="existing generated-kernel cache root containing b12x/ and triton/",
    )
    parser.add_argument(
        "--context", type=int, choices=ALLOWED_CONTEXTS, default=32768,
        help="per-request context length (measured profiles only)",
    )
    parser.add_argument(
        "--sequences", type=int, choices=ALLOWED_SEQUENCES, default=1,
        help="maximum concurrent sequences (measured profiles only)",
    )
    return parser


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    """Explicit opt-in entrypoint; never called by normal serving startup."""
    try:
        options = build_parser().parse_args(argv)
        result = DiagnosticRunner(
            reclaim_csf_cache=options.reclaim_csf_cache,
            kv_cache_memory_bytes=options.kv_cache_memory_bytes,
            kernel_cache_dir=options.kernel_cache_dir,
            listen_all=options.listen_all,
            context=options.context,
            sequences=options.sequences,
        ).execute()
        print(result)
    except KeyboardInterrupt:
        print("diagnostic run interrupted; evidence retained", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"diagnostic run failed; evidence retained: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

