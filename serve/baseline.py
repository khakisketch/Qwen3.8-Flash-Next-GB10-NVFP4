"""Frozen NVIDIA Spark baseline for Qwen3.8-Flash-Next-FP4-Local.

Source: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local
(DGX Spark section, retrieved 2026-10-28).

This module records the baseline as data. It performs no I/O, starts
nothing, and changes no settings on import. Model/performance settings
below are the NVIDIA Spark recipe; host-facing wiring carries explicit
local/user decisions recorded in DECISIONS_FROM_NVIDIA_EXAMPLE.

Current approved image (fetched ANONYMOUSLY by digest; the pre-existing
local ``:latest`` tag and all model files were left untouched):
  image ref : eugr/spark-vllm-b12x@sha256:57c24fc397c66d4d28a7d0d5f1d792934fd01a70f4574cff6315fb7bc329d5cb
  image id  : sha256:e919f534c4d9af67f14c3101f9e90438f0cc6294a117bb2bca0fe26cbdbcdb83
  vLLM      : 0.1.dev21830+g8244d1c44.d20261026
Prior observations (NOT authorization to modify anything) are kept as
OBSERVED_* records.
"""

from __future__ import annotations

MODEL_ID = "nvidia/Qwen3.8-Flash-Next-FP4-Local"
SNAPSHOT_REF = "b9ce18a6887e18c089fdea8509a33e63429a3300"

# Host model cache (existing, read-only from this project's perspective).
HF_HUB_HOST_PATH = "/home/ksi/models/hf-hub"
HF_REPO_DIRNAME = "models--nvidia--Qwen3.8-Flash-Next-FP4-Local"

# Pinned image (approved, anonymous digest fetch). ``:latest`` moves, so
# the launcher MUST use this digest reference, never ``:latest``.
IMAGE_PINNED = (
    "eugr/spark-vllm-b12x@sha256:"
    "57c24fc397c66d4d28a7d0d5f1d792934fd01a70f4574cff6315fb7bc329d5cb"
)
OBSERVED_NEW_IMAGE_ID = (
    "sha256:e919f534c4d9af67f14c3101f9e90438f0cc6294a117bb2bca0fe26cbdbcdb83"
)
OBSERVED_VLLM_VERSION = "0.1.dev21830+g8244d1c44.d20261026"

# Superseded local tag observations (kept for the record only).
OBSERVED_OLD_IMAGE_ID = (
    "sha256:a7f76ae9f18c7701c9cde5f72c606393d9a21ca94dd8e53b27589935f64fcbe2"
)
OBSERVED_OLD_REPO_DIGEST = (
    "sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"
)

OBSERVED_HOST_STATE = {
    # 21 weight shards under snapshots/<ref>/tensors/ at last check.
    "weight_shards": 21,
    "snapshot_ref": SNAPSHOT_REF,
    # Previous lower-resource boot (20K ctx / MTP 3 / 2 seq, NOT this
    # baseline) reached host MemAvailable ~6.12 GiB -- below the agreed
    # 18 GiB minimum. That run had no automatic stop guard.
    "prior_20k_mtp3_s2_min_memavailable_gib": 6.12,
    # Later host observation (10 s sampling): MemAvailable fell
    # 29.43 -> 14.39 GiB between two consecutive 10-second samples.
    # The samples do not show when or how quickly the drop occurred
    # between them; polling can miss a sudden change.
    "observed_memavailable_drop_gib_per_10s": (29.43, 14.39),
}

# Independently confirmed in tiny isolated no-network containers (no model
# started): the new image sees GB10, ``vllm serve --help=all`` lists all
# model-card flags, the cached revision resolves offline from a read-only
# mount of the repo (21 files), and Docker's default seccomp denies
# io_uring_setup (EPERM) while NVIDIA's seccomp/AppArmor options allow an
# empty io_uring to be created and closed. These validate basic
# prerequisites ONLY -- not the actual watchdog and not OOM coverage.
OBSERVED_IMAGE_EVIDENCE = {
    "gb10_visible": True,
    "vllm_serve_help_lists_card_flags": True,
    "offline_readonly_cache_resolves": True,
    "default_seccomp_denies_io_uring_setup_EPERM": True,
    "nvidia_seccomp_apparmor_allow_io_uring_create_close": True,
    "model_started": False,
    # New prerequisite evidence (harmless one-off containers, no model):
    "entrypoint_runs_python3_as_pid1_with_and_without_gpu": True,
    "exiting_python_pid1_with_30s_child_stops_container_in_024s": True,
    "image_under_2gib_cgroup_sees_host_proc_and_memory_events": True,
    # One KNOWN app-OOM path (code reading, not behavior): the pinned
    # image's vllm/v1/worker/gpu_model_runner.py catches
    # torch.cuda.OutOfMemoryError, logs "Failed to load model - not enough
    # GPU memory" with the original error, and re-raises. One path only:
    # does NOT guarantee every app OOM is observed.
    "vllm_model_runner_oom_log_and_reraise_one_path_only": True,
}

# Exact shard baseline for SNAPSHOT_REF: count AND names. The resolver
# requires exactly this set (plus index cross-check); anything else fails
# closed rather than serving a partial/foreign snapshot.
EXPECTED_SHARD_COUNT = 21
EXPECTED_SHARD_NAMES = frozenset(
    f"model-{i:05d}-of-00021.safetensors" for i in range(1, 22)
)

# Image entrypoint pieces (observed, read-only docker inspect).
IMAGE_ENTRYPOINT_SCRIPT = "/opt/nvidia/nvidia_entrypoint.sh"
# In-container watchdog mount (read-only). Resolved at runtime by the
# watchdog via PATH; missing binary fails closed (server never starts).
WATCHDOG_CONTAINER_PATH = "/watch/qwen38-watchdog.py"

# Exact vLLM server flags from the NVIDIA DGX Spark recipe.
# 262144 is the PER-REQUEST limit (input+output), not a guarantee of
# 8 x 262K concurrent requests against a shared ~1M-token KV cache.
VLLM_SERVE_ARGV = [
    "serve",
    MODEL_ID,
    "--host", "0.0.0.0",  # Overridden to 127.0.0.1 by launch.py (D1).
    "--port", "8000",
    "--tensor-parallel-size", "1",
    "--pipeline-parallel-size", "1",
    "--mamba-cache-mode", "align",
    "--enable-prefix-caching",
    "--enable-chunked-prefill",
    "--dtype", "bfloat16",
    "--kv-cache-dtype", "fp8",
    "--quantization", "modelopt_mixed",
    "--block-size", "16",
    "--load-format", "b12x",
    "--max-model-len", "262144",
    "--max-num-seqs", "8",
    "--max-num-batched-tokens", "4096",
    "--speculative-config", '{"method":"mtp","num_speculative_tokens":4}',
    "--gdn-decode-kernel", "b12x",
    "--linear-backend", "b12x",
    "--moe-backend", "b12x",
    "--no-enable-flashinfer-autotune",
    "--mm-encoder-tp-mode", "data",
    "--reasoning-parser", "qwen3",
    "--tool-call-parser", "qwen3_xml",
    "--enable-auto-tool-choice",
    "--compilation-config", '{"pass_config":{"fuse_act_quant":true}}',
    "--gpu-memory-utilization", "0.8",
]

# Exact container env from the NVIDIA DGX Spark recipe.
VLLM_CONTAINER_ENV = {
    "VLLM_PLE_TABLE_MEMORY": "disk",
    "B12X_DISK_BACKEND": "io_uring",
    "CUTE_DSL_ARCH": "sm_121a",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
    "VLLM_USE_AOT_COMPILE": "1",
    "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "B12X_POLICY_MODE": "auto",
    "VLLM_MXFP8_LM_HEAD": "1",
    "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE": "w13",
    "VLLM_B12X_MOE_FP4_FORCE_A16": "1",
    "B12X_W4A16_A4_PREFILL": "1",
}

# Host-facing decisions. The USER explicitly approved: current digest
# image (anonymous fetch), NVIDIA example seccomp+apparmor unconfined
# (io_uring evidence above; PLE disk backend needs io_uring), host
# IPC/network from the example. The API still binds localhost only.
DECISIONS_FROM_NVIDIA_EXAMPLE = (
    (
        "D1: vLLM --host 0.0.0.0",
        "bind 127.0.0.1 only (host network scope: container loopback IS "
        "host loopback, so the direct listener binds host localhost; "
        "host proxies or forwards need separate review)",
        "Restrict the direct listener on a shared machine; other host "
        "forwarding is outside this check.",
    ),
    (
        "D2: docker --network host",
        "KEPT as user-approved scope (no -p port mapping; host networking "
        "ignores -p); locality enforced by D1 instead",
        "User decision; with --network host the container loopback is the "
        "host loopback, so D1 is the mechanism that keeps the API local.",
    ),
    (
        "D3: docker --ipc host",
        "KEPT as user-approved scope",
        "User decision.",
    ),
    (
        "D4: --security-opt seccomp=unconfined + apparmor=unconfined",
        "KEPT as user explicitly chose (narrower profile rejected)",
        "User decision, grounded in evidence: default seccomp denies "
        "io_uring_setup (EPERM); NVIDIA options allow io_uring "
        "create+close; the PLE disk backend needs io_uring.",
    ),
    (
        "D5: -e HF_TOKEN=$HF_TOKEN with rw cache mounts",
        "no credential env; HF_HUB_OFFLINE=1; snapshot repo mounted "
        "read-only (:ro)",
        "Weights already cached; the server must not need secrets or be "
        "able to write the shared cache. Confirmed offline resolution "
        "from the read-only mount (21 files).",
    ),
    (
        "D6: interactive --entrypoint /bin/bash, manual launch",
        "supervised non-interactive launch THROUGH the image's own "
        "entrypoint script with an explicit server command",
        "The in-container PID-1 watchdog spawns "
        "/opt/nvidia/nvidia_entrypoint.sh with an explicit vLLM binary "
        "(resolved via PATH in-container, fail-closed) plus the card "
        "flags. The image entrypoint forwarded a harmless Python command "
        "in a probe; forwarding this watchdog/server combination remains "
        "unverified.",
    ),
    (
        "D7: nothing extra mounted",
        "one additional read-only file mount: the PID-1 watchdog script "
        "(/watch/qwen38-watchdog.py, :ro) + entrypoint python3",
        "Intended to keep monitoring if the host supervisor dies. "
        "Simultaneous failures or a hung watchdog are not ruled out, and "
        "application/CUDA OOM coverage is incomplete.",
    ),
)

# Agreed guard limits. MUST NOT be weakened (see guard.py).
MEM_AVAILABLE_MIN_GIB = 18
PSI_FULL_AVG10_MAX_PCT = 10.0
OOM_POLICY = (
    "immediate stop on any change of the host KERNEL oom_kill counter; "
    "application/CUDA OOM errors do NOT increment that counter and are "
    "covered only indirectly (host pressure/memory trip, server crash "
    "exit, health-loss trip) -- see serve/README.md"
)
