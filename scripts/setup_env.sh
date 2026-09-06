#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Environment prerequisite check for Qwen3.8-Flash-Next GB10 work.
#
# This script only *verifies* prerequisites and reports guidance. It does not
# install conflicting packages (vLLM, SGLang, torch) into a single venv.
#
# Dependency pinning for the experimental `qwen4_exp` architecture depends on
# the serving/quantization path chosen (NGC container vs. host venv) and is
# decided separately. See containers/ once that step is finalized.
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pass() { printf '  [OK]   %s\n' "$1"; }
warn() { printf '  [WARN] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; }

echo "=== Qwen3.8-Flash-Next GB10 Prerequisite Check ==="

# --- Architecture ---
case "$(uname -m)" in
    aarch64|arm64) pass "architecture: $(uname -m)";;
    *) warn "architecture: $(uname -m) (GB10 target is aarch64)";;
esac

# --- NVIDIA driver / CUDA ---
if command -v nvidia-smi &>/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA=$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9.]+' | head -1)
    NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    pass "GPU: $NAME"
    pass "driver: $DRIVER"
    if [[ "$CUDA" == 13* ]]; then pass "CUDA: $CUDA (sm_121/NVFP4 ready)";
    else warn "CUDA: $CUDA (13.x recommended for GB10 NVFP4)"; fi
else
    fail "nvidia-smi not found; install NVIDIA driver + CUDA toolkit"
fi

# --- Docker + container toolkit ---
command -v docker &>/dev/null && pass "docker: $(docker --version)" || warn "docker not found"
if command -v nvidia-ctk &>/dev/null; then pass "nvidia-container-toolkit present"; fi

# --- Python / tools ---
command -v python3 &>/dev/null && pass "python3: $(python3 --version 2>&1)" || fail "python3 missing"
python3 -c "import huggingface_hub; print('  [OK]   huggingface_hub ' + huggingface_hub.__version__)" 2>/dev/null \
    || warn "huggingface_hub not installed (pip install -U huggingface_hub)"
command -v hf &>/dev/null && pass "hf CLI: $(hf --version 2>&1 | head -1)" \
    || warn "hf CLI not on PATH"

echo ""
echo "=== Disk headroom ==="
df -h / | awk 'NR==2 {printf "  root free: %s of %s\n", $4, $2}'

echo ""
echo "=== Guidance ==="
echo "  Host venv (tools only, no CUDA):  python3 -m venv .venv && pip install -U huggingface_hub"
echo "  Quantization/serving:  use a pinned NGC container (nvcr.io) with sm_121 support;"
echo "  do NOT install vllm+sglang+torch together into one venv (CUDA/PyTorch conflicts)."
echo ""
echo "Prerequisite check complete."