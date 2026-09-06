#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Download Qwen3.8-Flash-Next (BF16, 360GB) from Hugging Face
#
# Uses the official `hf` CLI (huggingface_hub 1.15+). Downloads are
# resumable/idempotent: re-run to verify and fill in any missing files.
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-Flash-Next}"
# Pin to a known-good commit for reproducibility. Override with MODEL_REVISION.
MODEL_REVISION="${MODEL_REVISION:-de4b8e4d43b917e7706784d8bb445c9af86a3540}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/original_model}"

echo "=== Downloading $MODEL_ID @ $MODEL_REVISION ==="
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# Optional token (model is public/gated:false, but token avoids rate limits).
HF_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    HF_ARGS+=(--token "$HF_TOKEN")
fi

hf download "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --local-dir "$OUTPUT_DIR" \
    "${HF_ARGS[@]}"

echo ""
echo "=== Verifying download (idempotent re-run) ==="
hf download "$MODEL_ID" \
    --revision "$MODEL_REVISION" \
    --local-dir "$OUTPUT_DIR" \
    "${HF_ARGS[@]}"

echo ""
echo "=== Download complete ==="
echo "Model: $OUTPUT_DIR"
du -sh "$OUTPUT_DIR"