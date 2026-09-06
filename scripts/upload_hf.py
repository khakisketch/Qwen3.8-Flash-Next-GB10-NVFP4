#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Upload Quantized Model to Hugging Face
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANTIZED_DIR="${PROJECT_DIR}/quantized"

echo "=== Hugging Face Upload Script ==="

# Check required variables
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN environment variable not set"
    echo "Set it with: export HF_TOKEN=your_hf_token"
    exit 1
fi

if [[ -z "${1:-}" ]]; then
    echo "ERROR: Missing repository ID argument"
    echo "Usage: $0 <repo-id> [model-name]"
    echo "Example: $0 my-org/qwen3.8-flash-next-gb10 'Qwen 3.8 Flash Next GB10 NVFP4'"
    exit 1
fi

REPO_ID="$1"
MODEL_NAME="${2:-Qwen3.8-Flash-Next GB10 NVFP4}"

echo "Target repository: $REPO_ID"
echo "Model name: $MODEL_NAME"

# Check quantized model exists
if [[ ! -d "$QUANTIZED_DIR" ]]; then
    echo "ERROR: Quantized model directory not found: $QUANTIZED_DIR"
    echo "Run quantize_nvfp4.py first"
    exit 1
fi

# Check directory contents
echo "Quantized model files:"
find "$QUANTIZED_DIR" -type f | head -20
echo "..."

# Login to Hugging Face
echo "Logging in to Hugging Face..."
hf login --token "$HF_TOKEN" 2>/dev/null || \
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# Create or update repository
echo "Creating/updating Hugging Face repository: $REPO_ID"
hf repo create "$REPO_ID" --type model --visibility private 2>/dev/null || \
    echo "Repository already exists or creating private repo"

# Upload files
echo "Uploading model files..."
hf upload "$REPO_ID" "$QUANTIZED_DIR" \
    --token "$HF_TOKEN" \
    --repo-type model \
    --commit-message "Upload NVFP4-quantized Qwen3.8-Flash-Next for GB10 \
(BF16→NVFP4, PLE INT4 packing, MoE experts quantization)" \
    --create-pr false

# Upload README if it exists
if [[ -f "$PROJECT_DIR/README.md" ]]; then
    echo "Uploading project README..."
    hf upload "$REPO_DIR" "$PROJECT_DIR/README.md" \
        --token "$HF_TOKEN" \
        --repo-type model \
        --path-in-repo "README.md" \
        --commit-message "Add project documentation"
fi

# Upload license if it exists
if [[ -f "$PROJECT_DIR/LICENSE" ]]; then
    echo "Uploading license..."
    hf upload "$REPO_ID" "$PROJECT_DIR/LICENSE" \
        --token "$HF_TOKEN" \
        --repo-type model \
        --path-in-repo "LICENSE" \
        --commit-message "Add Qwen Community License 1.0"
fi

# Verify upload
echo ""
echo "=== Verification ==="
hf repo view "$REPO_ID" --type model 2>/dev/null || echo "Could not view repo"

echo ""
echo "=== Upload Complete ==="
echo "Model available at: https://huggingface.co/$REPO_ID"
echo ""
echo "GB10 deployment commands:"
echo "  # vLLM"
echo "  vllm serve \$REPO_ID --tp 1 --gpu-memory-utilization 0.8"
echo ""
echo "  # SGLang"
echo "  sglang serve --model-path \$REPO_ID --tp 1 --quantization modelopt_fp4"
echo ""
echo "  # veloGB10"
echo "  ./gb10_inference --server --model-dir /path/to/checkpoint"