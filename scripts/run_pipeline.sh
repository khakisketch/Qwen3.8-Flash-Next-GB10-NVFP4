#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Qwen3.8-Flash-Next Full Pipeline: Download → Quantize → Upload to HF
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "============================================================================="
echo "Qwen3.8-Flash-Next GB10 Optimization Pipeline"
echo "============================================================================="
echo ""

# Phase 1: Environment setup
echo "--- Phase 1: Environment Setup ---"
bash "$PROJECT_DIR/scripts/setup_env.sh"
echo ""

# Phase 2: Download model
echo "--- Phase 2: Downloading Original Model ---"
bash "$PROJECT_DIR/scripts/download_model.sh"
echo ""

# Phase 3: Quantization
echo "--- Phase 3: NVFP4 Quantization ---"
echo "This will take 10-30 minutes depending on calibration data."
python3 "$PROJECT_DIR/scripts/quantize_nvfp4.py" \
    --model-path "$PROJECT_DIR/original_model" \
    --output-dir "$PROJECT_DIR/quantized" \
    --ple-pack \
    --experts-only \
    --calib-samples 256
echo ""

# Phase 4: Upload to Hugging Face
echo "--- Phase 4: Upload to Hugging Face ---"
echo "Ensure HF_TOKEN is set in your environment"
echo "export HF_TOKEN=your_hf_token_here"
echo ""
read -p "Ready to upload? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 "$PROJECT_DIR/scripts/upload_hf.py" "$PROJECT_DIR/hf-repo-id" "Qwen 3.8 Flash Next GB10 NVFP4"
else
    echo "Skipping upload step."
fi

echo ""
echo "============================================================================="
echo "Pipeline Complete!"
echo "============================================================================="
echo ""
echo "Next steps for GB10 deployment:"
echo "1. Test with vLLM:"
echo "   vllm serve huggingface://<your-repo-id> --tp 1 --gpu-memory-utilization 0.8"
echo ""
echo "2. Test with SGLang:"
echo "   sglang serve --model-path <your-repo-id> --tp 1 --quantization modelopt_fp4"
echo ""
echo "3. veloGB10 engine:"
echo "   ./gb10_inference --server --model-dir <checkpoint-path>"
echo "============================================================================="