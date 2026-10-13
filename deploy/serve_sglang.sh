#!/usr/bin/env bash
set -euo pipefail
# GB10 SGLang 서빙 (단일 GPU, TP=1)
MODEL="${MODEL:-khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4}"
python3 -m sglang.launch_server --model-path "$MODEL" --tp 1 --quantization modelopt_fp4 --host 0.0.0.0 --port 30000
