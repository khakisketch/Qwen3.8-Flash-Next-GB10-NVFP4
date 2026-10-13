#!/usr/bin/env bash
set -euo pipefail
# GB10 vLLM 서빙 (단일 GPU, TP=1)
MODEL="${MODEL:-khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4}"
vllm serve "$MODEL" --tp 1 --gpu-memory-utilization 0.8 --max-model-len 262144 --trust-remote-code
