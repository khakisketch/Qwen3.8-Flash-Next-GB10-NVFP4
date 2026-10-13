# GB10 양자화/서빙 컨테이너 안내
# 원칙: 호스트 venv에 vLLM+SGLang+torch를 함께 설치하지 않음 (CUDA 충돌 방지).
# 양자화/서빙은 sm_121 지원이 고정된 NGC 컨테이너에서 수행.

# 예시 (양자화):
# docker run --rm --gpus all --ipc=host \
#   -v $PWD:/work -w /work \
#   nvcr.io/nvidia/pytorch:25.08-py3 \
#   .venv/bin/python -m quantization.streaming_convert --synthetic-test

# 예시 (vLLM 서빙):
# docker run --rm --gpus all --ipc=host -p 8000:8000 \
#   -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#   vllm/vllm-openai:latest \
#   vllm serve khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4 --tp 1 --gpu-memory-utilization 0.8
