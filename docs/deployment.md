# 배포 가이드 (GB10, 5분 시작)

## 0) 준비물 확인

```bash
bash scripts/setup_env.sh
.venv/bin/python tools/gb10_doctor.py --skip-docker
```

- 기대: `arch_ok true`, `gb10_detected true`, RAM 약 128GB, CUDA 13.x

## 1) 원본 다운로드 (재개 가능)

```bash
MODEL_ID=Qwen/Qwen3.8-Flash-Next bash scripts/download_model.sh
```

## 2) 합성 검증 (다운로드 없이도 됨)

```bash
.venv/bin/python -m quantization.ple_int4 --self-test
.venv/bin/python -m quantization.experts_nvfp4 --self-test
.venv/bin/python -m quantization.streaming_convert --synthetic-test
```

## 3) 양자화 (풀 체크포인트 필요, 10~30분+)

```bash
.venv/bin/python scripts/quantize_nvfp4.py \
  --model-path ./original_model \
  --output-dir ./quantized \
  --ple-pack --calib-samples 256
```

또는 경량 스트리밍 변환:

```bash
.venv/bin/python -m quantization.streaming_convert --src-dir ./original_model --out-dir ./quantized
```

## 4) Hugging Face 업로드

```bash
export HF_TOKEN=...
bash scripts/upload_hf.py khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4
```

- 베이스: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- 양자화 레포(예정/운영): https://huggingface.co/khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4

## 5) 서빙

```bash
# vLLM
bash deploy/serve_vllm.sh
# 또는
docker compose -f deploy/docker-compose.gb10.yaml up

# SGLang
bash deploy/serve_sglang.sh
```

## 트러블슈팅

- `torch` 없음 → `.venv/bin/python` 사용 (호스트 python 아님)
- vLLM/SGLang/torch 충돌 → 함께 설치 금지, NGC 컨테이너 사용 (`containers/README.md`)
- 디스크 부족 → `df -h /` 1TB+ 여유 필요 (풀 변환 시)
