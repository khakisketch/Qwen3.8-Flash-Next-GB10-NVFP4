---
language: ko
license: qwen-community-1.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - nvfp4
  - int4
  - moe
  - gb10
  - vllm
  - sglang
---

# Qwen3.8-Flash-Next-GB10-NVFP4

GB10(DGX Spark) 단일머신용 NVFP4 + PLE-INT4 양자화 모델입니다.
원본 [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (BF16, 약 360GB)를
샤드 스트리밍으로 변환하여 128GB 통합메모리에 상주시킵니다.

- MoE experts: NVFP4 E2M1 (block_k=16, 0.5B/param), 실측 rel_l2 약 0.09
- PLE n-gram 테이블: affine UINT4 g32 (0.625B/param), 실측 rel_l2 약 0.03, SNR 30dB
- 민감층(attention/router/norm/lm_head): BF16 유지
- 변환: `quantization/streaming_convert.py` (1샤드씩, 전체 적재 없음)
- PLE 런타임: NVMe + LRU 캐시 (`runtime/ple_store.py`)
- 서빙: vLLM / SGLang, TP=1 (본 GitHub 저장소 참조)

## 빠른 시작

```bash
# vLLM
vllm serve khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4 --tp 1 --gpu-memory-utilization 0.8

# SGLang
python3 -m sglang.launch_server --model-path khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4 --tp 1 --quantization modelopt_fp4
```

전체 파이프라인·검증 로그·연구일지는 GitHub를 보세요:
https://github.com/khakisketch/Qwen3.8-Flash-Next-GB10-NVFP4

## 라이선스

Qwen Community License 1.0. "AI Work Assistant" 상용 이용은 별도 Qwen 라이선스 필요.
제작: 주식회사 케이에스아이 (043-288-4860)
