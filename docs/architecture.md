# 아키텍처

![아키텍처](images/architecture.svg)
![파이프라인](images/pipeline.svg)
![용량 설계](images/memory.svg)

## 원본 모델 (인용)

- 출처: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- 125B MoE 본체(48층, GDN+QSA 하이브리드) + 51B PLE n-gram + MTP 1층 + 비전 27층
- 토큰당 활성 약 6B (routed 10 + shared 1), 네이티브 262K 컨텍스트
- BF16 약 360GB — GB10 128GB에 그대로 못 들어감

## 우리 설계 (실측 기반)

```mermaid
flowchart LR
  A[원본 BF16<br/>약 360GB] --> B[샤드 스트리밍 변환<br/>1샤드씩]
  B --> C[MoE NVFP4 E2M1<br/>0.5B/param]
  B --> D[PLE INT4 affine g32<br/>0.625B/param]
  B --> E[민감층 BF16 유지]
  C --> F[GB10 상주]
  D --> G[NVMe + LRU 캐시]
  E --> F
  F --> H[vLLM / SGLang TP=1]
  G --> H
```

- 양자화 정책: `configs/nvfp4.yaml`
- 변환기: `quantization/streaming_convert.py` (합성 실측 rel 0.03/0.09)
- PLE 런타임: `runtime/ple_store.py` + `runtime/patch_qwen4_exp.py`
- 하드웨어 게이트: `tools/gb10_doctor.py`, `kernels/fp4_microproof.py`

## 서빙

- vLLM: `deploy/serve_vllm.sh`, `configs/serve_vllm.yaml`
- SGLang: `deploy/serve_sglang.sh`, `configs/serve_sglang.yaml`
- Compose: `deploy/docker-compose.gb10.yaml`
