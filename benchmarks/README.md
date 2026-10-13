# Benchmarks — 합성 검증 결과 (실측)

> 전체 360GB 체크포인트 없이도 검증 가능한 합성 테스트 기준 실측값입니다.
> 원본 로그: `docs/evidence/` 참조. GB10 실머신에서 수행.

## 1) PLE INT4 (affine_uint4_g32)

| shape | sym_int4 rel_l2 | affine_g32 rel_l2 | affine SNR |
|---|---|---|---|
| [4096,160] | 0.07495 | **0.02968** | 30.6dB |
| [1000,64] | 0.06956 | **0.02957** | 30.6dB |
| [7,160] | 0.07575 | **0.03334** | 29.5dB |

- 재현: `.venv/bin/python -m quantization.ple_int4 --self-test`

## 2) MoE NVFP4 (E2M1FN, block_k=16)

| shape [E,K,N] | rel_l2 |
|---|---|
| [8,128,64] | 0.1027 |
| [4,160,32] | 0.0977 |
| [2,96,48] | 0.1012 |

- 재현: `.venv/bin/python -m quantization.experts_nvfp4 --self-test`

## 3) 샤드 스트리밍 변환 (합성 2샤드)

- `model-00001: 2 -> 5 tensors (1.7 MB)`
- `model-00002: 1 -> 1 tensors (26.2 MB)`
- PLE round-trip rel_l2=0.03043, experts rel_l2=0.09411
- 재현: `.venv/bin/python -m quantization.streaming_convert --synthetic-test`

## 4) 코덱 7종 비교 (합성 512x160)

| codec | bytes/elem | rel_l2 |
|---|---|---|
| bf16 | 2.000 | 0.001651 |
| fp8_row | 1.012 | 0.02687 |
| affine_uint4_g32 (채택) | 0.625 | 0.0302 |
| sym_int4 | 0.512 | 0.07561 |

- 재현: `.venv/bin/python -m quantization.ple_codecs`
