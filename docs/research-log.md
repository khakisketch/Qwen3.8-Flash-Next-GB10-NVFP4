# 연구일지 — GB10 단일머신에 125B MoE 얹기

> 결론부터: **MoE는 NVFP4로, 51B PLE는 INT4로, 나머지는 BF16으로.** 샤드 스트리밍으로 360GB를 통째로 올리지 않고 바꾼다.
> 전부 GB10 실머신 + 합성 테스트 실측 기반. 원본 로그는 `docs/evidence/`에 보관.

## Day 1 — 인벤토리: 무엇이 무거운가

- `tools/manifest_inventory.py`로 HF 헤더만 긁어 전체 텐서 분류. 다운로드 없이 카테고리별 파라미터/바이트 집계.
- 핵심 발견: 무게는 **MoE experts + PLE 테이블**에 몰려 있다. 나머지는 BF16으로 둬도 된다.
- 결정: `configs/nvfp4_gb10.yaml`에 정책 고정 (experts 0.5B/param, PLE 0.625B/param, 민감층 2.0).

## Day 2 — 코덱 선정: 숫자 대결

- `quantization/ple_codecs.py` 7종 후보 → 합성 512×160 실측 (`docs/evidence/ple_codecs_selftest.txt`)
- bf16 2.0B/param rel 0.0016 vs **affine_uint4_g32 0.625B/param rel 0.0302** → PLE 채택.
- `quantization/ple_int4.py` 실측: affine_g32 rel 0.029~0.033, SNR 30dB (`ple_int4_selftest.txt`)
- `quantization/experts_nvfp4.py` 실측: NVFP4 E2M1 rel 약 0.10 (`experts_nvfp4_selftest.txt`)

## Day 3 — 스트리밍 변환: 메모리 안 터지게

- `quantization/streaming_convert.py` — 1샤드씩 읽고 바로 쓰고 버린다.
- 합성 2샤드 실측: `2→5 tensors`, PLE rel 0.03043, experts rel 0.09411 (`streaming_convert_synthetic.txt`)
- 산출물: 변환 샤드 + index.json + config.json + `ple_int4_manifest.json` (런타임이 NVMe에서 찾아먹는 지도)

## Day 4 — 런타임: PLE를 NVMe에 두기

- `runtime/ple_store.py` — NVMe safetensors 슬라이스 + LRU 핫로우 캐시, BF16으로 디퀀트.
- `runtime/patch_qwen4_exp.py` — 거대 nn.Embedding을 NVMe 백엔드로 교체. 해싱 로직은 원본 그대로라 결과 동일.
- `runtime/verify_ple_runtime.py` — store/모듈/통합 3단 검증 (실머신·풀체크포인트 필요 시 실행).

## Day 5 — GB10 실머신 확인

- `tools/gb10_doctor.py --skip-docker` 실측 (`docs/evidence/gb10_doctor.json`):
  - `arch_ok: true (aarch64)`, `gb10_detected: true`
  - RAM 127.6GB, torch 2.13.0+cu130, GPU NVIDIA GB10 130.7GB, cap 12.1, CUDA 13.0
- `kernels/fp4_microproof.py`, `kernels/capability_survey.py` — sm_121 FP4/INT4 GEMM 게이트 실험 (GB10에서 직접 실행).
- `tools/architecture_smoke_test.py` — 축소 config로 forward/generate 통과 확인용.

## 남은 일 (솔직하게)

- [ ] 풀 360GB 변환 후 vLLM 실서빙 계측 (TTFT/TPOT, 262K 컨텍스트)
- [ ] `scripts/accuracy_baseline.py`로 BF16 참조 대비 사람 평가
- [ ] HF 양자화 레포 공개 후 링크 고정

적용 예시(현장명 무관): 건설 현장 문서요약, CCTV 관제 보조 요약, 사내 지식 어시스턴트. 각 현장 보안·개인정보 규정 준수 전제.
