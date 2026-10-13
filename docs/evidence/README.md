# Evidence — 원본 로그 보관소

> 포트폴리오의 숫자는 전부 여기서 재현됩니다. 가공된 스크린샷이 아닙니다.

| 파일 | 내용 | 재현 명령 |
|---|---|---|
| `ple_int4_selftest.txt` | PLE INT4 3형상 rel/SNR | `.venv/bin/python -m quantization.ple_int4 --self-test` |
| `experts_nvfp4_selftest.txt` | NVFP4 3형상 rel | `.venv/bin/python -m quantization.experts_nvfp4 --self-test` |
| `ple_codecs_selftest.txt` | 7종 코덱 비교 | `.venv/bin/python -m quantization.ple_codecs` |
| `streaming_convert_synthetic.txt` | 2샤드 변환 round-trip | `.venv/bin/python -m quantization.streaming_convert --synthetic-test` |
| `gb10_doctor.json/.txt` | GB10 실머신 판정 | `.venv/bin/python tools/gb10_doctor.py --skip-docker` |

- 측정일: 2026-10-13, 호스트 `aitopatom-332e`, GB10 실머신
- `gb10_detected: true`, `arch_ok: true`, RAM 127.6GB, torch 2.13.0+cu130, cap 12.1
