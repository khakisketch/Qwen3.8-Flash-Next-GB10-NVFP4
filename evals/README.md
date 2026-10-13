# Evals — 품질 베이스라인 운영법

`scripts/accuracy_baseline.py`는 전체 BF16 체크포인트를 CPU/RAM 스필로 천천히 돌려
양자화 전 참조 출력을 뽑는 하네스입니다. GB10 128GB 환경에서는 다음처럼 소량으로 운영하세요.

```bash
.venv/bin/python scripts/accuracy_baseline.py \
  --model-path ./original_model \
  --prompts "84 * 3 / 2는?" "사내 문서 요약해줘" \
  --max-new-tokens 8 \
  --output docs/evidence/reference_outputs.json
```

- 기본 프롬프트는 `evals/reference_prompts.txt` 참조
- 양자화 후 동일 프롬프트로 재생성 → 사람이 읽고 동등성 판정 (자동 점수 아님)
- 대량 평가는 vLLM/SGLang 서빙 후 별도 스위트에서 수행
