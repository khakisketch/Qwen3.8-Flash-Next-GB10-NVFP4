# NVIDIA Qwen3.8-Flash-Next FP4-Local on DGX Spark

단일 DGX Spark에서 NVIDIA의 공식
[`Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)을
systemd 상시 서비스로 운영하는 프로젝트입니다. 모델·이미지·캐시는 NVIDIA
모델 카드의 Spark 예제와 검증된 digest를 기준으로 고정합니다.

## 현재 운영 상태 (2026-11-09)

- 서비스: `qwen38-fp4.service` (`systemd --user`, `enabled`, linger 사용)
- API: `http://172.30.1.44:8000/v1` (LAN), `http://127.0.0.1:8000/v1`
- 모델 ID: `nvidia/Qwen3.8-Flash-Next-FP4-Local`
- 인증: `Authorization: Bearer <VLLM_API_KEY>` 필수. 키는 저장소 밖
  `~/.config/qwen38/qwen38.env`(0600)에만 있다.
- 실행 프로필: **256K(262,144) 컨텍스트**, 최대 2시퀀스,
  KV 10 GiB(531,950 tokens), 배치 4,096 tokens, MTP 4, FP8 KV,
  PLE 디스크 오프로드, `gpu-memory-utilization 0.8`.
- NVIDIA 카드의 예제는 262,144/8시퀀스입니다. 이 장비에서는
  256K/2시퀀스가 18 GiB 가드와 기동·문맥 시험을 통과해 운영 프로필로
  선택했습니다. 8시퀀스 전체는 검증하지 않았습니다.

```sh
./serve/systemd/install.sh          # 서비스 설치, 키가 없으면 생성
systemctl --user start qwen38-fp4.service
systemctl --user restart qwen38-fp4.service
systemctl --user stop qwen38-fp4.service
./serve/status.sh                   # 읽기 전용 상태 확인
journalctl --user -u qwen38-fp4.service -f
```

## 측정 결과

2026-11-09, 단일 DGX Spark에서 실제 모델로 측정했습니다. 소규모 표본이므로
벤치마크 순위가 아니라 운영 판단용 값입니다.

| 항목 | 결과 | 조건 |
|---|---|---|
| 256K/2 기동 | 285초 | 현재 프로필 |
| 64K/2 첫 기동 | 870초 | 역사 기록, 새 그래프/커널 포함 |
| 64K/2 재기동 | 272초 | 역사 기록, 캐시 재사용 |
| 비정상 종료 복구 | 336초 | 64K/2 시험, systemd 재시작 |
| 준비 후 `MemAvailable` | 약 25.98–26.5 GiB | 256K/2, KV 10 GiB |
| 기동 중 최저 `MemAvailable` | 약 23.61 GiB | 18 GiB 가드 통과 |
| TTFT | 0.294초 중앙값 | 256K/2, 스트리밍 3회 |
| 디코드 | 46.27 tok/s 중앙값 | 256K/2, 3회 |
| 긴 문맥 | 45,946토큰에서 정답 | 256K/2 needle 검색 |
| 동시 요청 | 2개 동시 완료 | 256K/2 |
| 반복 요청 | 20/20 성공, 0.37초 중앙값 | 256K/2 |
| OpenCode General | 25.2 / 20.7 / 19.0초, 테스트 2/4/2개 통과 | Spark에서 실제 버그 3건 수정 |

모델 로딩과 추론은 실제로 수행했습니다. API 응답만 확인한 것이 아닙니다.
자세한 증거와 한계는 [서빙 기록](serve/README.md)과
[런타임 증거](docs/evidence/2026-11-09-runtime.md)에 있습니다.

## OpenCode V2

[`serve/opencode.example.json`](serve/opencode.example.json)은 V2 형식의
공급자·모델·General 에이전트 설정입니다. Mac에서는 같은 API 키를
`VLLM_API_KEY` 환경변수로 설정하고, 기존 설정에 예시를 병합한 뒤
`/models`에서 `nvidia/Qwen3.8-Flash-Next-FP4-Local`을 선택합니다.

Spark에서 Build → General 위임을 실제 실행해 확인했습니다. General 세션은
`spark-qwen/nvidia/Qwen3.8-Flash-Next-FP4-Local`을 사용했고, 실패하던
`add()`와 할인 계산을 수정한 뒤 테스트를 통과했습니다. 배포용 예시 파일
그대로 `opencode run --agent general`을 실행해 `mathbox.py`를 수정하고
테스트 2개를 통과시키는 것도 확인했습니다. **Mac 기기에서의 직접 접속과
실행은 아직 검증하지 않았습니다.** 절차는
[Mac OpenCode 연결](docs/opencode-mac.md)에 정리했습니다.

## 안전과 한계

- 합의된 가드는 `MemAvailable` 18 GiB, memory PSI full 10%, OOM 발생 시
  즉시 중단입니다. 완화하지 않았습니다.
- 서비스는 실패 시 60초 뒤 재시작하며, 15분 안에 5회를 넘으면 멈춥니다.
  반복 점검 후에는 `systemctl --user reset-failed qwen38-fp4.service`가
  필요할 수 있습니다.
- API 키가 없으면 `/v1/models`는 401을 반환합니다. TLS는 아직 없습니다.
  `0.0.0.0` 바인딩이라 Ethernet/WiFi/Tailscale 인터페이스 모두에서 접근할
  수 있으므로, 키를 승인된 장치에만 두고 신뢰하는 네트워크에서만
  사용합니다.
- 애플리케이션/CUDA OOM 전체를 커널 `oom_kill`만으로 감지하지는 못합니다.
  로그·health·가드의 간접 감지이며, 모든 OOM 경로를 보장하지 않습니다.
- 256K(262,144) 최대 문맥 전체를 한 요청으로 시험하지는 않았습니다.
  검증한 긴 문맥은 45,946토큰입니다. 262K/8시퀀스 전체도 미검증입니다.
- 추론 강도는 `xhigh`(기본), `medium`, `low`만 지원하며 `high`는 400으로
  거부됩니다. 일반 OpenCode 설정은 thinking을 끄므로 추론 토큰을 쓰지 않습니다.
- `max-num-batched-tokens=4096`은 MTP 4와 함께 vLLM의 “suboptimal” 경고가
  있으며, 공식 기준값을 유지한 상태입니다. 상향 튜닝은 아직 검증하지
  않았습니다.
- 재부팅 자동 시작은 `enable`과 linger로 설정했지만 실제 재부팅을 수행해
  확인하지는 않았습니다. 장시간(일 단위) 연속 운용도 아직 미검증입니다.

## 모델 파일과 라이선스

로컬 캐시는 `/home/ksi/models/hf-hub/`에 있으며 컨테이너에는 읽기 전용으로
연결합니다. 스냅샷 파일은 캐시 내부 `blobs/`를 가리키므로 스냅샷 폴더만
연결하면 안 됩니다. 모델 이용 조건은 NVIDIA 모델 카드와 Qwen 라이선스를,
컨테이너는 NVIDIA Deep Learning Container License를 확인해야 합니다.

## 프로젝트 구성

```text
├── README.md                  현재 운영 상태와 측정값
├── AGENTS.md                  프로젝트 작업 지침
├── serve/                     실행·감시·상태 확인 코드
│   ├── systemd/               상시 서비스 설치와 정리 훅
│   └── opencode.example.json  OpenCode V2 연동 예시
├── tests/                     모의 시험
├── docs/evidence/             런타임 증거 요약
├── docs/opencode-mac.md       Mac OpenCode V2 연결 절차
├── docs/previous-work.md      이전 연구 정리 상태
├── LICENSE
└── .opencode/                 프로젝트 로컬 OpenCode 의존성/시험 기록
```

이전 자체 양자화·NVFP4 연구는 현재 운영 경로가 아니며 기록만 남겼습니다.
현재 변경사항은 `main`에 커밋·푸시되어 있습니다. 이전 `master`는 역사
기록으로 남기고, 원격 기본 브랜치도 `main`으로 전환합니다.
