# 서빙 운영: NVIDIA Qwen3.8-Flash-Next-FP4-Local

단일 DGX Spark에서 `nvidia/Qwen3.8-Flash-Next-FP4-Local`을 상시 서비스로
운영한다. 현재 운영 경로는 `systemd --user`의 `qwen38-fp4.service`이며,
`serve.diagnostic`의 감시 실행기를 사용한다. 일반 `serve.launch launch`
경로는 계속 비활성화되어 있다.

## 현재 운영 설정

| 항목 | 값 |
|---|---|
| 모델 | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| 스냅샷 | `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| 이미지 | `eugr/spark-vllm-b12x@sha256:57c24fc…b7bc329d5cb` |
| 컨텍스트 / 시퀀스 | 262,144 / 2 |
| KV 캐시 | FP8, 명시적 10 GiB = 531,950 tokens |
| 최대 동시성 | KV 기준 2.03x, `max-num-seqs=2` |
| MTP / PLE / 배치 | 4 / 디스크 `io_uring` / 4,096 batched tokens |
| API | `0.0.0.0:8000`, Bearer 키 필수 |
| 가드 | `MemAvailable` 18 GiB, PSI full 10%, OOM/감시 실패 중단 |

NVIDIA 카드의 Spark 예제는 262,144/8시퀀스다. 이 장비에서는 256K/2시퀀스가
가드·기동·문맥 시험을 통과해 운영 프로필로 선택했으며, 8시퀀스 전체는
검증하지 않았다. `--gpu-memory-utilization 0.8`은 유지하지만
`--kv-cache-memory-bytes`를 명시했으므로 KV 용량은 10 GiB 예산을 사용한다.

## 용량과 제약 (2026-11-09)

- `max_model_len=262,144`: 요청당 **입력+출력 합계** 한도다. OpenCode의
  출력 상한 8,192를 최대로 쓰려면 입력은 약 253,952토큰 이하로 두는 것이
  안전하다.
- KV 10 GiB = **531,950 tokens**. vLLM은 262,144토큰 요청 기준 최대
  **2.03배** 동시성을 계산했고 `max-num-seqs=2`와 일치한다. 2개의 최대
  길이 요청(524,288토큰)은 KV 여유가 약 7,662토큰뿐이라 매우 타이트하다.
  일반적인 짧은 요청 2개는 여유가 크다.
- `max-num-batched-tokens=4,096`. MTP 4와 함께
  `max_num_scheduled_tokens=4096` “suboptimal” 경고가 있다. 공식 기준값을
  유지했고, 상향 튜닝은 별도 검증이 필요하다.
- `gpu-memory-utilization=0.8`은 KV 산정에 사용되지 않는다. vLLM 로그도
  `kv_cache_memory_bytes` 때문에 memory profiling을 생략했다고 명시한다.
  실제 안전 경계는 호스트 18 GiB 가드다.
- 모델은 125B total / 6B active, 51B n-gram 테이블과 4B MTP를 포함한다.
  체크포인트는 약 96 GiB·21 shard이며 PLE 테이블은 디스크에 있다.
- GB10은 통합 메모리라 `nvidia-smi`의 memory total/used가 `N/A`로 나온다.
  호스트 `MemAvailable`이 판단 기준이다.

API 키는 `~/.config/qwen38/qwen38.env`에만 있다. Git, 문서, 로그에 넣지
않는다. OpenCode V2 예시는 [opencode.example.json](opencode.example.json)에
있으며 `{env:VLLM_API_KEY}`를 사용한다.

## 운영 명령

```sh
./serve/systemd/install.sh
systemctl --user start qwen38-fp4.service
systemctl --user restart qwen38-fp4.service
systemctl --user stop qwen38-fp4.service
./serve/status.sh
journalctl --user -u qwen38-fp4.service -f
```

`install.sh`는 서비스 파일을 설치하고, 키가 없으면 무작위로 만들어
`~/.config/qwen38/qwen38.env`(0600)에 저장한다. `enable`과 linger를
설정하므로 재부팅 후 자동 시작을 의도하지만, 실제 재부팅 시험은 아직
하지 않았다.

### 장애 복구

`ExecStartPre`와 `ExecStopPost`의
[`cleanup-stale.sh`](systemd/cleanup-stale.sh)는 이름과 `project` 라벨이
정확히 일치하는 컨테이너만 제거한다. 호스트 감독자가 SIGKILL로 죽어도
컨테이너 내부 watchdog이 heartbeat 상실을 감지해 서버를 종료하고,
systemd가 60초 뒤 새로 기동한다. 15분 안에 5회를 넘으면 start limit에
걸리므로 점검 후에는 다음이 필요할 수 있다.

```sh
systemctl --user reset-failed qwen38-fp4.service
```

## 검증 결과 (2026-11-09)

실제 모델을 로딩하고 추론한 결과다. 단일 장비·소규모 표본이다.

| 항목 | 결과 |
|---|---|
| 256K/2seq 기동 | 285초 |
| 64K/2seq 첫 기동 | 870초 (역사 기록, 새 그래프/커널 포함) |
| 64K/2seq 재기동 | 272초 (역사 기록) |
| 호스트 감독자 SIGKILL 후 복구 | 336초 (64K/2 시험) |
| 256K/2seq 기동 중 최저 | 23.61 GiB |
| 256K/2seq 준비 후 | 25.98–26.5 GiB |
| TTFT / 디코드 | 0.294초 / 46.27 tok/s 중앙값 (256K/2, 3회) |
| 긴 문맥 | 45,946토큰 needle 정답 (18.88초) |
| 동시 요청 | 2개 완료 (256K/2) |
| 반복 요청 | 20/20 성공, 0.37초 중앙값 (256K/2) |
| API 키 | 없는 요청 401, 있는 요청 200 |
| OpenCode General | 25.2 / 20.7 / 19.0초, 실제 버그 3건 수정 후 테스트 2/4/2개 통과 |

원시 측정 파일은 `.opencode/verification/`, 실행 증거는
`.opencode/diagnostics/trial-*`에 있다. Git에는 큰 로그를 넣지 않고
[docs/evidence/2026-11-09-runtime.md](../docs/evidence/2026-11-09-runtime.md)에
요약한다.

## 가드와 한계

- 기동 전부터 호스트와 컨테이너 PID-1이 각각 18 GiB/PSI/OOM을 감시한다.
  임계값을 넘기면 컨테이너만 종료하고 재시작하지 않는다.
- `oom_kill`은 커널 OOM 킬만 센다. `torch.cuda.OutOfMemoryError` 같은
  애플리케이션/CUDA OOM은 별도 로그·health·메모리 변화로 간접 감지하며,
  모든 경로를 보장하지 않는다.
- 샘플링은 예방이 아니다. `avg10`은 약 10초 평활이고 1초 polling은
  sub-second 변화를 놓칠 수 있다.
- API는 `0.0.0.0`에 열려 있어 Ethernet/WiFi/Tailscale 인터페이스 모두에서
  접근할 수 있고 TLS가 없다. 키를 가진 승인 장치만 사용하는 전제이며,
  인터넷에 직접 노출하지 않는다.
- 256K 최대 문맥 전체를 한 요청으로 시험하지 않았다. 검증한 최대는
  45,946토큰이다. 장시간(일 단위) 연속 운용도 아직 미검증이다.
- Mac의 OpenCode에서 직접 접속·General 실행은 아직 확인하지 않았다.
  Spark에서 같은 V2 설정으로 Build → General 로컬 실행은 확인했다.

## OpenCode V2

`serve/opencode.example.json`은 V2의 `providers`/`package` 형식이다.
`general` 에이전트의 모델을 `spark-qwen/nvidia/Qwen3.8-Flash-Next-FP4-Local`로
지정하고, vLLM이 discovery에서 도구 지원을 알리지 않으므로
`capabilities.tools: true`를 명시한다. `body.chat_template_kwargs`로
thinking을 끄면 짧은 코딩 작업의 지연과 토큰 사용이 줄어든다. 이 예시
파일 자체를 임시 프로젝트에 복사해 `opencode models`에서 공급자가
표시되는 것과, `opencode run --agent general`로 `mathbox.py` 버그를
수정하고 테스트 2개를 통과시키는 것도 확인했다. Mac 절차는
[opencode-mac.md](../docs/opencode-mac.md)에 있다.

Mac에서는 키를 환경변수로 설정한 뒤 예시를 기존 설정에 병합한다.

```sh
export VLLM_API_KEY='...'   # Spark의 ~/.config/qwen38/qwen38.env 값
opencode models | grep spark-qwen
```

## 과거 장애와 수정

- 2026-10-30/11-02 시험은 tmux에서 실행했다. tmux/호스트 heartbeat가
  사라지자 컨테이너 watchdog이 `host supervisor heartbeat missing/stale`로
  서버를 종료했고, 이것이 `container log stream ended unexpectedly`로
  보고되었다. 모델이나 메모리 오류가 아니었다. 상시 systemd 서비스로
  바꾸어 이 경로를 제거했다.
- 32K 이전 시험에서 CSF 가중치 후처리의 GPU 예약 캐시가 급증해 18 GiB
  가드를 침범했다. `--reclaim-csf-cache`로 사용하지 않는 CUDA 캐시를
  반환하고, KV를 명시적 4 GiB로 제한해 통과했다.
- 초기 64K/2seq는 KV 9 GiB로 상향했고, 18 GiB 가드에서 기동 최저
  24.89 GiB, 준비 후 약 25.9–27.6 GiB를 유지했다.
- 2026-11-09에 256K/2seq·KV 10 GiB로 상향했다. 기동 285초, 기동 최저
  23.61 GiB, 준비 후 25.98–26.5 GiB, 45,946토큰 문맥과 2시퀀스 동시
  요청을 통과했다.

## 시험

```sh
python3 -m unittest discover -s tests -v
```

현재 171개 시험이 통과한다. 실제 모델을 시작하지 않고 가드·argv·감시·
복구 경로를 검증하며, 실제 기동·복구는 위 측정 결과와 systemd 시험으로
확인했다.
