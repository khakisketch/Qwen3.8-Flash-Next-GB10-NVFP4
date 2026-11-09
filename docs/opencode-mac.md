# Mac OpenCode V2 연결 절차

이 문서는 Mac에서 DGX Spark의 로컬 Qwen을 OpenCode V2의 General
에이전트로 사용하는 최소 절차다. **Mac 실기기 검증은 아직 하지 않았으며**,
Spark에서 같은 V2 설정으로 직접 General 작업을 실행해 검증했다.

## 1. 키 준비

Spark에서 키 파일을 확인한다. 값을 채팅·Git·스크린샷에 남기지 않는다.

```sh
# Spark
cat ~/.config/qwen38/qwen38.env
```

Mac의 셸 프로필이나 비밀 관리 도구에 같은 값을 설정한다.

```sh
export VLLM_API_KEY='Spark에서_복사한_값'
```

OpenCode를 GUI로 실행한다면 환경변수가 상속되도록 셸에서 실행하거나,
OS 사용자 환경변수로 등록한다. macOS GUI 앱은 터미널의 `export`를
상속하지 않는 경우가 많으므로, GUI로 띄울 때는 다음처럼 등록한 뒤
OpenCode를 완전히 종료하고 다시 실행한다.

```sh
launchctl setenv VLLM_API_KEY 'Spark에서_복사한_값'
```

기존 설정에 `apiKey: "local"` 같은 자리표시자가 남아 있으면 실제 키로
바꾸거나 `"{env:VLLM_API_KEY}"`를 사용한다. `local`은 인증키가 아니다.

## 2. 설정 병합

OpenCode V2는 `providers`(복수)와 `package`를 사용한다. 예전
`provider`/`npm` 형식이 아니라
[`serve/opencode.example.json`](../serve/opencode.example.json)을 기준으로
기존 `~/.config/opencode/opencode.jsonc` 또는 프로젝트
`.opencode/opencode.jsonc`에 병합한다.

핵심 값:

```jsonc
{
  "providers": {
    "spark-qwen": {
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": {
        "baseURL": "http://172.30.1.44:8000/v1",
        "apiKey": "{env:VLLM_API_KEY}"
      },
      "models": {
        "nvidia/Qwen3.8-Flash-Next-FP4-Local": {
          "limit": { "context": 262144, "output": 8192 },
          "capabilities": {
            "tools": true,
            "input": ["text"],
            "output": ["text"],
            "reasoning": {
              "supported": true,
              "effort": ["low", "medium", "xhigh"]
            }
          },
          "body": { "chat_template_kwargs": { "enable_thinking": false } }
        }
      }
    }
  },
  "agents": {
    "general": {
      "mode": "subagent",
      "model": "spark-qwen/nvidia/Qwen3.8-Flash-Next-FP4-Local",
      "permissions": [
        { "action": "edit", "resource": "*", "effect": "allow" },
        { "action": "shell", "resource": "*", "effect": "allow" }
      ]
    }
  }
}
```

기존 Build, Plan, Explore 설정은 건드리지 않는다. `general` 항목만
추가·병합한다.

## 3. 연결 확인

```sh
opencode models | grep spark-qwen
# spark-qwen/nvidia/Qwen3.8-Flash-Next-FP4-Local
```

`/models`에서 같은 모델을 선택할 수 있어야 한다. 인증 오류가 나면
`VLLM_API_KEY`가 OpenCode 프로세스에 전달됐는지 확인한다.

## 4. General 실제 작업

가장 직접적인 확인은 General을 지정해 작은 버그를 고치게 하는 것이다.
Spark에서 검증한 명령과 같은 형태다.

```sh
opencode run --agent general --auto \
  "Read mathbox.py and test_mathbox.py. Fix only mathbox.py so the tests pass, run 'python3 -m unittest -v', and report the diff and result."
```

정상 출력의 첫 줄은 다음처럼 로컬 모델을 가리킨다.

```text
> general · nvidia/Qwen3.8-Flash-Next-FP4-Local
```

Build를 통해 위임하려면 다음처럼 요청한다.

```text
Use the general subagent to fix the failing test. Tell it to read the
relevant files, fix only the implementation, run the test command, and
report the diff and result. Do not edit files yourself.
```

## 5. 문제 해결

| 증상 | 확인 |
|---|---|
| `401 Unauthorized` | Mac의 `VLLM_API_KEY`와 Spark의 키가 같은지. GUI라면 `launchctl setenv` 후 OpenCode를 완전히 재시작했는지, 설정에 `apiKey: "local"` 자리표시자가 남아 있지 않은지 |
| 연결 거부 | `./serve/status.sh`, `ss -ltn '( sport = :8000 )'` |
| 모델 목록에 없음 | V2 `providers`/`package` 형식인지, baseURL이 `/v1`인지 |
| 도구 호출을 안 함 | `capabilities.tools: true`, 서버의 `qwen3_xml` 파서 |
| `Unexpected reasoning effort high` | `#high` 변형을 제거한다. 이 모델의 지원 값은 `xhigh`(기본), `medium`, `low` |
| 응답이 느림 | `body.chat_template_kwargs.enable_thinking=false` 확인 |
| 긴 문맥 오류 | 모델 한도 262,144, 출력 8,192 이내인지 |

## 6. 보안

- API는 내부망 `0.0.0.0:8000`에 열려 있고 **TLS가 없다**. 신뢰하는
  내부망에서만 사용한다.
- 키를 Git, 문서, 채팅, 스크린샷에 저장하지 않는다.
- Mac을 잃어버렸거나 키가 노출되면 Spark에서 키 파일을 교체하고 서비스를
  재시작한 뒤 Mac의 키도 바꾼다.

```sh
# Spark: 새 키 생성 예시 (값은 출력하지 않음)
umask 077
python3 -c 'import secrets; print("VLLM_API_KEY=" + secrets.token_urlsafe(32))' \
  > ~/.config/qwen38/qwen38.env
systemctl --user restart qwen38-fp4.service
```
