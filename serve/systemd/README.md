# systemd 운영

사용자 서비스 하나로 모델 서버를 상시 실행한다. 모델·이미지·캐시 경로는
프로젝트의 `serve/baseline.py`와 `serve/diagnostic.py`가 결정하며, 서비스
파일에는 비밀키가 없다.

```sh
./serve/systemd/install.sh
systemctl --user start qwen38-fp4.service
systemctl --user enable qwen38-fp4.service
```

API 키는 `~/.config/qwen38/qwen38.env`(0600)에 있다. Git에 넣지 않는다.
Mac에는 같은 값을 환경변수 `VLLM_API_KEY`로 설정한다.

```sh
systemctl --user status qwen38-fp4.service
systemctl --user restart qwen38-fp4.service
systemctl --user stop qwen38-fp4.service
journalctl --user -u qwen38-fp4.service -f
```

서비스는 실패 시 60초 뒤 재시작하며, 15분 안에 5회를 넘으면 멈춘다.
18 GiB/PSI 10% 가드 위반, heartbeat 상실, 모델 오류는 재시작 전에
preflight를 다시 통과해야 한다. 정상 정지(SIGINT, exit 130)는 재시작하지
않는다. 반복 점검 후 start limit에 걸리면
`systemctl --user reset-failed qwen38-fp4.service`를 실행한다.

## 재부팅 후 확인

실제 재부팅 시험은 아직 하지 않았다. 재부팅 후에는 다음을 확인한다.

```sh
systemctl --user is-enabled qwen38-fp4.service
loginctl show-user "$USER" -p Linger
./serve/status.sh
journalctl --user -u qwen38-fp4.service -n 50 --no-pager
```

`Linger=yes`와 `enabled`가 아니면 사용자 로그인 전에는 서비스가 시작되지
않는다. `install.sh`가 둘을 설정하지만, 실제 재부팅 뒤 상태는 별도로
확인해야 한다.
