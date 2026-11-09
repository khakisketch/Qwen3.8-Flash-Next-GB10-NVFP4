#!/bin/sh
# Read-only operational check. Never prints the API key.
set -u

unit=qwen38-fp4.service
envfile=${XDG_CONFIG_HOME:-"$HOME/.config"}/qwen38/qwen38.env
base=${QWEN38_BASE_URL:-http://127.0.0.1:8000}

[ -r "$envfile" ] && . "$envfile"

echo "unit: $(systemctl --user is-active "$unit" 2>/dev/null || true)"
echo "health: $(curl -sS -o /dev/null -w '%{http_code}' "$base/health" 2>/dev/null || true)"
if [ -n "${VLLM_API_KEY:-}" ]; then
    models=$(curl -sS -H "Authorization: Bearer $VLLM_API_KEY" "$base/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys; print(", ".join(m.get("id","") for m in json.load(sys.stdin).get("data", [])))' 2>/dev/null || true)
    echo "models: ${models:-unavailable}"
else
    echo "models: VLLM_API_KEY missing; run serve/systemd/install.sh"
fi
systemctl --user status "$unit" --no-pager -n 4 2>/dev/null | sed -n '1,7p'
