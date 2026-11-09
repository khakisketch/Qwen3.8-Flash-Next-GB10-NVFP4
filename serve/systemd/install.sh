#!/bin/sh
# Install the per-user service and create a local API key outside the repo.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
config_home=${XDG_CONFIG_HOME:-"$HOME/.config"}
config_dir="$config_home/qwen38"
unit_dir="$config_home/systemd/user"

mkdir -p "$config_dir" "$unit_dir"
chmod 700 "$config_dir"

if [ ! -f "$config_dir/qwen38.env" ]; then
    umask 077
    key=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    printf 'VLLM_API_KEY=%s\n' "$key" > "$config_dir/qwen38.env"
fi
chmod 600 "$config_dir/qwen38.env"

sed "s|@PROJECT_DIR@|$project_dir|g" \
    "$project_dir/serve/systemd/qwen38-fp4.service.in" \
    > "$unit_dir/qwen38-fp4.service"
chmod 644 "$unit_dir/qwen38-fp4.service"

systemctl --user daemon-reload
systemctl --user enable qwen38-fp4.service
loginctl enable-linger "$USER" 2>/dev/null || \
    echo "warning: could not enable linger; boot start is not guaranteed" >&2

echo "installed qwen38-fp4.service"
echo "API key file: $config_dir/qwen38.env (mode 600; never commit it)"
echo "start: systemctl --user start qwen38-fp4.service"
