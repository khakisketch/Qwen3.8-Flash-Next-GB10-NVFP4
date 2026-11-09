#!/bin/sh
# Remove only the container carrying this project's exact name and label.
# Safe to run before start and after stop; never touches a foreign container.
set -eu

name=qwen38-fp4-local-serve
label=qwen38-fp4-local
docker=/usr/bin/docker

id=$("$docker" inspect --format '{{.Id}}' "$name" 2>/dev/null || true)
[ -n "$id" ] || exit 0

actual_name=$("$docker" inspect --format '{{.Name}}' "$id" 2>/dev/null || true)
actual_label=$("$docker" inspect --format '{{index .Config.Labels "project"}}' "$id" 2>/dev/null || true)
if [ "$actual_name" != "/$name" ] || [ "$actual_label" != "$label" ]; then
    echo "refusing to touch a container with unexpected identity" >&2
    exit 1
fi

"$docker" rm -f "$id" >/dev/null 2>&1 || true
