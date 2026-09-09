#!/usr/bin/env bash
# Starts (or restarts) an nginx container that forwards TCP 4041 -> host sshd (22)
# using nginx's stream module. Run as root (docker access required).
set -euo pipefail

NAME="ssh-proxy"
IMAGE="nginx:latest"
HOST_PORT="4041"
CONFIG_DIR="/root/gpuenable"

# Remove any existing container with the same name
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    -p "${HOST_PORT}:4041" \
    -v "${CONFIG_DIR}/nginx.conf:/etc/nginx/nginx.conf:ro" \
    "$IMAGE"

echo "Container '$NAME' started. Mapping: host:${HOST_PORT} -> container:4041 -> 172.17.0.1:22"
echo "Verify with:  docker logs $NAME   |   nc -vz 127.0.0.1 ${HOST_PORT}"