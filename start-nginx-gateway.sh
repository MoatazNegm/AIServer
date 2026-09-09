#!/usr/bin/env bash
# Starts the nginx reverse-proxy that exposes vllm-gateway on host :80
# of the secondary IP 192.168.8.10 (so external requestors can hit
# http://192.168.8.10/ instead of needing the :9000 port).
#
# Why --network host:
#   The gateway binds to a specific host IP (192.168.8.10), which only
#   exists on the host's network namespace. With --network host, nginx
#   shares the host's network stack and can bind to that IP directly.
#   Without --network host, nginx would be in a bridge network and
#   couldn't see 192.168.8.10.
#
# Stops:
#   docker stop nginx-gateway

set -euo pipefail

NAME="nginx-gateway"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CONF_FILE="$SCRIPT_DIR/nginx-gateway.conf"

[[ -f "$CONF_FILE" ]] || { echo "missing $CONF_FILE"; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --network host \
    -v "${CONF_FILE}:/etc/nginx/nginx.conf:ro" \
    nginx:latest

echo
echo "Nginx reverse-proxy '$NAME' started (--network host, listens on host IP)."
echo "  config     $CONF_FILE"
echo "  external   http://192.168.8.10/healthz"
echo "  external   POST http://192.168.8.10/v1/prompt      (auth still enforced by gateway)"
echo "  logs       docker logs -f $NAME"
echo "  stop       docker stop $NAME"
