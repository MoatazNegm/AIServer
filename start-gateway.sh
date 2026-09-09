#!/usr/bin/env bash
# Starts the vLLM gateway container — a thin FastAPI layer in front of vLLM
# that exposes POST /v1/prompt. Listens on host :9000.
#
# Authentication:
#   The gateway enforces an API key on /v1/prompt. Keys are loaded from
#   /root/gpuenable/.api-keys (one key per line; # for comments) if that
#   file exists. Use ./gateway-keygen.sh to mint new keys.
#   When the file is absent OR has zero valid keys, the gateway runs in
#   open mode (no auth) — useful for the first bring-up.
#
# Usage:
#   ./start-gateway.sh                                      # default model
#   VLLM_MODEL=Qwen/Qwen2.5-3B-Instruct ./start-gateway.sh
#   GATEWAY_PORT=9100 ./start-gateway.sh
#
# Bring up in this order:
#   1. ./start-vllm.sh   Qwen/Qwen2.5-7B-Instruct     (slow first download)
#   2. ./start-gateway.sh
# Then ./prompt.sh "Hello!"
#
# Stops both:
#   docker stop vllm-server vllm-gateway

set -euo pipefail

NAME="vllm-gateway"
NETWORK="vllm-net"
PORT="${GATEWAY_PORT:-9000}"
# The IP the gateway binds to. Defaults to 192.168.8.10 (a secondary IP on
# eno1 added via NetworkManager). The nginx container reverse-proxies this
# port on 192.168.8.10:80. Override with GATEWAY_BIND_IP if needed.
BIND_IP="${GATEWAY_BIND_IP:-192.168.8.10}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
GATEWAY_DIR="$SCRIPT_DIR/gateway"
KEYS_FILE="$SCRIPT_DIR/.api-keys"

# Shared network so the gateway can resolve the upstream by name as
# `vllm-server:8000`.
docker network create "$NETWORK" >/dev/null 2>&1 || true

# Restart cleanly.
docker rm -f "$NAME" >/dev/null 2>&1 || true

# Build the docker run args. The image name goes BEFORE the container
# command — anything appended after the image is treated as command args.
# So we put the image in the middle and the command at the end.
run_args=(
    -d --name "$NAME"
    --restart unless-stopped
    --network "$NETWORK"
    -p "${BIND_IP}:${PORT}:9000"
    -v "${GATEWAY_DIR}:/app:ro"
    -e VLLM_URL="${VLLM_URL:-http://vllm-server:8000}"
    -e VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    -e VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-256}"
)

# If a keys file exists, bind-mount it read-only into the container and
# tell the gateway to load it. Otherwise the gateway runs in open mode.
auth_state="open (no API key required)"
if [[ -f "$KEYS_FILE" ]] && [[ -s "$KEYS_FILE" ]]; then
    n=$(grep -cE '^[A-Za-z0-9._-]{16,}$' "$KEYS_FILE" 2>/dev/null || echo 0)
    if [[ "$n" -gt 0 ]]; then
        run_args+=( -v "${KEYS_FILE}:/tmp/gateway-keys:ro" -e API_KEYS_FILE=/tmp/gateway-keys )
        auth_state="enforced ($n keys from $KEYS_FILE)"
    fi
fi

# Image + container command (everything after this is passed to the entrypoint).
run_args+=( --entrypoint python3 vllm-a4500:latest
            -m uvicorn app:app --app-dir /app --host 0.0.0.0 --port 9000 )

docker run "${run_args[@]}"

echo
echo "Gateway '$NAME' started."
echo "  auth     $auth_state"
echo
echo "  bind     ${BIND_IP}:${PORT}"
echo "  healthz  http://${BIND_IP}:${PORT}/healthz"
echo "  readyz   http://${BIND_IP}:${PORT}/readyz     (503 until vLLM is up)"
echo "  prompt   POST http://${BIND_IP}:${PORT}/v1/prompt   (requires API key)"
echo "           (also reachable via nginx on http://${BIND_IP}:80)"
echo "           body:     {\"prompt\":\"...\", \"model\":\"...\", \"system\":\"...\","
echo "                     \"temperature\":0.7, \"max_tokens\":256}"
echo
echo "  prompt.sh  ./prompt.sh 'What is the capital of France?'"
echo "              ./prompt.sh --api-key <key> '...'   # or set \$GATEWAY_API_KEY"
echo "  logs       docker logs -f $NAME"
echo "  keygen     ./gateway-keygen.sh --add <label>     # mint a new key"
echo "  stop       docker stop $NAME"
