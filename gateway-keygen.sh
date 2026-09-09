#!/usr/bin/env bash
# Mint a strong random API key for the vLLM gateway.
#
# Usage:
#   ./gateway-keygen.sh                  # print one key
#   ./gateway-keygen.sh 5               # print 5 keys
#   ./gateway-keygen.sh --add alice     # print a key and append it to /root/gpuenable/.api-keys
#
# Keys are 256 bits of entropy (32 random bytes) printed as hex.
# Run as root or as the gateway admin.

set -euo pipefail

COUNT=1
ADD_LABEL=""
ADD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --add) ADD=1; ADD_LABEL="${2:-}"; shift 2;;
        --add=*) ADD=1; ADD_LABEL="${1#*=}"; shift;;
        -h|--help)
            sed -n '2,11p' "$0"; exit 0;;
        [0-9]*) COUNT="$1"; shift;;
        *) echo "unknown arg: $1" >&2; exit 2;;
    esac
done

KEYS_FILE="/root/gpuenable/.api-keys"

# Generate the requested number of keys
keys=()
for _ in $(seq 1 "$COUNT"); do
    k=$(openssl rand -hex 32)
    keys+=("$k")
done

# Always print to stdout
if [[ "$COUNT" -eq 1 ]]; then
    echo "${keys[0]}"
else
    printf '%s\n' "${keys[@]}"
fi

# Optionally append to .api-keys
if [[ "$ADD" -eq 1 ]]; then
    mkdir -p "$(dirname "$KEYS_FILE")"
    touch "$KEYS_FILE"
    chmod 600 "$KEYS_FILE"
    {
        # Optional label comment above the key
        if [[ -n "$ADD_LABEL" ]]; then
            echo "# added $(date -u +%Y-%m-%dT%H:%M:%SZ) for $ADD_LABEL"
        fi
        for k in "${keys[@]}"; do
            echo "$k"
        done
    } >> "$KEYS_FILE"
    echo
    echo "appended to $KEYS_FILE (now $(wc -l < "$KEYS_FILE") lines, mode 600)"
    echo "the gateway reads this file at start; restart it to pick up new keys:"
    echo "  docker restart vllm-gateway"
fi
