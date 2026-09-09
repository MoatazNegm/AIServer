#!/usr/bin/env bash
# One-shot CLI: send a prompt to the vLLM gateway and print the reply.
# Pure Python 3 stdlib on the host (urllib + json). No jq, no curl needed.
#
# Authentication:
#   Pass the API key via --api-key <key> or set the GATEWAY_API_KEY env var.
#   The key is sent in `Authorization: Bearer <key>` to the gateway.
#
# Usage:
#   ./prompt.sh "What is the capital of France?"
#   GATEWAY_API_KEY=abc... ./prompt.sh "hello"
#   ./prompt.sh --api-key abc... "hello"
#   ./prompt.sh -s "You are a poet." "Compose a sonnet about containers."
#   ./prompt.sh -m Qwen/Qwen2.5-3B-Instruct "Tell me a joke"
#   ./prompt.sh -t 0.7 -n 200 "Be creative"
#   echo "Sum 5+7" | ./prompt.sh             # prompt from stdin
#
# Env:
#   GATEWAY_URL     (default http://localhost:9000)
#   GATEWAY_API_KEY (default unset — required when gateway has auth enabled)
#   MODEL           default model (overridable with -m)

set -euo pipefail

GATEWAY="${GATEWAY_URL:-http://localhost:9000}"
API_KEY="${GATEWAY_API_KEY:-}"
MODEL="${MODEL:-}"
SYSTEM=""
TEMPERATURE=""
MAX_TOKENS=""

usage() {
    sed -n '2,16p' "$0"
    exit "${1:-0}"
}

# Pre-parse for long options. getopts only handles short flags; rewrite
# --api-key=VALUE to -k VALUE before getopts runs.
new_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key=*) new_args+=(-k "${1#--api-key=}"); shift ;;
        --api-key)    new_args+=(-k "$2"); shift 2 ;;
        *)            new_args+=("$1"); shift ;;
    esac
done
set -- "${new_args[@]}"

while getopts ":hm:s:t:n:k:" opt; do
    case "$opt" in
        h) usage 0 ;;
        m) MODEL="$OPTARG" ;;
        s) SYSTEM="$OPTARG" ;;
        t) TEMPERATURE="$OPTARG" ;;
        n) MAX_TOKENS="$OPTARG" ;;
        k) API_KEY="$OPTARG" ;;
        :) echo "missing arg for -$OPTARG" >&2; usage 2 ;;
        \?) echo "unknown flag -$OPTARG" >&2; usage 2 ;;
    esac
done
shift $((OPTIND-1))

# Prompt: positional or stdin
if [[ $# -ge 1 ]]; then
    PROMPT="$*"
else
    PROMPT="$(cat)"
fi
if [[ -z "${PROMPT// }" ]]; then
    usage 2
fi

exec python3 - "$GATEWAY" "$PROMPT" "$MODEL" "$SYSTEM" "$TEMPERATURE" "$MAX_TOKENS" "$API_KEY" <<'PY'
import json
import sys
import urllib.error
import urllib.request

(gateway, prompt, model, system, temperature, max_tokens, api_key) = sys.argv[1:8]

body = {"prompt": prompt}
if model:       body["model"]       = model
if system:      body["system"]      = system
if temperature: body["temperature"] = float(temperature)
if max_tokens:  body["max_tokens"]  = int(max_tokens)

headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

req = urllib.request.Request(
    f"{gateway}/v1/prompt",
    data=json.dumps(body).encode(),
    headers=headers,
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=180) as r:
        ctype = r.headers.get("Content-Type", "")
        body_bytes = r.read()
except urllib.error.HTTPError as e:
    # Pass through the gateway's structured error verbatim.
    sys.stderr.write(f"HTTP {e.code} from {gateway}/v1/prompt:\n")
    sys.stderr.write(e.read().decode(errors="replace") + "\n")
    sys.exit(1)
except urllib.error.URLError as e:
    sys.stderr.write(f"could not reach {gateway}/v1/prompt: {e.reason}\n")
    sys.exit(1)

if "application/json" not in ctype:
    sys.stdout.write(body_bytes.decode(errors="replace"))
    sys.exit(0)

try:
    data = json.loads(body_bytes)
except json.JSONDecodeError:
    sys.stdout.write(body_bytes.decode(errors="replace"))
    sys.exit(0)

if "reply" in data:
    sys.stdout.write(data["reply"] + "\n")
else:
    sys.stdout.write(json.dumps(data, indent=2) + "\n")
    sys.exit(2)
PY
