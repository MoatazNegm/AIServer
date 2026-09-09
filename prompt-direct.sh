#!/usr/bin/env bash
# One-shot CLI for talking to vLLM directly from the host (bypasses the gateway).
#
# Use this when:
#   - You're running an automation/cronjob on THIS host that needs vLLM
#   - You don't want to deal with API keys
#   - You want the standard OpenAI /v1/chat/completions shape (full message
#     history, multi-turn, etc.)
#
# Use ./prompt.sh (with --api-key) instead when:
#   - Calling from outside the host (e.g. from a laptop, phone, CI on another box)
#   - You need the auth boundary
#
# The gateway enforces auth; this script skips the gateway and talks to the
# vLLM OpenAI-compatible endpoint directly. vLLM is reachable on host port
# 8000 because start-vllm.sh uses `docker run -p 8000:8000`.
#
# Usage:
#   ./prompt-direct.sh "What is the capital of France?"
#   ./prompt-direct.sh -m Qwen/Qwen2.5-7B-Instruct "Tell me a joke"
#   ./prompt-direct.sh -s "You are a poet." "Compose a sonnet about containers."
#   ./prompt-direct.sh -t 0.7 -n 200 "Be creative"
#   ./prompt-direct.sh -H 5 -S "You are concise." "Top 3 reasons …"
#   echo "Sum 5+7" | ./prompt-direct.sh
#
# Multi-turn example (full OpenAI chat-completions shape):
#   ./prompt-direct.sh -H 2 -S "You answer in haiku only." \
#     -U "What is 2+2?" "And what is 3+3?" "What is 5+5?"
#
# Flags:
#   -m MODEL        model name (default: $VLLM_MODEL or Qwen/Qwen2.5-7B-Instruct)
#   -s SYSTEM       system prompt (prepended as a system-role turn)
#   -t TEMP         sampling temperature (default: server's)
#   -n MAXTOKENS    max new tokens (default: server's, usually 256)
#   -H N_HISTORY    number of prior user prompts to keep in the conversation
#                   (default: 1 — only the current prompt)
#                   Higher values let the model see prior turns.
#   -U USER...      one or more user prompts (each becomes a turn).
#                   With -H 2 and three -U args, the second/third prompts
#                   see the earlier ones as assistant turns.
#
# Env:
#   VLLM_URL       (default http://localhost:8000)
#   VLLM_MODEL     default model name

set -euo pipefail

VLLM_URL="${VLLM_URL:-http://localhost:8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SYSTEM=""
TEMPERATURE=""
MAX_TOKENS=""
HISTORY=1
USERS=()

usage() {
    sed -n '2,40p' "$0"
    exit "${1:-0}"
}

while getopts ":hm:s:t:n:H:U:" opt; do
    case "$opt" in
        h) usage 0 ;;
        m) MODEL="$OPTARG" ;;
        s) SYSTEM="$OPTARG" ;;
        t) TEMPERATURE="$OPTARG" ;;
        n) MAX_TOKENS="$OPTARG" ;;
        H) HISTORY="$OPTARG" ;;
        U) USERS+=("$OPTARG") ;;
        :) echo "missing arg for -$OPTARG" >&2; usage 2 ;;
        \?) echo "unknown flag -$OPTARG" >&2; usage 2 ;;
    esac
done
shift $((OPTIND-1))

# If no -U prompts and there are positional args, treat them as one user turn.
if [[ ${#USERS[@]} -eq 0 ]]; then
    if [[ $# -ge 1 ]]; then
        USERS=("$*")
    else
        # Fall back to stdin.
        USERS=("$(cat)")
    fi
fi
if [[ ${#USERS[@]} -eq 0 ]] || [[ -z "${USERS[0]// }" ]]; then
    usage 2
fi

# HISTORY controls how many prior -U prompts are echoed back to the model
# in alternating user/assistant turns. The most recent prompt is always
# included; earlier ones are included only if HISTORY >= N.
n_users=${#USERS[@]}
if (( HISTORY < 1 )); then HISTORY=1; fi

exec python3 - "$VLLM_URL" "$MODEL" "$SYSTEM" "$TEMPERATURE" "$MAX_TOKENS" "$HISTORY" "${USERS[@]}" <<'PY'
import json
import sys
import urllib.error
import urllib.request

args = sys.argv[1:]
url, model, system, temperature, max_tokens, history = args[:6]
users = args[6:]

messages = []
if system:
    messages.append({"role": "system", "content": system})

# All user prompts go in
for u in users:
    messages.append({"role": "user", "content": u})

# Echo the last (history - 1) prior user prompts back as assistant turns
# so the model can see the conversation. NOTE: the assistant echoes here
# are placeholders — they're what we sent BEFORE; vLLM treats them as
# conversation context, not as fresh model output.
start = max(0, len(users) - int(history))
for prev in users[start:-1]:
    messages.append({"role": "assistant", "content": prev})

body = {"model": model, "messages": messages}
if temperature:
    body["temperature"] = float(temperature)
if max_tokens:
    body["max_tokens"] = int(max_tokens)

req = urllib.request.Request(
    f"{url}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=180) as r:
        ctype = r.headers.get("Content-Type", "")
        body_bytes = r.read()
except urllib.error.HTTPError as e:
    sys.stderr.write(f"HTTP {e.code} from {url}/v1/chat/completions:\n")
    sys.stderr.write(e.read().decode(errors="replace") + "\n")
    sys.exit(1)
except urllib.error.URLError as e:
    sys.stderr.write(f"could not reach {url}/v1/chat/completions: {e.reason}\n")
    sys.exit(1)

try:
    data = json.loads(body_bytes)
except json.JSONDecodeError:
    sys.stdout.write(body_bytes.decode(errors="replace"))
    sys.exit(0)

# OpenAI-shape response: choices[0].message.content
try:
    reply = data["choices"][0]["message"]["content"]
except (KeyError, IndexError, TypeError):
    sys.stdout.write(json.dumps(data, indent=2) + "\n")
    sys.exit(2)

sys.stdout.write(reply + "\n")
PY
