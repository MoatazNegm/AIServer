#!/usr/bin/env bash
# Starts the vllm-a4500 container running the OpenAI-compatible vLLM server.
# Listens on host :8000.
#
# Requirements on the host:
#   - The vllm-a4500 image built (see vllm.Dockerfile).
#   - Outbound HTTPS to huggingface.co for first-time model downloads.
#
# Usage:
#   ./start-vllm.sh                              # INTERACTIVE: list installed,
#                                                 # pick a number, or type a new id
#   ./start-vllm.sh Qwen/Qwen2.5-7B-Instruct     # direct
#   ./start-vllm.sh mistralai/Mistral-7B-Instruct-v0.3
#
# Stop:  docker stop vllm-server

set -euo pipefail

NAME="vllm-server"
NETWORK="vllm-net"
PORT="${VLLM_PORT:-8000}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PROMPT_SCRIPT="$SCRIPT_DIR/prompt.sh"

mkdir -p "$HF_CACHE"

# ---------------------------------------------------------------------------
# Resolve MODEL: either from $1, an interactive picker, or the default.
# ---------------------------------------------------------------------------
DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL="${1:-}"

if [[ -z "$MODEL" ]] && [[ -t 0 ]]; then
    # Discover already-downloaded model cache dirs.
    # HF stores them as: $HF_CACHE/hub/models--<Org>--<Name>
    # The Org/Name pattern allows letters, digits, _, -, .
    installed_names=()
    installed_sizes=()
    installed_total_bytes=0
    if [[ -d "$HF_CACHE/hub" ]]; then
        while IFS= read -r dir; do
            [[ "$dir" =~ ^models-- ]] || continue
            # Strip 'models--' prefix and convert the remaining '--' to '/'
            name="${dir#models--}"
            name="${name//--//}"
            # Some cache dirs may be partial / mid-download — skip if no snapshots
            [[ -d "$HF_CACHE/hub/$dir/snapshots" ]] || continue
            bytes=$(du -sb "$HF_CACHE/hub/$dir" 2>/dev/null | awk '{print $1}')
            bytes=${bytes:-0}
            installed_names+=("$name")
            installed_sizes+=("$bytes")
            installed_total_bytes=$((installed_total_bytes + bytes))
        done < <(ls -1 "$HF_CACHE/hub" 2>/dev/null | sort || true)
    fi

    echo "Models already in $HF_CACHE :"
    if [[ ${#installed_names[@]} -eq 0 ]]; then
        echo "  (none yet — type a model id below to download on first run)"
    else
        for i in "${!installed_names[@]}"; do
            human=$(numfmt --to=iec --suffix=B "${installed_sizes[$i]}" 2>/dev/null \
                    || echo "${installed_sizes[$i]} B")
            printf "  %2d. %-50s  %s\n" "$((i+1))" "${installed_names[$i]}" "$human"
        done
        if [[ ${#installed_names[@]} -gt 1 ]]; then
            total_h=$(numfmt --to=iec --suffix=B "$installed_total_bytes" 2>/dev/null \
                      || echo "$installed_total_bytes B")
            echo "  (${#installed_names[@]} models, total $total_h on disk)"
        fi
    fi
    echo
    echo "  or type any Hub model id (e.g. ${DEFAULT_MODEL}) to download."
    echo
    if [[ ${#installed_names[@]} -gt 0 ]]; then
        prompt_text="Pick [1-${#installed_names[@]}] or paste a model id"
    else
        prompt_text="Model id"
    fi

    while true; do
        read -r -p "$prompt_text: " MODEL
        # Empty -> ask again
        if [[ -z "$MODEL" ]]; then
            continue
        fi
        # Numeric selection from the list
        if [[ "$MODEL" =~ ^[0-9]+$ ]]; then
            if (( MODEL >= 1 && MODEL <= ${#installed_names[@]} )); then
                MODEL="${installed_names[$((MODEL-1))]}"
                break
            else
                echo "Enter 1-${#installed_names[@]} (or a model id)."
                continue
            fi
        fi
        # Plain model id: must look like Org/Name
        if [[ "$MODEL" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]]; then
            break
        fi
        echo "Model id must be 'Org/Name' (e.g. Qwen/Qwen2.5-7B-Instruct)."
    done
fi

# If still empty (no TTY, no arg), use the default.
if [[ -z "$MODEL" ]]; then
    MODEL="$DEFAULT_MODEL"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Is $1 (model id like "Org/Name") really downloaded?
# A "downloaded" model has at least one *.safetensors shard > 100 MB
# in its snapshots dir. Config + tokenizer files alone (a few KB)
# don't count — they'd fool a naive "is the dir non-empty?" check.
is_model_downloaded() {
    local id="$1"
    [[ -n "$id" ]] || return 1
    local slug="${id//\//--}"
    local snap="$HF_CACHE/hub/models--$slug/snapshots"
    [[ -d "$snap" ]] && \
        find "$snap" -name '*.safetensors' -size +100M 2>/dev/null | grep -q .
}

# What model is the running vllm-server serving? Reads from the last
# 4kB of its logs. vLLM prints "non-default args: {'model': '<id>', ...}"
# on startup, so we just look for any 'Org/Name'-shaped single-quoted token.
running_model() {
    docker logs vllm-server --tail 200 2>&1 \
        | grep -oE "'[A-Za-z0-9._-]+/[A-Za-z0-9._-]+'" \
        | tail -1 \
        | tr -d "'" \
        || true
}

# How much GPU memory is currently in use (MiB).
gpu_mem_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' '
}

# Wait until the GPU reports near-zero usage, or timeout.
# Args: timeout seconds (default 120)
wait_for_gpu_free() {
    local timeout="${1:-120}"
    local start=$SECONDS
    while :; do
        local used
        used="$(gpu_mem_used_mib)"
        if [[ -z "$used" || "$used" -lt 200 ]]; then
            return 0
        fi
        if (( SECONDS - start > timeout )); then
            echo "  (still ${used} MiB after ${timeout}s; continuing anyway)"
            return 1
        fi
        sleep 5
    done
}

# ---------------------------------------------------------------------------
# Decide what to do
# ---------------------------------------------------------------------------
docker network create "$NETWORK" >/dev/null 2>&1 || true

CURRENT_MODEL="$(running_model)"
echo "Current $NAME:  ${CURRENT_MODEL:-not running}"
echo "Requested:    $MODEL"
echo "Downloaded:   $(is_model_downloaded "$MODEL" && echo yes || echo no)"

# Case A: same model already running → no-op.
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME" \
        && [[ -n "$CURRENT_MODEL" && "$CURRENT_MODEL" == "$MODEL" ]]; then
    echo "vllm-server is already running with $MODEL — leaving it alone."
    echo "  logs:  docker logs -f $NAME"
    echo "  stop:  docker stop $NAME"
    exit 0
fi

# Case B: different model needed (or no container running) and the new
# model's weights aren't on disk yet → fetch them FIRST. The OLD vllm-server
# keeps serving until we explicitly stop it. Then we stop, wait for GPU to
# free, and start the new one (which loads from cache in ~30 s).
if ! is_model_downloaded "$MODEL"; then
    echo "Downloading $MODEL weights to $HF_CACHE (uses plain curl — the HF Python library stalls on this host) …"
    "$SCRIPT_DIR/download-model.sh" "$MODEL" "$HF_CACHE"
    echo "  download complete."
fi

# Stop the OLD vllm-server (brief service interruption starts here).
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$NAME"; then
    echo "Swapping $NAME from ${CURRENT_MODEL:-<unknown>} to $MODEL …"
    docker stop "$NAME" >/dev/null 2>&1 || true
    docker wait "$NAME" >/dev/null 2>&1 || true
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "  waiting for GPU memory to free…"
    wait_for_gpu_free 120
fi

# Start the NEW vllm-server (loads from cache — ~30 s instead of hours).
docker run -d \
    --name "$NAME" \
    --network "$NETWORK" \
    --restart unless-stopped \
    --gpus all \
    --ipc=host \
    -p "${PORT}:8000" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    --entrypoint python3 \
    vllm-a4500:latest \
    -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

echo
echo "vLLM server '$NAME' starting."
echo "  model:  $MODEL"
echo "  host:   http://localhost:${PORT}"
echo "  api:    http://localhost:${PORT}/v1"
echo "  logs:   docker logs -f $NAME"
echo "  stop:   docker stop $NAME"
echo
echo "Next steps:"
echo "  ./start-gateway.sh                              # FastAPI gateway on :9000"
echo "  ./prompt.sh 'What is the capital of France?'    # one-shot CLI"
