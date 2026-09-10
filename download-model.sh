#!/usr/bin/env bash
# download-model.sh — download a HuggingFace model's weights to the local
# HF cache, using plain `curl` (the HF Python library sometimes stalls on
# this host; plain curl at ~500 KB/s is reliable).
#
# Usage:
#   ./download-model.sh <Org/Name> [cache_dir]
#
# Default cache_dir = $HOME/.cache/huggingface (matches what vLLM uses).

set -euo pipefail

MODEL="${1:?usage: download-model.sh <Org/Name> [cache_dir]}"
CACHE="${2:-$HOME/.cache/huggingface}"
SLUG="${MODEL//\//--}"
HF_BASE="https://huggingface.co"
mkdir -p "$CACHE"

echo "Downloading $MODEL into $CACHE ..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Resolve the index — for sharded models this lists the shard filenames.
# For single-file models, there's no index; we'll just glob for safetensors.
echo "  → resolving file list"
INDEX_FILE="$TMP/index.json"
if curl -sf -L -o "$INDEX_FILE" "$HF_BASE/$MODEL/resolve/main/model.safetensors.index.json"; then
    files=($(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); [v for v in sorted(set(d['weight_map'].values()))]" "$INDEX_FILE"))
elif curl -sf -L -o "$TMP/single.safetensors" -r 0-0 "$HF_BASE/$MODEL/resolve/main/model.safetensors"; then
    # single-shard: HEAD shows it exists; download via resume below
    files=(model.safetensors)
else
    echo "ERROR: couldn't find $MODEL safetensors on HF" >&2
    exit 1
fi

# Also fetch the small support files (tokenizer/config) — needed by vLLM
echo "  → support files (tokenizer, config)"
SUPPORT_FILES=(config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt)
for f in "${SUPPORT_FILES[@]}"; do
    curl -sf -L -o "$TMP/$f" "$HF_BASE/$MODEL/resolve/main/$f" 2>/dev/null || true
done

# Place the model dir on disk in the HF cache layout
mkdir -p "$CACHE/hub/models--$SLUG/snapshots"
SNAP="$CACHE/hub/models--$SLUG/snapshots/$(date +%s)-download"
mkdir -p "$SNAP"

# Download each shard in parallel (up to 4 at a time)
echo "  → weights: ${#files[@]} shard(s)"
pids=()
for f in "${files[@]}"; do
    (
        curl -L --retry 5 --retry-delay 2 -C - \
             -o "$SNAP/$f" \
             "$HF_BASE/$MODEL/resolve/main/$f"
    ) &
    pids+=($!)
    # limit to 4 parallel
    if (( ${#pids[@]} >= 4 )); then
        wait "${pids[@]}"
        pids=()
    fi
done
wait "${pids[@]}"

# Copy the support files
for f in "${SUPPORT_FILES[@]}"; do
    [ -f "$TMP/$f" ] && cp "$TMP/$f" "$SNAP/$f"
done

# Make the snapshot the latest
rm -f "$CACHE/hub/models--$SLUG/latest"
ln -s "$SNAP" "$CACHE/hub/models--$SLUG/latest"

# Summary
size=$(du -sh "$CACHE/hub/models--$SLUG" 2>/dev/null | awk '{print $1}')
echo "Done. $MODEL → $CACHE/hub/models--$SLUG/  ($size)"
echo "  → snapshot: $SNAP"
