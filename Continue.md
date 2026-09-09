# vLLM + GPU bring-up — checkpoint log

Goal: run vLLM in a Docker container against the host's NVIDIA GPU
(**NVIDIA RTX A4500**, 20GB, GA102GL @ PCI 65:00.0). User chose the **docker** path.

## Status (2026-09-08)

- ✅ Kernel `5.14.0-687.42.1.el9_8` with NVIDIA proprietary driver `580.159.04`
  (akmod-built; see "Gotchas" for why the prebuilt kmod won't load)
- ✅ `nvidia-smi` shows RTX A4500, Driver 580.159.04, CUDA 13.0
- ✅ `nvidia-container-toolkit` wired into docker (`Default Runtime: nvidia`)
- ✅ **`vllm-a4500:latest` image is built, ready to run** — 4.48 GB
- ✅ **GPU smoke test passes inside the container**:
  `torch 2.7.1+cu126 cuda 12.6 avail True`,
  `device: NVIDIA RTX A4500`, 19.6 GiB
- ✅ Wetty ping timeouts patched (3s/7s → 30s/180s) and **persisted** via a
  custom `gpuenable/wetty:patched` image (boot-time `manage.sh` uses it
  instead of upstream `moataznegm/quickstor:wetty`, so the patch survives
  reboots).
- ✅ **API-key auth on the gateway**: `POST /v1/prompt` requires an
  `Authorization: Bearer <key>` (or `X-API-Key: <key>`) header. Keys live
  in `/root/gpuenable/.api-keys` (mode 600, one key per line, `#` for
  comments). Mint new keys with `./gateway-keygen.sh --add <label>`.
- ⚠️ **Model download is slow** — HuggingFace is reachable but at ~250 KB/s from
  this host. First model pull takes 1–6+ hours depending on size. After
  the first pull, the model is cached and subsequent runs are fast.

## Two-container setup

```
                    ┌────────────────────────────────┐
                    │  vllm-gateway (FastAPI :9000) │
                    │  /healthz   open              │
                    │  /readyz    open              │
                    │  /v1/prompt requires API key  │
                    └──────────────┬─────────────────┘
                                   │ http (vllm-net Docker DNS)
                                   ▼
                    ┌────────────────────────────────┐
                    │  vllm-server (vLLM OpenAI :8000)│
                    │  on private vllm-net, no auth   │
                    │  vllm-a4500:latest             │
                    └──────────────┬─────────────────┘
                                   │ CUDA
                                   ▼
                              RTX A4500 20GB
```

## How to run a model

The repo has one-shot helpers:

```bash
/root/gpuenable/start-vllm.sh                          # default: Qwen2.5-7B-Instruct
/root/gpuenable/start-vllm.sh Qwen/Qwen2.5-3B-Instruct # smaller, faster first download
/root/gpuenable/start-vllm.sh Qwen/Qwen2.5-1.5B-Instruct # smallest, instant
/root/gpuenable/start-vllm.sh mistralai/Mistral-7B-Instruct-v0.3
# No arg + TTY → interactive picker: lists already-downloaded models with
# disk size, lets you pick by number, or type a new model id to download.

/root/gpuenable/start-gateway.sh    # FastAPI gateway on :9000, auth enabled

/root/gpuenable/prompt.sh --api-key=<KEY> "your prompt here"   # one-shot CLI
# or: GATEWAY_API_KEY=<KEY> /root/gpuenable/prompt.sh "..."
```

Logs: `docker logs -f vllm-server`. Stop: `docker stop vllm-server vllm-gateway`.

## Admin: manage API keys

```bash
# Mint a new key, optionally append it to /root/gpuenable/.api-keys:
/root/gpuenable/gateway-keygen.sh --add "alice"
# → prints a fresh 64-char hex key, adds a labeled comment, restarts hint.

# Revoke / view existing keys:
cat /root/gpuenable/.api-keys       # chmod 600, root only
# Delete the line for the key you want to revoke, then:
docker restart vllm-gateway         # picks up the new file content
```

Auth rules: missing or wrong key → 401. `/healthz` and `/readyz` stay open
so a load balancer can probe the gateway without a key. The upstream
vLLM-server has no auth — it's only reachable from inside the `vllm-net`
Docker network, so the gateway is the single public edge.

## Models that fit the 20 GB A4500

| Model | ~Size (fp16) | Quality notes |
|-------|--------------|---------------|
| Qwen2.5-1.5B-Instruct      | ~3 GB  | Fast, weaker reasoning |
| **Qwen2.5-3B-Instruct**    | ~6 GB  | **Best balance** — strong general quality, fast download |
| **Qwen2.5-7B-Instruct**    | ~14 GB | **Best quality that fits** — Meta-Llama-3.1-8B tier |
| Mistral-7B-Instruct-v0.3    | ~14 GB | Comparable to Qwen 7B |
| Qwen2-VL-7B-Instruct        | ~14 GB | Vision-language; needs `--trust-remote-code` |

Anything bigger than ~14 GB of weights won't fit alongside the KV cache.
14B-class models (Qwen2.5-14B, Llama-3.1-13B) won't fit in fp16 on this GPU.

## Network restrictions on this host

Outbound HTTPS works ONLY to: **nvcr.io**, **pypi.org**, **huggingface.co**,
**download.pytorch.org**, plus Chinese PyPI mirrors (Tsinghua, Aliyun, etc.).
NOT reachable: docker.io, registry-1.docker.io, ghcr.io, quay.io.
- Use `nvcr.io/nvidia/...` images, NOT `nvidia/...` (which is docker.io).
- `vllm/vllm-openai` from docker.io is unreachable → we built it ourselves.
- Tsinghua mirror (`https://pypi.tuna.tsinghua.edu.cn/simple`) is faster
  than pypi.org from this host.
- HF model downloads work but are slow.

## Gotchas (worth knowing)

1. **akmod, not prebuilt kmod**. RPMFusion's `kmod-nvidia-580xx-5.14.0-687.el9_8`
   is built for the older `-687.5.3.el9_8` point release; loading it in
   `-687.42.1.el9_8` fails with
   `disagrees about version of symbol set_pages_array_wb`. Always use
   `akmod-nvidia-580xx` to compile against the running kernel.

2. **`nvidia-cusparselt-cu12` doesn't follow the `nvidia/` prefix convention** —
   it installs to `cusparselt/lib/`. A `find` under `nvidia/` misses it →
   `ImportError: libcusparseLt.so.0`. Scan the whole `site-packages`.

3. **Dockerfile `ENV` doesn't see shell variables from prior `RUN` steps.**
   `${NVIDIA_LIB_PATHS}` in ENV is empty if defined inside `RUN $(...)`.
   Hardcode the path list in `ENV LD_LIBRARY_PATH=...`.

4. **Triton needs gcc at runtime.** Without `gcc`, vLLM fails with
   "Failed to find C compiler" the first time you load a model. The base
   `cuda:*-base-ubi9` image does NOT include gcc — install it (`gcc gcc-c++ make`).

5. **vLLM 0.10.0 was tested against `transformers==4.45.x` but pip pulls 5.x.**
   `tokenizer.all_special_tokens_extended` is removed in transformers 5 → vLLM
   fails at tokenizer init on every modern model. Pin `transformers<5` in the
   Dockerfile (already done).

6. **`docker build` has no GPU** → `torch.cuda.is_available()=False` in the
   build sanity check is normal. The real GPU test happens at
   `docker run --gpus all` time. Don't `assert` it during build.

7. **wetty 2.5.0 hard-codes 3s/7s socket.io ping timeouts** in its bundled
   `/usr/src/app/build/server/socketServer/socket.js`. Patch in-place with sed +
   `docker restart wetty`. See `wetty-ping-timeout` memory for the one-liner.

8. **Long downloads from this host often stall mid-stream** (TCP connection
   established but no data flowing). Pure wait + retry eventually moves, but
   it's slow. HF downloads tend to be the bottleneck on this network.

## Files in this directory

- `vllm.Dockerfile` — Dockerfile for the vllm-a4500 image
- `start-vllm.sh` — one-line launcher; takes model name as arg
- `nginx.conf`, `start-nginx-ssh-proxy.sh`, `ngrok.env` — pre-existing
  SSH-tunnel proxy setup, unrelated to this work

## Resume instructions for a fresh Claude session

1. Read this file.
2. Read `/root/.claude/projects/-root-gpuenable/memory/` for auto-loaded project + wetty notes.
3. Verify state:
   ```bash
   uname -r                            # 5.14.0-687.42.1.el9_8.x86_64
   nvidia-smi | head -8                # RTX A4500, Driver 580.159.04
   docker images | grep vllm           # vllm-a4500:latest
   ```
4. If `nvidia-smi` is missing the A4500: reinstall the kmod (see Gotcha #1).
5. To serve a model: see "How to run a model" above.
