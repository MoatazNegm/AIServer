"""Tiny FastAPI gateway that forwards prompts to a vLLM OpenAI-compatible server.

Endpoints:
  GET  /healthz                       Liveness — this process is up. Public (no key).
  GET  /readyz                        Readiness — upstream vLLM is reachable. Public (no key).
  GET  /v1/models                     OpenAI-compatible. List models. Requires API key.
  POST /v1/chat/completions           OpenAI-compatible. Standard chat. Requires API key.
  POST /v1/completions                OpenAI-compatible. Legacy completions. Requires API key.
  POST /v1/prompt                     Convenience: {prompt} → {reply}. Requires API key.

Authentication:
  Set the API_KEYS env var with one or more keys (comma-separated). Each
  request to /v1/* must present the key via either:
      Authorization: Bearer <key>
      X-API-Key: <key>

  A request with a missing or wrong key gets HTTP 401 + a JSON error.
  The /healthz and /readyz endpoints stay unauthenticated so an
  orchestrator can probe the gateway without needing a key.

Why auth at the gateway and not at vLLM:
  vLLM runs inside the private `vllm-net` Docker network. It's not
  reachable from the host's external network — only the gateway is. So we
  treat the vLLM container as a backend that doesn't need to know who's
  calling; the gateway does the auth at the only public edge.

Env vars:
  VLLM_URL        (default http://vllm-server:8000) — upstream base URL
  VLLM_MODEL      (default Qwen/Qwen2.5-7B-Instruct) — model name to send
  VLLM_MAX_TOKENS (default 256) — cap on response length
  API_KEYS        (default unset) — comma-separated list of valid keys
  API_KEYS_FILE   (optional) — path to a file of keys (one per line);
                            takes precedence over API_KEYS if both are set
"""
import os
import secrets
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

VLLM_URL = os.environ.get("VLLM_URL", "http://vllm-server:8000").rstrip("/")
DEFAULT_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_MAX_TOKENS = int(os.environ.get("VLLM_MAX_TOKENS", "256"))

# Server-side caps to stop clients (e.g. KiloCode) from overshooting the
# model's context window. Two knobs:
#   - MODEL_MAX_CONTEXT: total input+output tokens the model supports.
#                        Must match start-vllm.sh --max-model-len (default 32768).
#   - MAX_OUTPUT_TOKENS_HARD_CAP: absolute ceiling for a single reply,
#                                 regardless of how much room the prompt uses.
MODEL_MAX_CONTEXT = int(os.environ.get("MODEL_MAX_CONTEXT", "32768"))
MAX_OUTPUT_TOKENS_HARD_CAP = int(os.environ.get("MAX_OUTPUT_TOKENS_HARD_CAP", "16384"))

# Model-agnostic proxying. The gateway queries vLLM every MODEL_CACHE_TTL
# seconds and rewrites incoming `model` fields to whatever is actually
# loaded. This lets an agent (KiloCode etc.) keep a fixed model in its
# config and still hit whatever is currently running — useful when the
# admin swaps models on the host. Set MODEL_REWRITE=disabled to disable.
import time

_MODEL_CACHE = {"id": None, "fetched_at": 0.0}
_MODEL_CACHE_TTL = int(os.environ.get("MODEL_CACHE_TTL", "30"))
_MODEL_REWRITE = os.environ.get("MODEL_REWRITE", "enabled").lower() != "disabled"


async def _current_vllm_model() -> str | None:
    """Return the model id vLLM is currently serving, cached for TTL seconds.

    Calls vLLM's /v1/models and returns the first `id`. Returns None if
    vLLM is unreachable or no model is loaded yet — callers should fall
    through to letting vLLM emit its own 503.
    """
    now = time.time()
    if _MODEL_CACHE["id"] is not None and (now - _MODEL_CACHE["fetched_at"]) < _MODEL_CACHE_TTL:
        return _MODEL_CACHE["id"]
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{VLLM_URL}/v1/models")
        if r.status_code != 200:
            return _MODEL_CACHE["id"]
        for m in r.json().get("data", []):
            if m.get("id"):
                _MODEL_CACHE["id"] = m["id"]
                _MODEL_CACHE["fetched_at"] = now
                return _MODEL_CACHE["id"]
    except Exception:
        pass
    return _MODEL_CACHE["id"]


async def _maybe_rewrite_model(body: dict) -> str | None:
    """If the client's requested model differs from what vLLM serves,
    rewrite it in-place. Returns a human-readable note describing the swap
    (suitable for an `X-Model-Rewrite` response header), or None if no
    rewrite was needed.
    """
    if not _MODEL_REWRITE:
        return None
    requested = body.get("model")
    if not requested:
        return None
    current = await _current_vllm_model()
    if not current or current == requested:
        return None
    body["model"] = current
    return f"{requested} → {current}"


def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate from a list of OpenAI-style chat messages.

    We can't reach HuggingFace from inside this container (vllm-net has
    no DNS for huggingface.co), so we can't load the real Qwen
    tokenizer. We use a deliberately-conservative heuristic: 2 chars
    per token. English prose is ~4 chars/token so this overestimates
    by ~2x (which only makes output caps slightly tighter — harmless).
    Sparse content like "x " repeated or short tokens in code can be
    1-2 chars/token, and our estimate handles that correctly.
    """
    total = 0
    for m in messages:
        # 8 tokens of per-message overhead (role, separators, etc.)
        total += 8
        c = m.get("content")
        if isinstance(c, str):
            total += len(c) // 2
        # Tool-call messages carry an extra JSON blob
        if "tool_calls" in m and isinstance(m["tool_calls"], list):
            for tc in m["tool_calls"]:
                total += len(json.dumps(tc)) // 2
        if m.get("tool_call_id"):
            total += 8 + len(m["tool_call_id"]) // 2
    # ~5% safety margin so we don't land exactly on the boundary
    return int(total * 1.05) + 16


def _cap_max_tokens(body: dict) -> tuple[int, int, str | None]:
    """Returns (requested, capped, note). `note` is non-null if we lowered."""
    requested = int(body.get("max_tokens") or 1024)
    # 1. Hard ceiling so a single reply can't be absurd.
    cap = min(requested, MAX_OUTPUT_TOKENS_HARD_CAP)
    # 2. Reserve room for input.
    msgs = body.get("messages") or []
    used_input = _estimate_tokens(msgs) if isinstance(msgs, list) else 0
    room_for_output = MODEL_MAX_CONTEXT - used_input - 32  # safety
    if room_for_output < 64:
        room_for_output = 64  # never go below 64 — agent still needs SOMETHING
    cap = min(cap, room_for_output)
    note = None
    if cap < requested:
        note = (
            f"requested {requested}; gateway capped to {cap} "
            f"(input est. {used_input}, model ctx {MODEL_MAX_CONTEXT})"
        )
    return requested, cap, note


# ---------------------------------------------------------------------------
# API key loading
# Reads once at startup. To rotate keys: edit .api-keys on the host,
# then `docker restart vllm-gateway`. (See gateway-keygen.sh --add for
# the canonical workflow.)
# ---------------------------------------------------------------------------
def _load_keys() -> set[str]:
    keys: set[str] = set()
    keys_file = os.environ.get("API_KEYS_FILE", "").strip()
    if keys_file:
        p = Path(keys_file)
        if p.is_file():
            for raw in p.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                keys.add(line)
    env_keys = os.environ.get("API_KEYS", "").strip()
    if env_keys:
        for k in env_keys.split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


VALID_KEYS: set[str] = _load_keys()
AUTH_REQUIRED = bool(VALID_KEYS)


# json.dumps is needed by _estimate_tokens when tool_calls are present.
import json


def _extract_key(request: Request) -> str | None:
    """Pull the API key out of the Authorization header or X-API-Key."""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        scheme, _, token = auth.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token.strip()
    return request.headers.get("x-api-key") or request.headers.get("X-API-Key")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="vLLM Gateway", version="1.0")


class PromptRequest(BaseModel):
    prompt: str = Field(..., description="The user prompt to send to vLLM.")
    model: str | None = Field(default=None, description="Override the model.")
    system: str | None = Field(default=None, description="Optional system message.")
    temperature: float | None = Field(default=None, description="Sampling temperature.")
    max_tokens: int | None = Field(default=None, description="Max new tokens.")


class PromptResponse(BaseModel):
    reply: str
    model: str
    usage: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Auth dependency (only applied to protected endpoints)
# ---------------------------------------------------------------------------
async def require_api_key(request: Request) -> str:
    """FastAPI dependency — returns the key on success, raises 401 on failure.

    Skipped entirely when AUTH_REQUIRED is False (no keys configured).
    Useful for local dev: don't ship without keys in any real deploy.
    """
    if not AUTH_REQUIRED:
        # No keys configured -> open mode. Useful for the very first
        # `start-gateway.sh` before the admin mints keys.
        return ""

    provided = _extract_key(request)
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing API key",
            headers={"WWW-Authenticate": 'Bearer realm="vllm-gateway"'},
        )
    # Constant-time comparison to avoid timing leaks.
    matched = any(secrets.compare_digest(provided, k) for k in VALID_KEYS)
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid API key",
            headers={"WWW-Authenticate": 'Bearer realm="vllm-gateway"'},
        )
    return provided


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> dict:
    """Liveness — never touches upstream, never requires auth."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> JSONResponse:
    """Readiness — pings vLLM's /v1/models to confirm it's serving."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{VLLM_URL}/v1/models")
        if r.status_code != 200:
            return JSONResponse(
                {"status": "not_ready", "upstream": VLLM_URL,
                 "upstream_status": r.status_code, "body": r.text[:500]},
                status_code=503,
            )
        body = r.json()
        return {"status": "ready", "upstream": VLLM_URL, "models": body.get("data", [])}
    except Exception as e:
        return JSONResponse(
            {"status": "not_ready", "upstream": VLLM_URL, "error": repr(e)},
            status_code=503,
        )


@app.post("/v1/prompt", response_model=PromptResponse, dependencies=[Depends(require_api_key)])
async def prompt(req: PromptRequest) -> PromptResponse:
    """Forward one prompt to vLLM and return the reply as a single string."""
    model = req.model or DEFAULT_MODEL

    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": req.max_tokens if req.max_tokens is not None else DEFAULT_MAX_TOKENS,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature

    try:
        async with httpx.AsyncClient(timeout=180.0) as c:
            r = await c.post(f"{VLLM_URL}/v1/chat/completions", json=body)
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"could not reach upstream vLLM at {VLLM_URL}: {e!r}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"upstream vLLM returned HTTP {r.status_code}: {r.text[:500]}",
        )

    data = r.json()
    try:
        reply = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502, detail=f"unexpected upstream response: {data!r}"
        )

    return PromptResponse(
        reply=reply,
        model=data.get("model", model),
        usage=data.get("usage", {}) or {},
    )


# ---------------------------------------------------------------------------
# OpenAI-compatible passthrough endpoints.
#
# These accept the standard OpenAI request shape, send it verbatim to
# vLLM's own /v1/* endpoints, and return vLLM's response verbatim. Any
# OpenAI client (langchain, openai-python, curl, OpenAI Playground, …)
# that points its base URL at this gateway will Just Work.
# ---------------------------------------------------------------------------
async def _proxy(request: Request, method: str) -> Response:
    """Forward the current request to `request.url.path` on vLLM and
    return whatever vLLM returns (status, headers, body) verbatim."""
    # Re-build the upstream URL from the path the client sent.
    # We strip any /v1 prefix because vLLM's /v1 path is the same as ours.
    upstream_path = request.url.path
    upstream_url = f"{VLLM_URL}{upstream_path}"
    body = await request.body()

    # Forward only the headers that affect the response (e.g. content-type);
    # skip the host-internal ones.
    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "authorization", "x-api-key")
    }
    # Re-attach auth if vLLM had its own --api-key set; harmless otherwise.
    provided = _extract_key(request)
    if provided:
        fwd_headers["Authorization"] = f"Bearer {provided}"

    try:
        async with httpx.AsyncClient(timeout=180.0) as c:
            upstream = await c.request(
                method, upstream_url,
                content=body,
                params=request.query_params.multi_items(),
                headers=fwd_headers,
            )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"could not reach upstream vLLM at {VLLM_URL}: {e!r}",
        )

    # Pass the response through. Content-type stays whatever vLLM set
    # (application/json for /v1/*). Truncate body for sanity.
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


from fastapi import Response  # noqa: E402  (kept with the route that uses it)


@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def openai_list_models(request: Request) -> Response:
    """OpenAI-compatible: GET /v1/models → list models served by vLLM."""
    return await _proxy(request, "GET")


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def openai_chat_completions(request: Request) -> Response:
    """OpenAI-compatible: POST /v1/chat/completions.

    Two things happen before forwarding:
      - max_tokens is clamped to fit the model's context window
        (MODEL_MAX_CONTEXT minus estimated input, hard-capped at
        MAX_OUTPUT_TOKENS_HARD_CAP).
      - the `model` field is silently rewritten to whatever vLLM is
        actually serving (cached for MODEL_CACHE_TTL). This lets an
        agent keep its configured model and still hit the running one.
    """
    body = await request.json()
    rewrite_note = await _maybe_rewrite_model(body)
    requested, capped, cap_note = _cap_max_tokens(body)
    if capped != requested or "max_tokens" not in body:
        body["max_tokens"] = capped
    request._body = json.dumps(body).encode()
    resp = await _proxy(request, "POST")
    if cap_note:
        resp.headers["X-vLLM-Gateway-Cap"] = cap_note
    if rewrite_note:
        resp.headers["X-Model-Rewrite"] = rewrite_note
    return resp


@app.post("/v1/completions", dependencies=[Depends(require_api_key)])
async def openai_completions(request: Request) -> Response:
    """OpenAI-compatible: POST /v1/completions (legacy text completions)."""
    body = await request.json()
    rewrite_note = await _maybe_rewrite_model(body)
    requested, capped, cap_note = _cap_max_tokens(body)
    if capped != requested or "max_tokens" not in body:
        body["max_tokens"] = capped
    request._body = json.dumps(body).encode()
    resp = await _proxy(request, "POST")
    if cap_note:
        resp.headers["X-vLLM-Gateway-Cap"] = cap_note
    if rewrite_note:
        resp.headers["X-Model-Rewrite"] = rewrite_note
    return resp


