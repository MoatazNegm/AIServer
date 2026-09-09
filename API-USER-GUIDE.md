# vLLM Gateway — API User Guide

This guide is for an outside requestor who has just been given an API key
by the admin. It covers everything you need to send prompts and get replies.
No knowledge of Docker, vLLM, or the server is required.

---

## 1. What you've been given

- An **API key**: a long hex string, e.g. `41d487a8d26c99c0…09630d8d`.
- A **gateway URL**: the host and port of the gateway, e.g.
  `https://api.example.com:9000` or `http://192.168.8.62:9000`.

The gateway forwards your prompts to a local Qwen large-language-model and
returns the reply. Your API key is required on every request.

---

## 2. The only endpoint you need

```
POST /v1/prompt
```

Send a single prompt, get a single reply back. **There is no streaming,
no chat history, and no session state** — every call is independent.

---

## 3. Authentication

Include the key in **one** of these headers on every request:

```http
Authorization: Bearer 41d487a8d26c99c0…09630d8d
```

```http
X-API-Key: 41d487a8d26c99c0…09630d8d
```

A request with a missing or wrong key gets `401 {"detail":"missing API key"}`
or `401 {"detail":"invalid API key"}`.

---

## 4. Request body

| Field          | Type    | Required | Notes |
|----------------|---------|----------|-------|
| `prompt`       | string  | **yes**  | The user prompt. |
| `model`        | string  | no       | Override the model. Use whatever the admin tells you is loaded (e.g. `Qwen/Qwen2.5-7B-Instruct`). |
| `system`       | string  | no       | A system prompt prepended to the conversation. Use for "you are a poet", "answer in French", etc. |
| `temperature`  | number  | no       | Sampling temperature 0–2. Lower = more deterministic. Omit to use the server default. |
| `max_tokens`   | integer | no       | Maximum length of the reply in tokens. Omit to use the server default (typically 256). |

Anything else in the body is ignored.

---

## 5. Response body (on success)

```json
{
  "reply": "Paris is the capital of France.",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 6,
    "total_tokens": 18
  }
}
```

`reply` is the only field you usually care about. `usage` is informational.

---

## 6. Status codes

| Code | Meaning | What to do |
|------|---------|------------|
| `200` | Success | Use the `reply`. |
| `401` | Bad or missing API key | Check the header value. Ask the admin to confirm the key is still active. |
| `422` | Malformed body | The gateway didn't like the JSON (missing `prompt`, wrong types). |
| `500` / `502` | Gateway or upstream failure | The model server is unreachable, still loading, or crashed. Retry once after a few seconds; if persistent, ask the admin. |
| `503` | (only on `/readyz`) | vLLM is not ready yet. |
| Time-out | Network or model latency | Increase your client's timeout (give vLLM up to a minute for a long reply) and retry. |

---

## 7. Examples

### curl (command line)

```bash
curl -sS https://api.example.com:9000/v1/prompt \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?"}'
```

`-sS` is "silent + show errors". Drop the `-s` if you want the progress bar.

### Python (httpx / requests)

```python
import httpx

resp = httpx.post(
    "https://api.example.com:9000/v1/prompt",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"prompt": "What is the capital of France?"},
    timeout=120,
)
resp.raise_for_status()
print(resp.json()["reply"])
```

### JavaScript / fetch

```js
const r = await fetch("https://api.example.com:9000/v1/prompt", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ prompt: "What is the capital of France?" }),
});
const { reply } = await r.json();
console.log(reply);
```

### VS Code (e.g. Continue, Codeium, Cline extensions)

Most LLM-style VS Code extensions accept an **OpenAI-compatible base URL**.
Point them at `https://api.example.com:9000/v1` and put the API key in
the extension's "API Key" field. The extension will send standard
OpenAI-shaped requests; the gateway will return a 200 with `reply` instead
of `choices[0].message.content`. If the extension insists on the standard
OpenAI shape, you can also hit `vllm-server:8000` directly (the admin
exposes that port too — ask).

---

## 8. Things to avoid

- **Don't log the API key.** It will end up in screenshots, bug reports, and CI artifacts.
- **Don't share the key.** If you do, ask the admin to revoke it (`delete the line in /root/gpuenable/.api-keys; docker restart vllm-gateway`).
- **Don't put the key in source code.** Use environment variables, your platform's secret manager, or your IDE's secret store.
- **Don't reuse the key across multiple people.** Ask the admin for a key per person so revoking one doesn't affect the others.

---

## 9. Quick sanity-check (once you have the key + URL)

```bash
curl -sS https://api.example.com:9000/healthz
# → {"status":"ok"}

curl -sS https://api.example.com:9000/readyz
# → {"status":"ready",...}   ← when vLLM is up and the model is loaded
# → {"status":"not_ready",...} ← when vLLM is still starting
```

Both work **without** an API key (the admin keeps them open for liveness probes).
The one that needs your key is `POST /v1/prompt`.

---

## 10. If something goes wrong

| Symptom | Likely cause | What to try |
|---------|--------------|-------------|
| `401 missing API key` | No header sent, or wrong scheme (`Token …` instead of `Bearer …`) | Add `Authorization: Bearer <key>` |
| `401 invalid API key` | Wrong key, or the admin revoked yours | Ask the admin to mint a new one |
| `502 could not reach upstream vLLM` | vLLM is down or still booting | Wait a few seconds, retry. If it persists >5 min, ask the admin. |
| Empty / weird `reply` | vLLM was killed mid-response | Retry the call. |
| Connection refused / timeout | Wrong host or port, or firewall blocking | Verify the URL with the admin. |
