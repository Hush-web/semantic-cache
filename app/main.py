from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger
from typing import Optional

from .router import ProviderRouter
from .cache import SemanticCache
from .auth import APIKeyAuth
from .usage import UsageTracker

app = FastAPI(title="Semantic Cache Proxy")
router = ProviderRouter()
cache = SemanticCache(threshold=0.5)
auth = APIKeyAuth()
usage = UsageTracker()


LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Semantic Cache Gateway — Cut your LLM costs</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 760px;
      margin: 60px auto;
      padding: 0 20px;
      line-height: 1.65;
      color: #111;
      background: #fafafa;
    }
    h1 { font-size: 32px; margin-bottom: 8px; letter-spacing: -0.02em; }
    h2 { font-size: 20px; margin-top: 40px; margin-bottom: 12px; letter-spacing: -0.01em; }
    p { color: #333; }
    a { color: #0369a1; }
    code { background: #eef2f6; padding: 2px 6px; border-radius: 4px; font-size: 13px; font-family: ui-monospace, "SF Mono", Menlo, monospace; }
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 18px;
      border-radius: 10px;
      overflow-x: auto;
      font-size: 12.5px;
      font-family: ui-monospace, "SF Mono", Menlo, monospace;
      line-height: 1.6;
    }
    .badge {
      display: inline-block;
      background: #dcfce7;
      color: #166534;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    .card {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 20px;
      margin: 20px 0;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }
    .grid div {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 14px;
    }
    .grid strong { display: block; font-size: 13px; color: #666; margin-bottom: 4px; font-weight: 500; }
    footer { margin-top: 60px; padding-top: 24px; border-top: 1px solid #e5e7eb; font-size: 13px; color: #666; }
    @media (max-width: 600px) {
      .grid { grid-template-columns: 1fr; }
      h1 { font-size: 26px; }
    }
  </style>
</head>
<body>
  <span class="badge">LIVE IN PRODUCTION</span>
  <h1>Semantic Cache Gateway</h1>
  <p>Cut your LLM API costs by caching queries <strong>by meaning</strong>, not exact text. Bring your own Groq or OpenAI key. One line to integrate.</p>

  <h2>What it does</h2>
  <div class="card">
    <p>Your client points at our URL instead of OpenAI's. We embed the query, search for a semantically similar past query, and return the cached answer instantly. On a cache miss, we forward to your LLM provider using <strong>your</strong> key.</p>
    <p style="margin-bottom:0"><strong>Result:</strong> identical or similar queries cost nothing after the first call.</p>
  </div>

  <h2>Why BYOK (Bring Your Own Key)</h2>
  <div class="grid">
    <div><strong>You keep control</strong>Your Groq/OpenAI key never leaves your server.</div>
    <div><strong>You pay the provider</strong>Direct billing with Groq or OpenAI.</div>
    <div><strong>You pay us for the cache</strong>Only the caching layer is billed by us.</div>
    <div><strong>No vendor lock-in</strong>Point away any time. One URL change.</div>
  </div>

  <h2>Try it in 30 seconds</h2>
  <p>Send this twice. The first call is a <code>MISS</code>. The second is a <code>HIT</code> — served from cache, zero cost.</p>
  <pre>curl -X POST https://semantic-cache-6qvc.onrender.com/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer demo_key" \\
  -H "X-Upstream-Key: YOUR_GROQ_KEY" \\
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'</pre>

  <h2>Endpoints</h2>
  <div class="grid">
    <div><strong>POST /v1/chat/completions</strong>OpenAI-compatible chat completions with caching.</div>
    <div><strong>GET /stats</strong>Your usage, hit rate, and cost savings.</div>
    <div><strong>GET /health</strong>Service health and cache entry count.</div>
    <div><strong>GET /docs</strong>Interactive API documentation.</div>
  </div>

  <h2>Response headers</h2>
  <div class="grid">
    <div><strong>X-Cache</strong>HIT or MISS</div>
    <div><strong>X-Cache-Distance</strong>Cosine distance of the match (lower is closer)</div>
    <div><strong>X-BYOK</strong>true if your key was used</div>
    <div><strong>X-Tenant</strong>Your tenant ID</div>
  </div>

  <footer>
    <p><a href="/docs">API docs</a> · <a href="https://github.com/Hush-web/semantic-cache">GitHub</a></p>
    <p>Built with FastAPI, Qdrant Cloud, Fastembed, and Groq.</p>
  </footer>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return LANDING_PAGE


@app.get("/health")
async def health():
    return {"status": "healthy", "cache_entries": cache.count()}


@app.get("/stats")
async def stats(authorization: Optional[str] = Header(None)):
    api_key = _extract_key(authorization)
    tenant_id = auth.validate(api_key)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return usage.get_stats(tenant_id)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
    x_upstream_key: Optional[str] = Header(None, alias="X-Upstream-Key")
):
    try:
        # 1. Auth against our service
        api_key = _extract_key(authorization)
        tenant_id = auth.validate(api_key)
        if not tenant_id:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        body = await request.json()
        if "messages" not in body:
            raise HTTPException(status_code=400, detail="Missing 'messages' field")

        messages = body["messages"]
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1]["content"]
        logger.info(
            f"Request: model={body.get('model')} tenant={tenant_id} "
            f"byok={x_upstream_key is not None}"
        )

        # 2. Cache check
        cached = cache.check(query, tenant_id)
        if cached:
            usage.record_hit(tenant_id)
            return JSONResponse(
                content={
                    "id": "cache-hit",
                    "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cached["response"]
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                },
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Distance": f"{cached['distance']:.3f}",
                    "X-Tenant": tenant_id,
                    "X-BYOK": "true" if x_upstream_key else "false"
                }
            )

        # 3. Cache miss — need upstream key
        if not x_upstream_key:
            raise HTTPException(
                status_code=402,
                detail=(
                    "Cache miss. Provide your own LLM key in 'X-Upstream-Key' header, "
                    "or use our managed key."
                )
            )

        # 4. Call provider with customer's key
        response = await router.route(body, upstream_key=x_upstream_key)
        usage.record_miss(tenant_id)

        # 5. Store in cache
        try:
            assistant_message = response["choices"][0]["message"]["content"]
            cache.store(query, assistant_message, tenant_id)
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

        return JSONResponse(
            content=response,
            headers={
                "X-Cache": "MISS",
                "X-Tenant": tenant_id,
                "X-BYOK": "true"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _extract_key(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization