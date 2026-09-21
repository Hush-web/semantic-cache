from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse, HTMLResponse
from loguru import logger
from typing import Optional
from pathlib import Path
import time

from .router import ProviderRouter
from .cache import SemanticCache
from .auth import APIKeyAuth
from .usage import UsageTracker
from .metrics import metrics
from .guardrails import scan_messages

app = FastAPI(title="Semantic Cache Gateway")
router = ProviderRouter()
cache = SemanticCache(threshold=0.5)
auth = APIKeyAuth()
usage = UsageTracker()

LANDING_PAGE_PATH = Path(__file__).parent / "landing.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page."""
    try:
        return HTMLResponse(content=LANDING_PAGE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Semantic Cache Gateway</h1><p>landing.html not found.</p>",
            status_code=200,
        )


@app.get("/health")
async def health():
    return {"status": "healthy", "cache_entries": cache.count()}


@app.get("/metrics")
async def get_metrics():
    """Return live metrics for the gateway."""
    return metrics.snapshot()


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
    x_upstream_key: Optional[str] = Header(None, alias="X-Upstream-Key"),
):
    start_time = time.perf_counter()
    tenant_id = "unknown"

    try:
        # 1. Auth
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

        # 2. Guardrail scan
        guard = scan_messages(messages)

        if guard["blocked"]:
            metrics.record_blocked()
            logger.warning(
                f"BLOCKED injection from tenant={tenant_id} "
                f"score={guard['score']} matches={guard['matches']}"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    "Request blocked: prompt injection detected. "
                    f"score={guard['score']}"
                ),
            )

        if guard["flagged"]:
            logger.warning(
                f"FLAGGED prompt from tenant={tenant_id} "
                f"score={guard['score']} matches={guard['matches']}"
            )

        # 3. Cache check — exact first, then semantic
        cache_start = time.perf_counter()
        cached = cache.check_exact(query, tenant_id)

        if not cached:
            cached = cache.check(query, tenant_id)

        cache_latency_ms = (time.perf_counter() - cache_start) * 1000

        if cached:
            usage.record_hit(tenant_id)
            total_latency_ms = (time.perf_counter() - start_time) * 1000

            metrics.record_request(
                tenant_id=tenant_id,
                latency_ms=total_latency_ms,
                cache_status="HIT",
                cache_latency_ms=cache_latency_ms,
            )

            return JSONResponse(
                content={
                    "id": "cache-hit",
                    "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": cached["response"],
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                },
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Type": cached.get("cache_type", "SEMANTIC"),
                    "X-Cache-Distance": f"{cached['distance']:.3f}",
                    "X-Tenant": tenant_id,
                    "X-BYOK": "true" if x_upstream_key else "false",
                    "X-Latency-Ms": f"{total_latency_ms:.0f}",
                    "X-Security-Score": str(guard["score"]),
                    "X-Security-Status": guard["category"].upper(),
                },
            )

        # 4. Cache miss — need upstream key
        if not x_upstream_key:
            raise HTTPException(
                status_code=402,
                detail=(
                    "Cache miss. Provide your own LLM key in 'X-Upstream-Key' "
                    "header, or use our managed key."
                ),
            )

        # 5. Call provider with customer's key
        llm_start = time.perf_counter()
        response = await router.route(body, upstream_key=x_upstream_key)
        llm_latency_ms = (time.perf_counter() - llm_start) * 1000

        usage.record_miss(tenant_id)

        # 6. Store in cache
        try:
            assistant_message = response["choices"][0]["message"]["content"]
            cache.store(query, assistant_message, tenant_id)
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        metrics.record_request(
            tenant_id=tenant_id,
            latency_ms=total_latency_ms,
            cache_status="MISS",
            cache_latency_ms=cache_latency_ms,
            llm_latency_ms=llm_latency_ms,
        )

        return JSONResponse(
            content=response,
            headers={
                "X-Cache": "MISS",
                "X-Tenant": tenant_id,
                "X-BYOK": "true",
                "X-Latency-Ms": f"{total_latency_ms:.0f}",
                "X-Security-Score": str(guard["score"]),
                "X-Security-Status": guard["category"].upper(),
            },
        )

    except HTTPException as e:
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        if tenant_id != "unknown":
            metrics.record_request(
                tenant_id=tenant_id,
                latency_ms=total_latency_ms,
                cache_status="NONE",
                error=True,
            )
        raise

    except Exception as e:
        logger.error(f"Request failed: {e}")
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        if tenant_id != "unknown":
            metrics.record_request(
                tenant_id=tenant_id,
                latency_ms=total_latency_ms,
                cache_status="NONE",
                error=True,
            )
        raise HTTPException(status_code=500, detail=str(e))


def _extract_key(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization