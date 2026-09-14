from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
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

@app.get("/")
async def root():
    return {
        "service": "Semantic Cache Proxy",
        "version": "0.3.0",
        "cache_entries": cache.count(),
        "status": "running"
    }

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

@app.get("/admin/stats")
async def admin_stats(authorization: Optional[str] = Header(None)):
    api_key = _extract_key(authorization)
    if api_key != "admin_key_change_me":
        raise HTTPException(status_code=403, detail="Admin access required")
    return usage.get_all_stats()

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    try:
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
        logger.info(f"Request: model={body.get('model')} tenant={tenant_id}")

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
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                },
                headers={
                    "X-Cache": "HIT",
                    "X-Cache-Distance": f"{cached['distance']:.3f}",
                    "X-Tenant": tenant_id
                }
            )

        response = await router.route(body)
        usage.record_miss(tenant_id)

        try:
            assistant_message = response["choices"][0]["message"]["content"]
            cache.store(query, assistant_message, tenant_id)
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

        return JSONResponse(
            content=response,
            headers={"X-Cache": "MISS", "X-Tenant": tenant_id}
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