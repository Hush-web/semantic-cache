from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

from .router import ProviderRouter
from .cache import SemanticCache

app = FastAPI(title="Semantic Cache Proxy")
router = ProviderRouter()
cache = SemanticCache(threshold=0.35)

@app.get("/")
async def root():
    return {
        "service": "Semantic Cache Proxy",
        "version": "0.2.0",
        "cache_entries": cache.count(),
        "status": "running"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "cache_entries": cache.count()}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with semantic cache."""
    try:
        body = await request.json()

        if "messages" not in body:
            raise HTTPException(status_code=400, detail="Missing 'messages' field")

        # Extract the last user message as the cache key
        messages = body["messages"]
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        query = user_messages[-1]["content"]
        tenant_id = request.headers.get("X-Tenant-Id", "default")

        logger.info(f"Request: model={body.get('model')} tenant={tenant_id}")

        # Check cache
        cached = cache.check(query, tenant_id)
        if cached:
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
                    "X-Cache-Distance": f"{cached['distance']:.3f}"
                }
            )

        # Cache miss: route to provider
        response = await router.route(body)

        # Store in cache
        try:
            assistant_message = response["choices"][0]["message"]["content"]
            cache.store(query, assistant_message, tenant_id)
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

        return JSONResponse(
            content=response,
            headers={"X-Cache": "MISS"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
