import os
from typing import Dict, Any, Optional
from groq import AsyncGroq
from loguru import logger
from .base import BaseProvider

class GroqProvider(BaseProvider):
    """Groq provider adapter with BYOK support."""

    async def chat_completion(
        self,
        request: Dict[str, Any],
        upstream_key: Optional[str] = None
    ) -> Dict[str, Any]:
        # Prefer customer's key. Fall back to server key only if allowed.
        api_key = upstream_key or os.getenv("GROQ_API_KEY")
        using_byok = upstream_key is not None

        if not api_key:
            raise ValueError("No Groq API key available")

        client = AsyncGroq(api_key=api_key)

        try:
            response = await client.chat.completions.create(
                model=request.get("model", "groq/compound-mini"),
                messages=request.get("messages", []),
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 500),
            )
            logger.info(f"Groq call succeeded (BYOK={using_byok})")
            return response.model_dump()
        except Exception as e:
            logger.error(f"Groq error (BYOK={using_byok}): {e}")
            raise