import os
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from loguru import logger
from .base import BaseProvider

class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter with BYOK support."""

    async def chat_completion(
        self,
        request: Dict[str, Any],
        upstream_key: Optional[str] = None
    ) -> Dict[str, Any]:
        api_key = upstream_key or os.getenv("OPENAI_API_KEY")
        using_byok = upstream_key is not None

        if not api_key:
            raise ValueError("No OpenAI API key available")

        client = AsyncOpenAI(api_key=api_key)

        try:
            response = await client.chat.completions.create(
                model=request.get("model", "gpt-4o-mini"),
                messages=request.get("messages", []),
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 500),
            )
            logger.info(f"OpenAI call succeeded (BYOK={using_byok})")
            return response.model_dump()
        except Exception as e:
            logger.error(f"OpenAI error (BYOK={using_byok}): {e}")
            raise