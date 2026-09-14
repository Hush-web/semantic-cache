import os
from typing import Dict, Any
from openai import AsyncOpenAI
from loguru import logger
from .base import BaseProvider

class OpenAIProvider(BaseProvider):
    """OpenAI provider adapter."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=request.get("model", "gpt-4o-mini"),
                messages=request.get("messages", []),
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 500),
            )
            return response.model_dump()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            raise