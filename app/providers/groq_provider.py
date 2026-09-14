import os
from typing import Dict, Any
from groq import AsyncGroq
from loguru import logger
from .base import BaseProvider

class GroqProvider(BaseProvider):
    """Groq provider adapter."""

    def __init__(self):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    async def chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=request.get("model", "groq/compound-mini"),
                messages=request.get("messages", []),
                temperature=request.get("temperature", 0.7),
                max_tokens=request.get("max_tokens", 500),
            )
            return response.model_dump()
        except Exception as e:
            logger.error(f"Groq error: {e}")
            raise
