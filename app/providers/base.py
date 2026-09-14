from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    async def chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Accept an OpenAI-compatible request, return OpenAI-compatible response."""
        pass