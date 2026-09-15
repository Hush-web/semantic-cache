from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseProvider(ABC):
    """Base class for all LLM providers."""

    @abstractmethod
    async def chat_completion(
        self,
        request: Dict[str, Any],
        upstream_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Accept an OpenAI-compatible request, return OpenAI-compatible response."""
        pass