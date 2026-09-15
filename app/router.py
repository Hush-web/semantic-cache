from typing import Optional
from loguru import logger
from .providers.groq_provider import GroqProvider
from .providers.openai_provider import OpenAIProvider
from .config import settings


class ProviderRouter:
    """Routes requests to the right provider based on model name."""

    def __init__(self):
        self.providers = {
            "groq": GroqProvider(),
            "openai": OpenAIProvider(),
        }

    def pick_provider(self, model: str) -> str:
        """Decide which provider to use based on the model name."""
        model_lower = model.lower()

        # Namespaced models (e.g., "groq/compound-mini", "openai/gpt-oss-20b",
        # "qwen/qwen3.6-27b") are Groq-hosted. OpenAI does not use namespaces.
        if "/" in model_lower:
            return "groq"

        # Plain OpenAI models
        if any(x in model_lower for x in ["gpt-", "o1-", "o3-", "text-embedding-"]):
            return "openai"

        # Plain Groq models (older naming, no namespace)
        if any(x in model_lower for x in ["llama", "mixtral", "gemma", "qwen"]):
            return "groq"

        logger.warning(
            f"Unknown model '{model}', using default: {settings.DEFAULT_PROVIDER}"
        )
        return settings.DEFAULT_PROVIDER

    async def route(
        self,
        request: dict,
        upstream_key: Optional[str] = None
    ) -> dict:
        """Route the request to the correct provider."""
        model = request.get("model", "")
        provider_name = self.pick_provider(model)
        provider = self.providers.get(provider_name)

        if not provider:
            raise ValueError(f"Provider '{provider_name}' not configured")

        logger.info(
            f"Routing model={model} -> provider={provider_name} "
            f"(BYOK={upstream_key is not None})"
        )
        return await provider.chat_completion(request, upstream_key=upstream_key)