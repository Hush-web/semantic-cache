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
        model_lower = model.lower()

        if any(x in model_lower for x in ["llama", "mixtral", "gemma", "qwen"]):
            return "groq"
        if any(x in model_lower for x in ["gpt", "o1", "o3"]):
            return "openai"

        logger.warning(f"Unknown model '{model}', using default: {settings.DEFAULT_PROVIDER}")
        return settings.DEFAULT_PROVIDER

    async def route(self, request: dict) -> dict:
        model = request.get("model", "")
        provider_name = self.pick_provider(model)
        provider = self.providers.get(provider_name)

        if not provider:
            raise ValueError(f"Provider '{provider_name}' not configured")

        logger.info(f"Routing model={model} -> provider={provider_name}")
        return await provider.chat_completion(request)
