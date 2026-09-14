import os
from loguru import logger
from typing import Optional

class APIKeyAuth:
    """Simple API key authentication with per-tenant keys."""

    def __init__(self):
        keys_str = os.getenv("API_KEYS", "dev_key:default")
        self.keys = {}
        for pair in keys_str.split(","):
            if ":" in pair:
                key, tenant = pair.strip().split(":", 1)
                self.keys[key] = tenant
        logger.info(f"Loaded {len(self.keys)} API keys")

    def validate(self, api_key: Optional[str]) -> Optional[str]:
        if not api_key:
            return None
        return self.keys.get(api_key)

    def is_valid(self, api_key: Optional[str]) -> bool:
        return api_key in self.keys