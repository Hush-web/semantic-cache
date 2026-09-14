from loguru import logger
from typing import Dict, Any
from datetime import datetime
import json
import os

class UsageTracker:
    """Track per-tenant usage and savings."""

    def __init__(self, storage_path: str = "./data/usage.json"):
        self.storage_path = storage_path
        self.usage: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r") as f:
                    self.usage = json.load(f)
                logger.info(f"Loaded usage for {len(self.usage)} tenants")
        except Exception as e:
            logger.error(f"Usage load failed: {e}")
            self.usage = {}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(self.usage, f, indent=2)
        except Exception as e:
            logger.error(f"Usage save failed: {e}")

    def _get_tenant(self, tenant_id: str) -> Dict[str, Any]:
        if tenant_id not in self.usage:
            self.usage[tenant_id] = {
                "total_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "tokens_saved": 0,
                "cost_saved_usd": 0.0,
                "first_seen": datetime.utcnow().isoformat(),
                "last_seen": datetime.utcnow().isoformat()
            }
        return self.usage[tenant_id]

    def record_hit(self, tenant_id: str, tokens_saved: int = 500, cost_per_1k: float = 0.001):
        t = self._get_tenant(tenant_id)
        t["total_requests"] += 1
        t["cache_hits"] += 1
        t["tokens_saved"] += tokens_saved
        t["cost_saved_usd"] += (tokens_saved / 1000) * cost_per_1k
        t["last_seen"] = datetime.utcnow().isoformat()
        self.save()

    def record_miss(self, tenant_id: str):
        t = self._get_tenant(tenant_id)
        t["total_requests"] += 1
        t["cache_misses"] += 1
        t["last_seen"] = datetime.utcnow().isoformat()
        self.save()

    def get_stats(self, tenant_id: str) -> Dict[str, Any]:
        if tenant_id not in self.usage:
            return {"error": "no usage recorded"}

        t = self.usage[tenant_id]
        total = t["total_requests"]
        hit_rate = (t["cache_hits"] / total * 100) if total > 0 else 0

        return {
            "tenant_id": tenant_id,
            "total_requests": total,
            "cache_hits": t["cache_hits"],
            "cache_misses": t["cache_misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "tokens_saved": t["tokens_saved"],
            "cost_saved_usd": round(t["cost_saved_usd"], 4),
            "first_seen": t["first_seen"],
            "last_seen": t["last_seen"]
        }

    def get_all_stats(self) -> Dict[str, Any]:
        return {tid: self.get_stats(tid) for tid in self.usage}