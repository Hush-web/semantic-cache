from collections import defaultdict, deque
from typing import Dict, Any
from datetime import datetime, timezone
from loguru import logger


class MetricsTracker:
    """In-memory metrics for the gateway."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.started_at = datetime.now(timezone.utc).isoformat()

        # Counters
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.blocked = 0

        # Latencies (keep last N for percentile calculation)
        self.latencies_ms = deque(maxlen=window_size)
        self.cache_latencies_ms = deque(maxlen=window_size)
        self.llm_latencies_ms = deque(maxlen=window_size)

        # Per-tenant counters
        self.per_tenant = defaultdict(lambda: {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
        })

    def record_request(
        self,
        tenant_id: str,
        latency_ms: float,
        cache_status: str,
        cache_latency_ms: float = 0.0,
        llm_latency_ms: float = 0.0,
        error: bool = False,
    ):
        """Record a single request."""
        self.total_requests += 1
        self.latencies_ms.append(latency_ms)

        t = self.per_tenant[tenant_id]
        t["requests"] += 1

        if error:
            self.errors += 1
            t["errors"] += 1

        if cache_status == "HIT":
            self.cache_hits += 1
            t["cache_hits"] += 1
            self.cache_latencies_ms.append(cache_latency_ms)
        elif cache_status == "MISS":
            self.cache_misses += 1
            t["cache_misses"] += 1
            if llm_latency_ms:
                self.llm_latencies_ms.append(llm_latency_ms)

    def record_blocked(self):
        self.blocked += 1

    @staticmethod
    def _percentile(data: deque, p: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        # Nearest-rank method: p-th percentile of n values is the
        # value at index int(p * (n - 1))
        idx = int(p * (len(sorted_data) - 1))
        return round(sorted_data[idx], 2)
    
    def snapshot(self) -> Dict[str, Any]:
        """Return current metrics."""
        hit_rate = (
            (self.cache_hits / self.total_requests * 100)
            if self.total_requests > 0
            else 0.0
        )

        return {
            "started_at": self.started_at,
            "uptime_requests": self.total_requests,
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "errors": self.errors,
            "blocked_requests": self.blocked,
            "latency_ms": {
                "p50": self._percentile(self.latencies_ms, 0.50),
                "p95": self._percentile(self.latencies_ms, 0.95),
                "p99": self._percentile(self.latencies_ms, 0.99),
            },
            "cache_latency_ms": {
                "p50": self._percentile(self.cache_latencies_ms, 0.50),
                "p95": self._percentile(self.cache_latencies_ms, 0.95),
            },
            "llm_latency_ms": {
                "p50": self._percentile(self.llm_latencies_ms, 0.50),
                "p95": self._percentile(self.llm_latencies_ms, 0.95),
            },
            "per_tenant": dict(self.per_tenant),
        }


# Global instance
metrics = MetricsTracker()