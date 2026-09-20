import tempfile
from app.usage import UsageTracker


def test_new_tenant_has_zero_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = UsageTracker(storage_path=f"{tmpdir}/usage.json")
        stats = tracker.get_stats("brand_new")
        assert stats == {"error": "no usage recorded"}


def test_record_hit_increments_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = UsageTracker(storage_path=f"{tmpdir}/usage.json")
        tracker.record_hit("tenant_a")
        stats = tracker.get_stats("tenant_a")
        assert stats["total_requests"] == 1
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 0
        assert stats["hit_rate_percent"] == 100.0
        assert stats["tokens_saved"] > 0


def test_record_miss_increments_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = UsageTracker(storage_path=f"{tmpdir}/usage.json")
        tracker.record_miss("tenant_a")
        stats = tracker.get_stats("tenant_a")
        assert stats["total_requests"] == 1
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 1
        assert stats["hit_rate_percent"] == 0.0


def test_hit_rate_calculated_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = UsageTracker(storage_path=f"{tmpdir}/usage.json")
        tracker.record_hit("t")
        tracker.record_hit("t")
        tracker.record_miss("t")
        tracker.record_miss("t")
        stats = tracker.get_stats("t")
        assert stats["total_requests"] == 4
        assert stats["hit_rate_percent"] == 50.0