from app.metrics import MetricsTracker


def test_empty_metrics():
    m = MetricsTracker()
    snap = m.snapshot()
    assert snap["total_requests"] == 0
    assert snap["hit_rate_percent"] == 0.0


def test_record_hit_and_miss():
    m = MetricsTracker()
    m.record_request("t1", latency_ms=50, cache_status="HIT", cache_latency_ms=10)
    m.record_request("t1", latency_ms=800, cache_status="MISS", llm_latency_ms=700)

    snap = m.snapshot()
    assert snap["total_requests"] == 2
    assert snap["cache_hits"] == 1
    assert snap["cache_misses"] == 1
    assert snap["hit_rate_percent"] == 50.0
    assert snap["per_tenant"]["t1"]["requests"] == 2
    assert snap["per_tenant"]["t1"]["cache_hits"] == 1


def test_percentiles():
    m = MetricsTracker()
    for i in range(1, 101):
        m.record_request("t1", latency_ms=i, cache_status="HIT", cache_latency_ms=i)
    snap = m.snapshot()
    assert snap["latency_ms"]["p50"] == 50
    assert snap["latency_ms"]["p95"] == 95
    assert snap["latency_ms"]["p99"] == 99