import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.main import app
import app.main as main_module


@pytest.fixture
def client():
    """TestClient with cache and router replaced by mocks."""
    # Cache: default is a MISS, store is no-op, count is 0
    main_module.cache = MagicMock()
    main_module.cache.check.return_value = None
    main_module.cache.store.return_value = None
    main_module.cache.count.return_value = 0

    # Router: return a fake OpenAI-style response
    main_module.router = MagicMock()

    async def fake_route(request, upstream_key=None):
        return {
            "id": "test-id",
            "object": "chat.completion",
            "model": request.get("model", "test"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "4"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    main_module.router.route = fake_route

    return TestClient(app)


def test_root_serves_landing_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Semantic Cache Gateway" in r.text


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_missing_auth_returns_401(client):
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_invalid_auth_returns_401(client):
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong_key"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 401


def test_missing_messages_returns_400(client):
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test_key"},
        json={},
    )
    assert r.status_code == 400


def test_cache_miss_without_upstream_key_returns_402(client):
    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test_key"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 402


def test_cache_miss_with_upstream_key_returns_200(client):
    r = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test_key",
            "X-Upstream-Key": "gsk_fake",
        },
        json={
            "model": "openai/gpt-oss-20b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
    )
    assert r.status_code == 200
    assert r.headers["x-cache"] == "MISS"
    assert r.headers["x-byok"] == "true"
    assert r.json()["choices"][0]["message"]["content"] == "4"


def test_cache_hit_returns_200_and_hit_header(client):
    # Force a cache hit
    main_module.cache.check.return_value = {
        "response": "4",
        "distance": 0.0,
        "cached_query": "What is 2+2?",
    }

    r = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test_key"},
        json={"messages": [{"role": "user", "content": "What is 2+2?"}]},
    )
    assert r.status_code == 200
    assert r.headers["x-cache"] == "HIT"
    assert r.json()["choices"][0]["message"]["content"] == "4"


def test_stats_endpoint(client):
    r = client.get(
        "/stats",
        headers={"Authorization": "Bearer test_key"},
    )
    assert r.status_code == 200