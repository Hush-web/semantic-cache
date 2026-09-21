from unittest.mock import MagicMock
from app.cache import SemanticCache


def _make_cache():
    """Build a cache with Qdrant and embeddings mocked out."""
    from unittest.mock import patch

    with patch("app.cache.QdrantClient", MagicMock()), \
         patch("app.cache.TextEmbedding", MagicMock()):
        cache = SemanticCache.__new__(SemanticCache)
        cache.threshold = 0.5
        cache.collection_name = "test_cache"
        cache.vector_size = 384
        cache.embedding_fn = MagicMock()
        cache.embedding_fn.embed.return_value = [[0.0] * 384]
        cache.client = MagicMock()
        return cache


def test_exact_hit_returns_response():
    cache = _make_cache()
    cache.client.retrieve.return_value = [
        MagicMock(payload={"response": "4", "query": "What is 2+2?"})
    ]

    result = cache.check_exact("What is 2+2?", "tenant_a")
    assert result is not None
    assert result["response"] == "4"
    assert result["cache_type"] == "EXACT"


def test_exact_miss_returns_none():
    cache = _make_cache()
    cache.client.retrieve.return_value = []

    result = cache.check_exact("something never asked", "tenant_a")
    assert result is None