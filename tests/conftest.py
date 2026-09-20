import os
from unittest.mock import MagicMock, patch

# Set fake env vars BEFORE importing app
os.environ["GROQ_API_KEY"] = "test_groq_key"
os.environ["OPENAI_API_KEY"] = "test_openai_key"
os.environ["QDRANT_URL"] = "http://test-qdrant:6333"
os.environ["QDRANT_API_KEY"] = "test_qdrant_key"
os.environ["API_KEYS"] = "test_key:test_tenant,demo_key:demo_tenant"
os.environ["DEFAULT_PROVIDER"] = "groq"

# Patch external clients so importing app does not connect anywhere
_qdrant_patch = patch("qdrant_client.QdrantClient", MagicMock())
_fastembed_patch = patch("fastembed.TextEmbedding", MagicMock())
_qdrant_patch.start()
_fastembed_patch.start()


def pytest_sessionfinish(session, exitstatus):
    _qdrant_patch.stop()
    _fastembed_patch.stop()