# Semantic Cache Gateway

An OpenAI-compatible API gateway that caches LLM responses **by meaning**, not exact text. Drop-in replacement for OpenAI or Groq clients. Cuts inference costs up to 70% on repeat-query workloads.

**Live:** https://semantic-cache-6qvc.onrender.com

---

## What it does

Point your existing OpenAI or Groq client at our URL. Bring your own API key. When two queries mean the same thing, the second one is served from cache — zero LLM cost, sub-50ms latency.

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://semantic-cache-6qvc.onrender.com/v1",
    api_key="YOUR_CACHE_GATEWAY_KEY",
    default_headers={"X-Upstream-Key": "YOUR_GROQ_OR_OPENAI_KEY"},
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "What is 2+2?"}],
)
