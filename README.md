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
That is the entire integration. One URL change.

Why BYOK (Bring Your Own Key)
You stay billed directly with OpenAI or Groq — no markup, no reselling tokens

Your key never leaves your server

Remove us tomorrow by changing one line back

Your existing terms, quotas, and rate limits still apply

Features
Semantic matching — queries compared by meaning using sentence embeddings (all-MiniLM-L6-v2 via fastembed)

Multi-provider routing — Groq and OpenAI, auto-detected from model name

Persistent storage — powered by Qdrant Cloud; caches survive restarts and deploys

Multi-tenant — every API key has its own isolated cache namespace

Usage tracking — per-tenant request counts, hit rates, and estimated savings

OpenAI-compatible — same request and response shape as the OpenAI SDK

Stateless compute — no data on the gateway; all state lives in Qdrant

How it works
text
Client → Gateway → check Qdrant for similar past query
                       │
              ┌────────┴────────┐
              │                 │
            HIT               MISS
              │                 │
       return cached      call provider
       (<50ms, $0)        with YOUR key
              │                 │
              └────────┬────────┘
                       │
                  store in Qdrant
Cache key: the full user message. For RAG systems, include the retrieved context — do not cache on the question alone.

Threshold: cosine distance ≤ 0.5 counts as a hit

Response headers: X-Cache: HIT|MISS, X-Cache-Distance, X-Tenant, X-BYOK

API
POST /v1/chat/completions
OpenAI-compatible chat completions. Requires two headers:

Header	Value
Authorization	Bearer YOUR_CACHE_GATEWAY_KEY
X-Upstream-Key	Your OpenAI or Groq API key
GET /stats
Returns usage for the authenticated tenant.

bash
curl -H "Authorization: Bearer YOUR_KEY" \
  https://semantic-cache-6qvc.onrender.com/stats
json
{
  "tenant_id": "your_tenant",
  "total_requests": 1240,
  "cache_hits": 892,
  "cache_misses": 348,
  "hit_rate_percent": 71.94,
  "tokens_saved": 446000,
  "cost_saved_usd": 0.446
}
GET /health
json
{"status": "healthy", "cache_entries": 1547}
GET /docs
Interactive OpenAPI documentation.

Try it in 30 seconds
Use the public demo key. Bring your own Groq key.

bash
curl -X POST https://semantic-cache-6qvc.onrender.com/v1/chat/completions \
  -H "Authorization: Bearer demo_key" \
  -H "X-Upstream-Key: YOUR_GROQ_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
Send it twice. First response: X-Cache: MISS. Second: X-Cache: HIT.

Tech stack
Layer	Tool
API	FastAPI + Uvicorn
Vector store	Qdrant Cloud
Embeddings	fastembed (all-MiniLM-L6-v2, 384-dim)
LLM providers	Groq, OpenAI
Deployment	Render
Language	Python 3.12
Local development
bash
git clone https://github.com/Hush-web/semantic-cache
cd semantic-cache
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Create .env:

env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
DEFAULT_PROVIDER=groq
API_KEYS=dev_key:test_tenant,demo_key:demo_tenant
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_key
Run:

bash
python run.py
Open http://localhost:8000
