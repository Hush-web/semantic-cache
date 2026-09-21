import os
import hashlib
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)
from fastembed import TextEmbedding
from loguru import logger
from typing import Optional, List


class FastEmbedFunction:
    """Local embedding function using fastembed."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]


class SemanticCache:
    """Semantic cache using Qdrant Cloud with exact-match fast path."""

    def __init__(
        self,
        collection_name: str = "llm_cache",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.collection_name = collection_name

        self.embedding_fn = FastEmbedFunction()
        self.vector_size = 384  # all-MiniLM-L6-v2 dimension

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        self._ensure_collection()
        logger.info(f"Qdrant cache ready (collection: {collection_name})")

    def _ensure_collection(self):
        """Create the collection and payload index if they do not exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")
        else:
            logger.info(f"Qdrant collection already exists: {self.collection_name}")

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tenant_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Payload index on 'tenant_id' is ready")
        except Exception as e:
            logger.info(f"Payload index already exists or could not be created: {e}")

    # ---------------------------------------------------------------
    # Exact-match helpers
    # ---------------------------------------------------------------

    def _exact_id(self, query: str, tenant_id: str) -> str:
        """Deterministic ID for an exact query + tenant pair."""
        raw = f"exact:{tenant_id}:{query}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))

    def check_exact(self, query: str, tenant_id: str) -> Optional[dict]:
        """Look up an exact-match entry by ID. Returns dict or None."""
        try:
            point_id = self._exact_id(query, tenant_id)
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )

            if not result:
                return None

            payload = result[0].payload or {}
            logger.info("EXACT CACHE HIT")
            return {
                "response": payload.get("response"),
                "distance": 0.0,
                "cached_query": payload.get("query"),
                "cache_type": "EXACT",
            }
        except Exception as e:
            logger.error(f"Exact cache lookup failed: {e}")
            return None

    # ---------------------------------------------------------------
    # Semantic-match helpers
    # ---------------------------------------------------------------

    def check(self, query: str, tenant_id: str) -> Optional[dict]:
        """Semantic match against cached entries."""
        try:
            query_vector = self.embedding_fn.embed([query])[0]

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id),
                        )
                    ]
                ),
                limit=1,
            ).points

            if not results:
                logger.info("CACHE MISS (no entries for tenant)")
                return None

            hit = results[0]
            distance = 1.0 - hit.score

            if distance <= self.threshold:
                logger.info(f"CACHE HIT (distance={distance:.3f})")
                return {
                    "response": hit.payload["response"],
                    "distance": distance,
                    "cached_query": hit.payload["query"],
                    "cache_type": "SEMANTIC",
                }

            logger.info(f"CACHE MISS (closest distance={distance:.3f})")
            return None

        except Exception as e:
            logger.error(f"Cache query failed: {e}")
            return None

    # ---------------------------------------------------------------
    # Store
    # ---------------------------------------------------------------

    def store(self, query: str, response: str, tenant_id: str):
        """Store an entry under both exact and semantic indexes."""
        try:
            vector = self.embedding_fn.embed([query])[0]

            # Exact-match point (deterministic ID)
            exact_point_id = self._exact_id(query, tenant_id)

            # Semantic point (random ID so duplicates don't collide)
            semantic_point_id = str(uuid.uuid4())

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=exact_point_id,
                        vector=vector,
                        payload={
                            "query": query,
                            "response": response,
                            "tenant_id": tenant_id,
                            "entry_type": "exact",
                        },
                    ),
                    PointStruct(
                        id=semantic_point_id,
                        vector=vector,
                        payload={
                            "query": query,
                            "response": response,
                            "tenant_id": tenant_id,
                            "entry_type": "semantic",
                        },
                    ),
                ],
            )
            logger.info(f"Stored in Qdrant (tenant={tenant_id})")

        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0