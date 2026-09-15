import os
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
import hashlib
import uuid


class FastEmbedFunction:
    """Local embedding function using fastembed."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]


class SemanticCache:
    """Semantic cache using Qdrant Cloud + fastembed."""

    def __init__(self, collection_name: str = "llm_cache", threshold: float = 0.5):
        self.threshold = threshold
        self.collection_name = collection_name

        self.embedding_fn = FastEmbedFunction()
        self.vector_size = 384

        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set")

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

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
                    distance=Distance.COSINE
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

    def check(self, query: str, tenant_id: str) -> Optional[dict]:
        """Return cached response if a similar query exists."""
        try:
            query_vector = self.embedding_fn.embed([query])[0]

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="tenant_id",
                            match=MatchValue(value=tenant_id)
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
                    "cached_query": hit.payload["query"]
                }

            logger.info(f"CACHE MISS (closest distance={distance:.3f})")
            return None

        except Exception as e:
            logger.error(f"Cache query failed: {e}")
            return None

    def store(self, query: str, response: str, tenant_id: str):
        """Store a query and its response."""
        try:
            entry_id = hashlib.md5(f"{tenant_id}:{query}".encode()).hexdigest()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry_id))

            vector = self.embedding_fn.embed([query])[0]

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "query": query,
                            "response": response,
                            "tenant_id": tenant_id,
                        }
                    )
                ]
            )
            logger.info(f"Stored in Qdrant (tenant={tenant_id})")

        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    def count(self) -> int:
        """Return total entries in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0