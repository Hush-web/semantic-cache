import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from fastembed import TextEmbedding
from loguru import logger
from typing import Optional
import hashlib


class FastEmbedFunction(EmbeddingFunction):
    """Custom embedding function compatible with ChromaDB."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = list(self.model.embed(input))
        return [emb.tolist() for emb in embeddings]

    @staticmethod
    def name() -> str:
        return "fastembed-all-MiniLM-L6-v2"


class SemanticCache:
    """Semantic cache using ChromaDB + fastembed with cosine distance."""

    def __init__(self, collection_name: str = "llm_cache", threshold: float = 0.5):
        self.threshold = threshold
        self.collection_name = collection_name

        self.embedding_fn = FastEmbedFunction()

        self.client = chromadb.PersistentClient(path="./data/chromadb")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Cache ready: {self.collection.count()} entries")

    def check(self, query: str, tenant_id: str) -> Optional[dict]:
        """Return cached response if a similar query exists."""
        if self.collection.count() == 0:
            return None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=1,
                where={"tenant_id": tenant_id}
            )
        except Exception as e:
            logger.error(f"Cache query failed: {e}")
            return None

        if not results["ids"] or not results["ids"][0]:
            return None

        distance = results["distances"][0][0]
        if distance <= self.threshold:
            logger.info(f"CACHE HIT (distance={distance:.3f})")
            return {
                "response": results["metadatas"][0][0]["response"],
                "distance": distance,
                "cached_query": results["documents"][0][0]
            }

        logger.info(f"CACHE MISS (closest distance={distance:.3f})")
        return None

    def store(self, query: str, response: str, tenant_id: str):
        """Store a query and its response."""
        entry_id = hashlib.md5(f"{tenant_id}:{query}".encode()).hexdigest()

        try:
            self.collection.add(
                ids=[entry_id],
                documents=[query],
                metadatas=[{
                    "response": response,
                    "tenant_id": tenant_id,
                }]
            )
            logger.info(f"Stored in cache (tenant={tenant_id}, total={self.collection.count()})")
        except Exception as e:
            logger.error(f"Cache store failed: {e}")

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0