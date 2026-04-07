from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from core.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, chroma_path: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection("jarvis_memory")
        self._model: SentenceTransformer | None = None
        self._model_name = model_name
        self._loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading sentence-transformers: {self._model_name} on {device}")
        self._model = SentenceTransformer(self._model_name, device=device)
        self._loaded = True

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self._load_model()
        embedding = self._model.encode(text).tolist()
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def search(self, query: str, top_k: int = 5) -> list[str]:
        self._load_model()
        count = self._collection.count()
        if count == 0:
            return []
        embedding = self._model.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
        )
        return results["documents"][0] if results["documents"] else []
