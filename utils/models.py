"""
utils/models.py
Global singleton model loaders for production RAG system.

Ensures models are loaded only once per process.
"""

from typing import Any
from sentence_transformers import SentenceTransformer, CrossEncoder

from config.settings import EMBEDDING_MODEL, CROSS_ENCODER_MODEL

# ── Global Model Instances ────────────────────────────────────
_embedding_model: SentenceTransformer | None = None
_reranker_model: CrossEncoder | None = None

# ── Embedding Model ───────────────────────────────────────────
def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Global singleton loader for embedding model.
    Loads only once, reuses across all modules.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[ModelLoader] Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        print("[ModelLoader] Embedding model ready!")
    return _embedding_model

# ── Reranker Model ────────────────────────────────────────────
def get_reranker(model_name: str = CROSS_ENCODER_MODEL) -> CrossEncoder:
    """
    Global singleton loader for reranker model.
    Loads only once, reuses across all modules.
    """
    global _reranker_model
    if _reranker_model is None:
        print(f"[ModelLoader] Loading reranker model: {model_name}")
        _reranker_model = CrossEncoder(model_name)
        print("[ModelLoader] Reranker model ready!")
    return _reranker_model

# ── Warmup Function (Optional) ────────────────────────────────
def warmup_models() -> None:
    """
    Pre-load both models and run dummy encoding to avoid first-call latency.
    Call this at application startup.
    """
    print("[ModelLoader] Warming up models...")
    emb = get_embedding_model()
    rerank = get_reranker()

    # Dummy operations to initialize
    emb.encode(["warmup text"])
    rerank.predict([("warmup query", "warmup text")])
    print("[ModelLoader] Models warmed up!")

# ── Custom Embedding Class for LangChain ─────────────────────
class SingletonEmbeddings:
    """
    LangChain-compatible embedding class that uses the global singleton model.
    Prevents double-loading in Chroma and other LangChain components.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = get_embedding_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    # For Chroma compatibility
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)