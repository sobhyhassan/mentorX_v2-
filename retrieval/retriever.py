import logging
import re
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import StrOutputParser
        from langchain.schema import Document
    except ImportError:
        from langchain.prompts import ChatPromptTemplate
        from langchain.output_parsers import StrOutputParser
        from langchain.docstore.document import Document

from vectorStore.chroma_store import VectorStoreManager
from config.settings import (
    CACHE_MAX_SIZE,
    CACHE_SIMILARITY_THRESHOLD,
    ENABLE_HYDE,
    FINAL_TOP_K,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TOKENS,
    MIN_RELEVANCE_SCORE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K_RESULTS,
    HALLUCINATION_THRESHOLD,
)

from utils.models import SingletonEmbeddings, get_reranker

try:
    import tiktoken
except ImportError:
    tiktoken = None

# ── LLM Imports ───────────────────────────────────────────────
try:
    from langchain_groq import ChatGroq
    _HAS_GROQ = True
except ImportError:
    ChatGroq = None
    _HAS_GROQ = False

try:
    from langchain_openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    OpenAI = None
    _HAS_OPENAI = False




logger = logging.getLogger("mentor-x-ai.retriever")

# ── HyDE LLM Singleton ────────────────────────────────────────
# Built once on first use, reused for all subsequent query rewrites.
_hyde_llm = None

def _get_hyde_llm():
    """Return a cached OpenAI client for HyDE query rewriting."""
    global _hyde_llm
    if _hyde_llm is None:
        if not _HAS_OPENAI or not OPENAI_API_KEY:
            return None
        _hyde_llm = OpenAI(
            model_name=OPENAI_MODEL,
            temperature=0.0,
            openai_api_key=OPENAI_API_KEY,
        )
        logger.info("HyDE LLM client initialized (singleton): %s", OPENAI_MODEL)
    return _hyde_llm


# ── Constants ─────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "I couldn't find relevant information in the document "
    "to answer this question."
)


# ── Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Mentor-X, an expert AI assistant that answers questions \
based strictly on the provided context from academic documents.

Rules:
- Answer ONLY from the context below.
- Do not infer any information that is not explicitly stated in the context.
- If the answer is not in the context, say exactly:
  "I couldn't find relevant information in the document to answer this question."
- Avoid hallucination completely.
- Combine information carefully when multiple chunks are relevant.
- Be concise and precise.
- Cite page numbers when possible.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query or "").strip()


def _normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _reject_empty_query(query: str) -> bool:
    return not bool(query)


def _query_rewrite(query: str) -> str:
    """Rewrite the query for better retrieval using HyDE-style expansion."""
    if not ENABLE_HYDE or not _HAS_OPENAI or not OPENAI_API_KEY:
        return query

    rewritten = query
    try:
        prompt = (
            "Rewrite the following question so it is more explicit and retrieval-friendly. "
            "Do not add new facts and keep the meaning unchanged.\n\n"
            f"Question: {query}\n"
            "Rewritten question:"
        )

        # Reuse the singleton client — no new instance per call
        llm = _get_hyde_llm()
        if llm is None:
            return query

        response = llm.generate([prompt])
        rewritten = response.generations[0][0].text.strip()
        if rewritten:
            logger.debug("HyDE query rewrite: %s -> %s", query, rewritten)
        else:
            rewritten = query
    except Exception:
        logger.debug("HyDE rewrite failed, using original query.")
        rewritten = query

    return rewritten


def _semantic_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    if np.linalg.norm(a_arr) == 0 or np.linalg.norm(b_arr) == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _detect_hallucination(answer: str, context: str, threshold: float = HALLUCINATION_THRESHOLD) -> dict:
    if not answer or not context:
        return {
            "faithfulness_score": 0.0,
            "hallucination_flag": True,
            "hallucination_threshold": threshold,
        }

    answer_tokens = _token_set(answer)
    context_tokens = _token_set(context)
    if not answer_tokens:
        return {
            "faithfulness_score": 0.0,
            "hallucination_flag": True,
            "hallucination_threshold": threshold,
        }

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    return {
        "faithfulness_score": round(overlap, 4),
        "hallucination_flag": overlap < threshold,
        "hallucination_threshold": threshold,
    }


def _prepare_cache_key(query: str) -> str:
    return _normalize_query(query).lower()


class SemanticCache:
    def __init__(self, max_size: int = CACHE_MAX_SIZE, similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.cache: OrderedDict[str, tuple[list[tuple[Document, float]], list[float]]] = OrderedDict()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def get(self, key: str, query_embedding: list[float]) -> Optional[list[tuple[Document, float]]]:
        for existing_key, (results, embedding) in self.cache.items():
            similarity = _semantic_similarity(query_embedding, embedding)
            if similarity >= self.similarity_threshold:
                self.cache.move_to_end(existing_key)
                self.hits += 1
                logger.info("Semantic cache hit (sim=%.4f) for key=%s", similarity, existing_key)
                return results
        self.misses += 1
        logger.info("Semantic cache miss for key=%s", key)
        return None

    def set(self, key: str, query_embedding: list[float], results: list[tuple[Document, float]]) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (results, query_embedding)
        if len(self.cache) > self.max_size:
            removed = self.cache.popitem(last=False)
            logger.debug("Semantic cache evicted oldest entry: %s", removed[0])


# ── Reranking ─────────────────────────────────────────────────
def _rerank(
    docs_with_scores: List[Tuple[Document, float]],
    query: str,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> List[Tuple[Document, float]]:
    """
    Rerank retrieved chunks using cross-encoder.
    """

    if not docs_with_scores:
        return []


    filtered = docs_with_scores
    cos_scores = [float(score) for _, score in filtered]

    # ── Cross Encoder ─────────────────────────────
    cross_encoder = get_reranker()
    try:
        pairs = [
            (query, doc.page_content)
            for doc, _ in filtered
        ]
        cross_scores = [float(score) for score in cross_encoder.predict(pairs)]
    except Exception:
        logger.exception("Reranker prediction failed, falling back to semantic scores.")
        return filtered

    def normalize(values: List[float]) -> List[float]:
        min_val, max_val = min(values), max(values)
        if max_val <= min_val:
            return [0.5 for _ in values]
        return [(value - min_val) / (max_val - min_val) for value in values]

    norm_cross = normalize(cross_scores)
    norm_cos = normalize(cos_scores)

    reranked = []

    # ── Hybrid Scoring ────────────────────────────
    for (doc, _), cross_score, cos_score in zip(filtered, norm_cross, norm_cos):
        combined_score = 0.6 * cross_score + 0.4 * cos_score
        reranked.append((doc, combined_score))
        logger.debug(
            "Chunk score debug | source=%s page=%s semantic=%.6f reranker=%.6f hybrid=%.6f",
            doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
            doc.metadata.get("page_number", "?"),
            cos_score,
            cross_score,
            combined_score,
        )

    # ── Sort by relevance ─────────────────────────
    reranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ── Final lightweight filtering ───────────────
    reranked = [
        (doc, score)
        for doc, score in reranked
        if score >= min_score
    ]

    logger.info(
        "Reranking: %d/%d chunks survived.",
        len(reranked),
        len(docs_with_scores)
    )

    return reranked


# ── Context Builder ───────────────────────────────────────────
def _count_tokens(text: str, model: str) -> int:
    if tiktoken is None:
        return max(1, len(text) // 4)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def _build_context(
    docs: List[Document],
    max_chars: int = MAX_CONTEXT_CHARS,
    max_tokens: int = MAX_CONTEXT_TOKENS,
) -> str:
    """
    Build final context safely.
    """

    if not docs:
        return "No relevant context found."

    docs = docs[:FINAL_TOP_K]

    context_parts = []
    seen_chunks = set()
    current_tokens = 0

    for doc in docs:
        page = doc.metadata.get("page_number", "?")
        chunk_text = f"[Page {page}]\n{doc.page_content}"
        normalized_chunk = _normalize_chunk_text(chunk_text)

        if normalized_chunk in seen_chunks:
            logger.debug(
                "Deduplicated chunk from source=%s page=%s",
                doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
                page,
            )
            continue

        seen_chunks.add(normalized_chunk)
        chunk_tokens = _count_tokens(chunk_text, LLM_MODEL)

        if current_tokens + chunk_tokens > max_tokens:
            if max_tokens - current_tokens > 50:
                truncated_chunk = chunk_text[: max(0, (max_tokens - current_tokens) * 4)]
                context_parts.append(truncated_chunk + "...")
            logger.warning(
                "Context truncated at %d tokens.",
                max_tokens
            )
            break

        context_parts.append(chunk_text)
        current_tokens += chunk_tokens

    context = "\n\n======== DOCUMENT CHUNK ========\n\n".join(context_parts)

    if len(context) > max_chars:
        context = context[:max_chars] + "..."
        logger.warning("Context additionally truncated by chars at %d.", max_chars)

    return context


# ── Hybrid Retrieval (BM25 + Vector) ──────────────────────────
class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval.
    Better for keyword-heavy queries and proper nouns.
    """

    def __init__(self, store: "VectorStoreManager", alpha: float = 0.7):
        """
        alpha: weight for dense retrieval (1-alpha for BM25)
        """
        from rank_bm25 import BM25Okapi
        self._BM25Okapi = BM25Okapi
        self.store = store
        self.alpha = alpha
        self._bm25 = None
        self._bm25_docs: list = []

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in the vector store."""
        try:
            # Get all docs from chroma
            results = self.store.db.get(include=["documents", "metadatas"])
            texts = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not texts:
                logger.warning("No documents found for BM25 index")
                return

            self._bm25_docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(texts, metadatas)
            ]
            tokenized = [doc.page_content.lower().split() for doc in self._bm25_docs]
            self._bm25 = self._BM25Okapi(tokenized)
            logger.info("BM25 index built with %d documents", len(self._bm25_docs))

        except Exception:
            logger.exception("Failed to build BM25 index, falling back to dense only")

    def search(self, query: str, k: int = 8) -> list:
        """Hybrid search: dense + BM25 with score fusion."""

        # Dense retrieval
        dense_results = self.store.search_with_score(query, k=k)

        # Build BM25 index lazily
        if self._bm25 is None:
            self._build_bm25_index()

        if self._bm25 is None or not self._bm25_docs:
            # Fallback to dense only
            return dense_results

        # BM25 retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        top_bm25_idx = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:k]

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_results = [
            (self._bm25_docs[i], float(bm25_scores[i]) / max_bm25)
            for i in top_bm25_idx
        ]

        # Merge with Reciprocal Rank Fusion (RRF)
        scores: dict = {}
        doc_map: dict = {}

        for rank, (doc, _) in enumerate(dense_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + self.alpha * (1 / (rank + 1))
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(bm25_results):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + (1 - self.alpha) * (1 / (rank + 1))
            doc_map[key] = doc

        # Sort by fused score
        sorted_keys = sorted(
            scores,
            key=lambda k: scores[k],
            reverse=True
        )[:k]

        return [(doc_map[key], scores[key]) for key in sorted_keys]


# ── Main Retriever ────────────────────────────────────────────
class MentorRetriever:

    def __init__(
        self,
        store: Optional[VectorStoreManager] = None
    ):

        logger.info("Connecting to vector store...")

        self.store = store or VectorStoreManager()
        self.hybrid_retriever = HybridRetriever(self.store)

        # ── LLM ───────────────────────────────────
        if _HAS_GROQ:
            if not GROQ_API_KEY:
                raise RuntimeError(
                    "Groq backend available but GROQ_API_KEY is missing."
                )

            logger.info(
                "Loading Groq LLM: %s",
                LLM_MODEL
            )

            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
            )

        elif _HAS_OPENAI and OPENAI_API_KEY:

            logger.info(
                "Loading OpenAI LLM: %s",
                OPENAI_MODEL
            )

            self.llm = OpenAI(
                model_name=OPENAI_MODEL,
                temperature=LLM_TEMPERATURE,
                openai_api_key=OPENAI_API_KEY,
            )
        elif _HAS_OPENAI and not OPENAI_API_KEY:
            raise RuntimeError(
                "OpenAI backend available but OPENAI_API_KEY is missing."
            )

        else:
            raise RuntimeError(
                "No supported LLM backend installed."
            )

        # ── Output Parser ─────────────────────────
        self.parser = StrOutputParser()
        self.embeddings = SingletonEmbeddings()
        self.semantic_cache = SemanticCache()

        # 🔥 Build chain ONCE
        self.chain = PROMPT | self.llm | self.parser

        logger.info("RAG chain ready!")

    # ─────────────────────────────────────────────
    def ask(
        self,
        question: str,
        k: int = TOP_K_RESULTS
    ) -> dict:

        start_time = time.perf_counter()
        question = _normalize_query(question)

        if _reject_empty_query(question):

            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "context": ""
            }

        logger.info(
            "Question: %s",
            question[:80]
        )

        # ── Step 1: Query rewrite + semantic cache ─────
        rewritten_query = _query_rewrite(question)
        query_embedding = self.embeddings.embed_query(rewritten_query)
        cache_key = _prepare_cache_key(rewritten_query)
        cached_results = self.semantic_cache.get(cache_key, query_embedding)

        retrieval_start = time.perf_counter()
        if cached_results is not None:
            docs_with_scores = cached_results
            logger.info("Using semantic cache for query.")
        else:
            docs_with_scores = self.hybrid_retriever.search(
                rewritten_query,
                k=k
            )
            self.semantic_cache.set(cache_key, query_embedding, docs_with_scores)
        retrieval_latency = time.perf_counter() - retrieval_start
        logger.info("Retrieval latency: %.3fs", retrieval_latency)

        # ── Step 2: Reranking ─────────────────────
        rerank_start = time.perf_counter()
        ranked_results = _rerank(
            docs_with_scores,
            rewritten_query
        )
        reranking_latency = time.perf_counter() - rerank_start
        logger.info("Reranking latency: %.3fs", reranking_latency)

        # ── Step 3: Empty Handling ────────────────
        if not ranked_results:

            logger.warning(
                "No relevant chunks found."
            )

            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "context": ""
            }

        # ── Step 4: Build Context ─────────────────
        ranked_docs = [doc for doc, _ in ranked_results]

        context = _build_context(ranked_docs)

        # ── Step 5: Invoke LLM ────────────────────
        llm_start = time.perf_counter()
        try:
            answer = self.chain.invoke({
                "context": context,
                "question": question,
            })
        except Exception:
            logger.exception("LLM invocation failed")
            return {
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "context": context,
            }
        llm_latency = time.perf_counter() - llm_start
        total_latency = time.perf_counter() - start_time
        logger.info("LLM generation latency: %.3fs", llm_latency)
        logger.info("Total pipeline latency: %.3fs", total_latency)

        hallucination = _detect_hallucination(answer, context)
        if hallucination["hallucination_flag"]:
            logger.warning(
                "Potential hallucination detected: faithfulness=%.4f threshold=%.4f",
                hallucination["faithfulness_score"],
                hallucination["hallucination_threshold"],
            )

        # ── Step 6: Sources ───────────────────────
        sources = [
            {
                "source": doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
                "page": doc.metadata.get("page_number", "?"),
                "score": round(score, 4),
                "preview": doc.page_content[:150] + "...",
            }
            for doc, score in ranked_results[:FINAL_TOP_K]
        ]

        logger.info(
            "Done. Sources used: %d",
            len(sources)
        )

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "hallucination": hallucination,
        }