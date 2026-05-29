import logging
import re
import time
from typing import List, Optional, Tuple

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
    GROQ_API_KEY,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    TOP_K_RESULTS,
    MIN_RELEVANCE_SCORE,
    FINAL_TOP_K,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_TOKENS,
)

from utils.models import get_reranker

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


# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger("MentorRetriever")


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


# ── Main Retriever ────────────────────────────────────────────
class MentorRetriever:

    def __init__(
        self,
        store: Optional[VectorStoreManager] = None
    ):

        logger.info("Connecting to vector store...")

        self.store = store or VectorStoreManager()

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

        # ── Step 1: Retrieval ─────────────────────
        retrieval_start = time.perf_counter()
        docs_with_scores = self.store.search_with_score(
            question,
            k=k
        )
        retrieval_latency = time.perf_counter() - retrieval_start
        logger.info("Retrieval latency: %.3fs", retrieval_latency)

        # ── Step 2: Reranking ─────────────────────
        rerank_start = time.perf_counter()
        ranked_results = _rerank(
            docs_with_scores,
            question
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
            "context": context
        }