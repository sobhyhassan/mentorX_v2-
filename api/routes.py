import asyncio
import json
import logging
import re
import time as _time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, constr
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
logger = logging.getLogger("mentor-x-ai.routes")

_start_time = _time.time()
limiter = Limiter(key_func=get_remote_address)


# ── Prompt Injection Protection ───────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all |previous |above )?instructions?",
    r"you are now",
    r"forget (everything|your instructions)",
    r"(system|assistant)\s*:",
    r"<\s*(script|iframe|img)[^>]*>",
    r"prompt\s*injection",
]


def _sanitize_question(text: str) -> str:
    """Detect and reject obvious prompt injection attempts."""
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            raise HTTPException(
                status_code=400,
                detail="Invalid input detected.",
            )
    return text


# ── Models ────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=1, max_length=1200)


def get_retriever(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        logger.warning("Retrieval service requested before startup completion")
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Please try again later.",
        )
    return retriever


# ── Endpoints ─────────────────────────────────────────────────
@router.get("/health")
async def health(request: Request):
    retriever = getattr(request.app.state, "retriever", None)
    ready = retriever is not None

    info: dict = {
        "status": "ok" if ready else "starting",
        "service": "Mentor-X AI",
        "ready": ready,
        "uptime_seconds": round(_time.time() - _start_time, 1),
    }

    if ready:
        try:
            store_info = retriever.store.info()
            info["vector_store"] = {
                "docs_count": store_info["docs_count"],
                "collection": store_info["collection"],
                "embedding_model": store_info["embedding_model"],
            }
        except Exception:
            info["vector_store"] = {"error": "unavailable"}

        if hasattr(retriever, "semantic_cache"):
            info["semantic_cache"] = {
                "cache_size": len(retriever.semantic_cache.cache),
                "cache_hits": retriever.semantic_cache.hits,
                "cache_misses": retriever.semantic_cache.misses,
                "cache_max_size": retriever.semantic_cache.max_size,
            }

    return info


@router.post("/chat")
@limiter.limit("20/minute")
async def chat(request: ChatRequest, retriever=Depends(get_retriever), debug: bool = False):
    question = _sanitize_question(request.question)
    start_time = _time.perf_counter()
    try:
        # Run retriever.ask() in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(retriever.ask, question)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the chat request.",
        )

    response_time_ms = (_time.perf_counter() - start_time) * 1000
    response = {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "response_time_ms": round(response_time_ms, 2),
    }

    if debug:
        response["hallucination"] = result.get("hallucination")

    return response


@router.post("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: ChatRequest, retriever=Depends(get_retriever)):
    """Stream LLM response token by token using Server-Sent Events (SSE)."""
    question = _sanitize_question(request.question)
    start_time = _time.perf_counter()

    async def generate() -> AsyncGenerator[str, None]:
        try:
            logger.info("Streaming chat request: %s", question[:80])

            # Retrieve + rerank in thread pool (CPU-bound)
            result = await asyncio.to_thread(
                _prepare_streaming_context,
                retriever,
                question,
            )

            if "error" in result:
                yield f"data: {json.dumps({'error': result['error']})}\n\n"
                return

            context = result["context"]
            sources = result["sources"]

            # Stream LLM response token by token
            try:
                if hasattr(retriever.chain, "astream"):
                    async for token in retriever.chain.astream({
                        "context": context,
                        "question": question,
                    }):
                        yield f"data: {json.dumps({'token': token})}\n\n"
                else:
                    # Fallback: run sync chain in thread pool, simulate streaming
                    result_text = await asyncio.to_thread(
                        lambda: retriever.chain.invoke({
                            "context": context,
                            "question": question,
                        })
                    )
                    for word in result_text.split():
                        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                        await asyncio.sleep(0.01)

            except Exception:
                logger.exception("LLM streaming failed")
                yield f"data: {json.dumps({'error': 'LLM streaming failed'})}\n\n"
                return

            # Send sources + timing at the end
            response_time_ms = (_time.perf_counter() - start_time) * 1000
            yield f"data: {json.dumps({'sources': sources, 'response_time_ms': round(response_time_ms, 2)})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception:
            logger.exception("Stream generation failed")
            yield f"data: {json.dumps({'error': 'Stream generation failed'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Helpers ───────────────────────────────────────────────────
def _prepare_streaming_context(retriever, question: str) -> dict:
    """Retrieve and rerank context for streaming (runs in thread pool)."""
    try:
        from retrieval.retriever import (
            _build_context,
            _normalize_query,
            _query_rewrite,
            _prepare_cache_key,
            _reject_empty_query,
            _rerank,
        )

        question = _normalize_query(question)

        if _reject_empty_query(question):
            return {"error": "Empty question"}

        rewritten_query = _query_rewrite(question)
        query_embedding = retriever.embeddings.embed_query(rewritten_query)
        cache_key = _prepare_cache_key(rewritten_query)
        cached_results = retriever.semantic_cache.get(cache_key, query_embedding)

        if cached_results is not None:
            docs_with_scores = cached_results
            logger.info("Streaming using semantic cache for query.")
        else:
            docs_with_scores = retriever.hybrid_retriever.search(rewritten_query, k=8)
            retriever.semantic_cache.set(cache_key, query_embedding, docs_with_scores)

        ranked_results = _rerank(docs_with_scores, rewritten_query)

        if not ranked_results:
            return {"error": "No relevant context found"}

        ranked_docs = [doc for doc, _ in ranked_results]
        context = _build_context(ranked_docs)

        sources = [
            {
                "source": doc.metadata.get("source", doc.metadata.get("filename", "unknown")),
                "page": doc.metadata.get("page_number", "?"),
                "score": round(score, 4),
                "preview": doc.page_content[:150] + "...",
            }
            for doc, score in ranked_results[:3]
        ]

        return {"context": context, "sources": sources}

    except Exception as exc:
        logger.exception("Context preparation failed")
        return {"error": str(exc)}