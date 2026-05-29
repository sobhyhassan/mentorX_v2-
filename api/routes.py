import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, constr

router = APIRouter()
logger = logging.getLogger("mentor-x-ai.routes")


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
    return {
        "status": "ok",
        "service": "Mentor-X AI",
        "ready": getattr(request.app.state, "retriever", None) is not None,
    }


@router.post("/chat")
async def chat(request: ChatRequest, retriever=Depends(get_retriever)):
    start_time = time.perf_counter()
    try:
        # Run retriever.ask() in thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(retriever.ask, request.question)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the chat request.",
        )

    response_time_ms = (time.perf_counter() - start_time) * 1000
    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "response_time_ms": round(response_time_ms, 2),
    }


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, retriever=Depends(get_retriever)):
    """Stream LLM response token by token using Server-Sent Events (SSE)."""
    start_time = time.perf_counter()

    async def generate() -> AsyncGenerator[str, None]:
        try:
            question = request.question
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
            response_time_ms = (time.perf_counter() - start_time) * 1000
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
            _reject_empty_query,
            _rerank,
        )

        question = _normalize_query(question)

        if _reject_empty_query(question):
            return {"error": "Empty question"}

        docs_with_scores = retriever.store.search_with_score(question, k=8)
        ranked_results = _rerank(docs_with_scores, question)

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