import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routes import router
from retrieval.retriever import MentorRetriever
from utils.logging import setup_logging
from utils.models import warmup_models


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown."""

    # ── Startup ───────────────────────────────────────────────
    setup_logging()
    logger.info("Mentor-X AI startup initiated")

    # Run CPU/IO-bound model loading in thread pool
    # to avoid blocking the async event loop
    await asyncio.to_thread(warmup_models)
    app.state.retriever = await asyncio.to_thread(MentorRetriever)

    logger.info("Mentor-X AI startup complete")

    yield

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("Mentor-X AI shutdown complete")


logger = logging.getLogger("mentor-x-ai")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Mentor-X AI",
    description="RAG-powered academic document Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(router)