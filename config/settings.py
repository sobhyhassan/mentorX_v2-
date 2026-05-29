from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Project Root ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data Paths ────────────────────────────────────────────────
PDF_DATA_DIR  = BASE_DIR / "dataIngestion" / "pdf_data"
CHROMA_DB_DIR = BASE_DIR / "vectorStore"   / "chroma_db"

# ── PDF Processor ─────────────────────────────────────────────
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
MIN_CHUNK_LENGTH = 100

# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# ── Vector Store ──────────────────────────────────────────────
COLLECTION_NAME = "mentor_x_docs"

# [TUNED] واسّعنا الـ initial retrieval عشان الـ reranker يشتغل على عينة أكبر
TOP_K_RESULTS = 8

# ── Retrieval Tuning ──────────────────────────────────────────
# [TUNED] خفّضنا الـ threshold عشان منحجبش chunks صح
# الـ cross-encoder reranker هو اللي بيعمل الفلترة الحقيقية
MIN_RELEVANCE_SCORE = 0.2
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# [NEW] عدد الـ chunks اللي بتتبعت للـ LLM بعد الـ reranking
FINAL_TOP_K = 3

# ── Context Limits ────────────────────────────────────────────
# Moved from hardcoded constants in retriever.py
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 4000))
MAX_CONTEXT_CHARS  = int(os.getenv("MAX_CONTEXT_CHARS", 16000))

# ── LLM ──────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MODEL = GROQ_MODEL
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "false").strip().lower() in ("1", "true", "yes")
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", 0.45))
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.90))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 128))

# ── API Authentication ────────────────────────────────────────
MENTOR_API_KEY = os.getenv("MENTOR_API_KEY", "")

# ── Startup Validation ────────────────────────────────────────
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is required but not set. "
        "Please set the GROQ_API_KEY in your .env file."
    )