"""
main.py — Mentor-X AI Entry Point
"""

import argparse
from pathlib import Path

from config.settings import (
    PDF_DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH,
)

from dataIngestion.pdf_processor import SmartPDFProcessor
from retrieval.retriever import MentorRetriever
from vectorStore.chroma_store import VectorStoreManager

from utils.models import warmup_models


# ── PDF Ingestion ─────────────────────────────────────────────
def ingest_pdf(pdf_path: str) -> None:

    print("\n" + "=" * 55)
    print("MENTOR-X AI — PDF Ingestion")
    print("=" * 55)

    processor = SmartPDFProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_length=MIN_CHUNK_LENGTH,
    )

    chunks = processor.process(pdf_path)
    print(f"\nExtracted {len(chunks)} chunks")

    store = VectorStoreManager()

    existing_docs = store.get_collection_count()
    if existing_docs > 0:
        print("\nVector DB already contains documents")
        print("Skipping ingestion")
        return

    store.add_documents(chunks)
    print("Documents stored in Chroma")

    print("\n" + "=" * 55)
    print("Ingestion complete!")
    print("=" * 55 + "\n")


# ── RAG Test ──────────────────────────────────────────────────
def run_rag_test() -> None:

    print("\n" + "=" * 55)
    print("MENTOR-X AI — RAG Test")
    print("=" * 55)

    mentor = MentorRetriever()

    questions = [
        "What is the attention mechanism?",
        "What is the Transformer architecture?",
    ]

    for q in questions:

        print("\n" + "-" * 55)
        print(f"\n Question:\n{q}")

        result = mentor.ask(q)

        print(f"\nAnswer:\n{result['answer']}")
        print("\nSources:")

        for src in result["sources"]:
            print(f"• Page {src['page']} | Score: {src['score']}")
            print(f"  {src['preview']}")

    print("\n" + "=" * 55 + "\n")


# ── API Server ────────────────────────────────────────────────
def run_api() -> None:

    import uvicorn

    print("\nStarting Mentor-X AI API...")
    print("Swagger Docs: http://127.0.0.1:8000/docs\n")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


# ── Main ──────────────────────────────────────────────────────
def main() -> None:

    parser = argparse.ArgumentParser(description="Mentor-X AI")

    parser.add_argument("--ingest", action="store_true", help="Ingest PDF into vector database")
    parser.add_argument("--test", action="store_true", help="Run RAG pipeline test")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--pdf", type=str, default=str(PDF_DATA_DIR / "attention.pdf"),
                        help="PDF file path to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset vector store before ingestion")

    args = parser.parse_args()

    pdf_file = Path(args.pdf)

    if args.api:
        run_api()
        return

    warmup_models()

    if args.reset:
        store = VectorStoreManager()
        store.reset()

    if args.ingest:
        ingest_pdf(str(pdf_file))
        return

    if args.test:
        run_rag_test()
        return

    # Default
    print("\nNo command provided.")
    print("\nExamples:")
    print("  python main.py --ingest")
    print("  python main.py --test")
    print("  python main.py --api")


# ── Entry Point ───────────────────────────────────────────────
if __name__ == "__main__":
    main()