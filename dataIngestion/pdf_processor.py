from pathlib import Path
import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_LENGTH
)

try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain.docstore.document import Document


class SmartPDFProcessor:

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_length: int = MIN_CHUNK_LENGTH,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        # Better semantic splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,

            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],

            add_start_index=True,
        )

    def process(self, file_path: str) -> List[Document]:

        pages = self._load_pdf(file_path)

        total_pages = len(pages)

        cleaned_pages = []

        # ── Clean Pages ───────────────────────────────
        for page_num, page in enumerate(pages):

            cleaned = self._clean_text(page.page_content)

            # 🔥 Remove useless pages/chunks
            if len(cleaned.split()) < 40:
                continue

            if len(cleaned.strip()) < MIN_CHUNK_LENGTH:
                continue

            page.page_content = cleaned

            page.metadata.update({
                "source": file_path,
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "processor": "smart_pdf_v2",
            })

            cleaned_pages.append(page)

        # ── Split Documents ──────────────────────────
        chunks = self.text_splitter.split_documents(cleaned_pages)

        # ── Remove Tiny Chunks ───────────────────────
        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.page_content.strip()) >= self.min_chunk_length
        ]

        # ── Deduplicate Chunks ───────────────────────
        unique_chunks = []
        seen = set()

        for chunk in chunks:

            text = chunk.page_content.strip()

            if text not in seen:
                seen.add(text)
                unique_chunks.append(chunk)

        print(f"Final clean chunks: {len(unique_chunks)}")

        return unique_chunks

    def _load_pdf(self, file_path: str) -> List[Document]:

        pdf_path = Path(file_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_path}"
            )

        return PyPDFLoader(str(pdf_path)).load()

    def _clean_text(self, text: str) -> str:

        # Remove emails
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove weird symbols
        text = re.sub(r"[∗†‡§¶•]", "", text)

        # Remove isolated page numbers
        text = re.sub(r"\b\d+\s*$", "", text, flags=re.MULTILINE)

        # Normalize spacing
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" \n|\n ", "\n", text)

        # Remove control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize quotes
        text = (
            text
            .replace("”", '"')
            .replace("“", '"')
            .replace("’", "'")
        )

        return text.strip()