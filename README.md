# 🧠 Mentor-X AI

> **AI-powered intelligent tutoring system** using Retrieval-Augmented Generation (RAG) to answer questions based on custom educational documents with semantic search capabilities.

---

## 📌 About This Project

Mentor-X AI is a **Retrieval-Augmented Generation (RAG)** system built for your graduation project. It intelligently processes educational PDFs and creates an intelligent Q&A assistant that:

- **Ingests** educational documents (PDFs) and chunks them intelligently
- **Embeds** text into vector space for semantic understanding
- **Stores** vectors in a ChromaDB vector database for fast retrieval
- **Retrieves** the most relevant document chunks based on user queries
- **Generates** accurate, context-aware answers using the Groq LLM

Instead of relying solely on an LLM's training data, Mentor-X grounds all answers in your actual documents, ensuring accuracy and relevance.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **PDF Processing** | Intelligent extraction and chunking of PDF documents |
| ✂️ **Smart Chunking** | Configurable chunk size with overlap for context preservation |
| 🧬 **Embeddings** | Uses `sentence-transformers` (all-MiniLM-L6-v2) for efficient semantic embeddings |
| 🗄️ **Vector Storage** | ChromaDB for persistent and fast vector database |
| 🔍 **Semantic Search** | Find relevant content using similarity search on embeddings |
| 🤖 **LLM Integration** | Groq API for high-quality, context-aware answer generation |
| ⚡ **Fast Inference** | Optimized for quick retrieval and response generation |

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Framework** | LangChain |
| **Vector DB** | ChromaDB |
| **Embeddings** | Sentence-Transformers (HuggingFace) |
| **LLM** | Groq API (Llama 3) |
| **PDF Processing** | PyPDF |
| **Environment** | uv (fast Python package manager) |
| **Dependencies** | langchain-groq, langchain-openai, langchain-community |

---

## 📂 Project Structure

```
mentor-x-ai/
├── main.py                 # Main ingestion pipeline
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
│
├── api/                   # API endpoints (future expansion)
│   ├── __init__.py
│   └── main.py
│
├── config/                # Configuration settings
│   └── settings.py        # Chunk size, embedding model, LLM config
│
├── dataIngestion/         # Document processing
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF extraction & chunking
│   ├── embedding_manager.py # Embedding generation
│   └── pdf_data/          # Input PDFs directory
│
├── retrieval/             # Query retrieval logic
│   ├── __init__.py
│   └── retriever.py       # Semantic search implementation
│
└── vectorStore/           # Vector database
    ├── __init__.py
    ├── chroma_store.py    # ChromaDB management
    └── chroma_db/         # Persisted vector database
```

---

## ⚙️ Configuration

All settings are managed in `config/settings.py`:

```python
# PDF Chunking
CHUNK_SIZE = 800              # Characters per chunk
CHUNK_OVERLAP = 150           # Overlap between chunks for context

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast & free
EMBEDDING_BATCH_SIZE = 32

# Vector Store
COLLECTION_NAME = "mentor_x_docs"
TOP_K_RESULTS = 5             # Retrieved chunks per query

# LLM
LLM_MODEL = "llama3-8b-8192"  # Via Groq API
LLM_TEMPERATURE = 0.2         # Lower = more deterministic
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- pip or uv
- Groq API key (get free at [console.groq.com](https://console.groq.com))

### 1️⃣ Clone & Navigate
```bash
cd mentor-x-ai
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
```

### 3️⃣ Activate Environment (Windows PowerShell)
```powershell
.\.venv\Scripts\Activate
```

Or on Mac/Linux:
```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 5️⃣ Setup Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Quick Start

### Step 1: Add Your PDF
Place your PDF file in `dataIngestion/pdf_data/` directory.

### Step 2: Ingest the Document
```bash
python main.py
```

This will:
- ✅ Extract and chunk the PDF
- ✅ Generate embeddings
- ✅ Store vectors in ChromaDB
- ✅ Run a sample search to verify

### Step 3: Query the System
```python
from vectorStore.chroma_store import VectorStoreManager

store = VectorStoreManager()
results = store.similarity_search("Your question here", k=3)

for result in results:
    print(result.page_content)
    print(result.metadata)
```

---

## 📊 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    MENTOR-X AI PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

  📄 PDF File
     │
     ├──► [PDF Processor] → Extract text & create chunks
     │
     ├──► [Embedding Manager] → Convert chunks to vectors
     │    (using sentence-transformers)
     │
     ├──► [Vector Store] → Store in ChromaDB (persistent)
     │
     └──► User Query
          │
          ├──► [Semantic Search] → Find top-k similar chunks
          │    (using embedding similarity)
          │
          ├──► [LLM] → Generate answer using Groq
          │    (conditioned on retrieved context)
          │
          └──► 📝 Answer
```

---

## 📚 Key Components

### 1. **PDF Processor** (`dataIngestion/pdf_processor.py`)
Extracts text from PDFs and splits into semantic chunks while preserving context.

### 2. **Embedding Manager** (`dataIngestion/embedding_manager.py`)
Converts text chunks into 384-dimensional vectors using `sentence-transformers`.

### 3. **Vector Store** (`vectorStore/chroma_store.py`)
Manages ChromaDB for efficient vector storage and retrieval.

### 4. **Retriever** (`retrieval/retriever.py`)
Implements semantic search to find relevant document chunks.

### 5. **Settings** (`config/settings.py`)
Centralized configuration for all system parameters.

---

## 🔧 API Endpoints (Future)

The `api/` folder is prepared for REST API endpoints to:
- Upload documents
- Query the system
- Manage collections

---

## 🧪 RAGAS Evaluation

A new evaluation package is available in `RAGASevaluation/` for measuring retrieval and answer quality.

Features:
- sample dataset loader and JSON dataset support
- answer similarity, exact match, and token F1 scoring
- retrieval precision, recall, F1, and Hit@K
- CLI runner for quick evaluation runs

Run evaluation from the repository root:

```bash
python main.py --evaluate
```

Or provide a custom dataset:

```bash
python main.py --evaluate --dataset RAGASevaluation/sample_questions.json
```

Launch the Streamlit dashboard:

```bash
streamlit run streamlit_dashboard.py
```

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Embedding Dimension | 384 |
| Model Size | ~22MB |
| Inference Time | < 1 second per query |
| Storage | ~50MB for 100 pages |

---

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

---

## 📝 License

This is a graduation project. Use and modify as needed.

---

## 📞 Support

For issues or questions about the project, refer to the code comments (including Arabic explanations) and configuration files.

---

**Made with ❤️ for your graduation project**
