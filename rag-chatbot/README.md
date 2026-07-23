# 🤖 RAG Chatbot — Capstone AI/ML Project

A production-style **Retrieval-Augmented Generation (RAG) Chatbot** that lets you chat with your own documents (PDF, DOCX, TXT, MD). Built with `LangChain`, `FAISS`, `Sentence-Transformers`, and `Streamlit`.

---

## 📌 Features

- 📄 Ingests PDF, DOCX, TXT, and Markdown files
- ✂️ Smart chunking with configurable overlap
- 🔎 Semantic search using FAISS vector store
- 🧠 Local (HuggingFace) or Cloud (OpenAI) embeddings & LLMs — switch via `.env`
- 💬 Chat UI built with Streamlit, with source citations
- 🧪 Unit tests for ingestion + pipeline
- 🗂️ Clean, modular, GitHub-ready structure

---

## 🏗️ Project Structure

```
rag-chatbot/
├── app.py                     # Streamlit chatbot UI (entry point)
├── config.py                  # Central configuration (paths, models, params)
├── requirements.txt           # Python dependencies
├── .env.example                # Sample environment variables
├── .gitignore
├── README.md
├── data/
│   ├── raw/                   # Put your source documents here
│   └── processed/             # Cached chunked text (auto-generated)
├── vectorstore/                # FAISS index (auto-generated, gitignored)
├── src/
│   ├── __init__.py
│   ├── ingest.py               # Load & chunk documents
│   ├── embeddings.py           # Embedding model wrapper
│   ├── vectorstore.py          # Build / load / save FAISS index
│   ├── llm.py                  # LLM wrapper (OpenAI or local HF model)
│   ├── rag_pipeline.py         # Retrieval + generation pipeline
│   └── utils.py                # Logging & helper functions
├── scripts/
│   └── build_index.py          # CLI script to (re)build the vector index
└── tests/
    ├── __init__.py
    ├── test_ingest.py
    └── test_rag_pipeline.py
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Choose "local" (free, runs on CPU, no key needed) or "openai"
LLM_PROVIDER=local

# Only needed if LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_MODEL=gpt-4o-mini

# Only needed if LLM_PROVIDER=local (any HF text-generation model)
LOCAL_LLM_MODEL=google/flan-t5-base

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=4
```

### 3. Add your documents

Drop `.pdf`, `.docx`, `.txt`, or `.md` files into `data/raw/`.

### 4. Build the vector index

```bash
python scripts/build_index.py
```

### 5. Launch the chatbot

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 🧠 How It Works (Architecture)

```
 ┌─────────────┐   chunk    ┌───────────────┐  embed   ┌─────────────┐
 │  Documents  │ ─────────▶ │  Text Chunks  │ ───────▶ │ FAISS Index │
 └─────────────┘            └───────────────┘          └─────────────┘
                                                                │
 User Question ──▶ Embed Query ──▶ Similarity Search ◀─────────┘
                                          │
                                          ▼
                              Top-K Relevant Chunks
                                          │
                                          ▼
                         Prompt = Question + Context
                                          │
                                          ▼
                                    LLM Generates
                                          │
                                          ▼
                              Answer + Cited Sources
```

1. **Ingestion** (`src/ingest.py`): Loads raw files and splits them into overlapping chunks using `RecursiveCharacterTextSplitter`.
2. **Embeddings** (`src/embeddings.py`): Converts chunks into dense vectors using a Sentence-Transformers model.
3. **Vector Store** (`src/vectorstore.py`): Stores/searches vectors using FAISS; persisted to disk in `vectorstore/`.
4. **Retrieval + Generation** (`src/rag_pipeline.py`): Retrieves the top-K relevant chunks for a query and feeds them, along with the question, to the LLM to generate a grounded answer.
5. **UI** (`app.py`): Streamlit chat interface with conversation history and source-document citations.

---

## 🚀 Possible Extensions (great for a capstone writeup)

- Swap FAISS for a managed vector DB (Pinecone, Weaviate, Chroma Cloud)
- Add re-ranking (Cohere Rerank / cross-encoder)
- Add conversation memory / multi-turn follow-ups
- Deploy on Streamlit Community Cloud / HuggingFace Spaces / Docker
- Add evaluation harness (RAGAS) for faithfulness & answer relevancy

---

## 📄 License

MIT License — free to use for learning and portfolio projects.
