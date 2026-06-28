Yes! Here's a complete guide to deploy on Streamlit Community Cloud (free).

---

## The problem first

Streamlit Cloud **cannot run Ollama** — it's a local LLM server that needs to run on your machine. For cloud deployment you need to swap Ollama for a cloud LLM API. The easiest free option is **Groq** (extremely fast, free tier, no credit card).Here's the complete step-by-step guide.

---

## Step 1 — Get a free Groq API key

Go to **console.groq.com** → sign up → API Keys → Create API Key. Copy it. It's free, no credit card needed, and very fast (runs Llama 3 etc.).

---

## Step 2 — Swap Ollama → Groq in your code

Replace `rag/rag_pipeline.py` with this updated version:

```python
import os
import logging

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from storage.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer ONLY from the context below.\n"
    "If the answer is not in the context, say: "
    "'I could not find relevant information in the document.'\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\nAnswer:"
)


class RAGPipeline:
    def __init__(self):
        self.store = InMemoryVectorStore()
        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )

    def ingest(self, file_path: str) -> None:
        loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path, encoding="utf-8")
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50)),
        ).split_documents(loader.load())
        if not chunks:
            raise ValueError(f"No text extracted from: {file_path}")
        self.store.add_documents(chunks)

    def _score(self, answer: str, docs: list) -> dict:
        context = " ".join(d.page_content.lower() for d in docs)
        a_words = set(answer.lower().split())
        c_words = set(context.split())
        relevance = round(min(len(a_words & c_words) / max(len(a_words), 1), 1.0), 2)
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        grounded = sum(1 for s in sentences if any(w in context for w in s.lower().split()))
        faithfulness = round(grounded / max(len(sentences), 1), 2)
        denom = relevance + faithfulness
        confidence = round(2 * relevance * faithfulness / denom, 2) if denom else 0.0
        return {"relevance": relevance, "faithfulness": faithfulness, "confidence": confidence}

    def query(self, question: str) -> dict:
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY is not set.")
        retriever = self.store.get_retriever()
        retrieved_docs = retriever.invoke(question)
        chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | _PROMPT | self.llm | StrOutputParser()
        )
        answer = chain.invoke(question)
        return {"answer": answer, **self._score(answer, retrieved_docs)}
```

Update `.env` — swap Ollama vars for Groq:
```env
# Groq (replaces Ollama)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama3-8b-8192

# HuggingFace Embeddings (unchanged)
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Gmail SMTP (unchanged)
EMAIL_SENDER=you@gmail.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx

# Config (unchanged)
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
```

Update `pyproject.toml` — one line change:
```toml
"langchain-groq>=0.2.0",      # replaces langchain-ollama
```

Install it locally:
```bash
pip install langchain-groq
```

---

## Step 3 — Prepare GitHub repo

```bash
# In your project root
git init
git add .
git commit -m "RAG multi-agent app"
```

Create a `.gitignore` so secrets never get pushed:
```
.env
.venv/
__pycache__/
*.pyc
```

Push to GitHub:
```bash
git remote add origin https://github.com/YOUR_USERNAME/your-repo-name.git
git push -u origin main
```

---

## Step 4 — Add `requirements.txt`

Streamlit Cloud uses `requirements.txt`, not `pyproject.toml`:

```txt
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-text-splitters>=0.3.0
langchain-groq>=0.2.0
langchain-huggingface>=0.1.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
pypdf>=4.0.0
streamlit>=1.40.0
python-dotenv>=1.0.0
```

---

## Step 5 — Deploy on Streamlit Cloud

1. Go to **share.streamlit.io** and sign in with GitHub
2. Click **"New app"**
3. Select your repo, branch `main`, and set main file path to `ui/app.py`
4. Click **"Advanced settings"** → **Secrets** and paste all your secrets:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_MODEL = "llama3-8b-8192"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMAIL_SENDER = "you@gmail.com"
EMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"
CHUNK_SIZE = "500"
CHUNK_OVERLAP = "50"
TOP_K = "3"
```

5. Click **Deploy** — it'll be live in ~3 minutes at `https://your-app-name.streamlit.app`

---

**One small change needed in `ui/app.py`** — Streamlit Cloud reads secrets via `st.secrets`, not `.env`. Add this near the top, before imports that need env vars:

```python
import streamlit as st

# Push Streamlit secrets into os.environ so existing code works unchanged
for key, val in st.secrets.items():
    os.environ[key] = str(val)
```

That's it — no other code changes needed. Your emails, scoring, and RAG pipeline all stay exactly the same.