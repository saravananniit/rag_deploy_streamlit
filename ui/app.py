import os
import sys
import logging
import tempfile
import traceback
from pathlib import Path

import streamlit as st

# ── Secrets: load from Streamlit Cloud first, fall back to .env locally ──
# This must happen BEFORE any other import that reads os.environ
def _load_secrets():
    try:
        # Streamlit Cloud: push st.secrets into os.environ
        for key, val in st.secrets.items():
            os.environ[key] = str(val)
    except Exception:
        # Local: fall back to .env file
        from dotenv import load_dotenv
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        load_dotenv(PROJECT_ROOT / ".env")

_load_secrets()

# ── sys.path so local packages resolve ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_agents.multi_agent import RAGMultiAgentSystem  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Email Assistant", layout="wide")
st.title("📄 RAG Email Assistant")
st.caption(
    f"Model: `{os.getenv('GROQ_MODEL', 'llama3-8b-8192')}` · "
    f"Embeddings: `{os.getenv('HF_EMBED_MODEL', 'all-MiniLM-L6-v2')}` · "
    f"Powered by Groq"
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Config")
    st.code(
        f"GROQ_MODEL     = {os.getenv('GROQ_MODEL', 'llama3-8b-8192')}\n"
        f"HF_EMBED_MODEL = {os.getenv('HF_EMBED_MODEL', 'all-MiniLM-L6-v2')}\n"
        f"CHUNK_SIZE     = {os.getenv('CHUNK_SIZE', '500')}\n"
        f"TOP_K          = {os.getenv('TOP_K', '3')}",
        language="ini",
    )

    # Show whether GROQ_API_KEY is configured (without revealing it)
    key_set = bool(os.getenv("GROQ_API_KEY"))
    if key_set:
        st.success("GROQ_API_KEY ✓ set")
    else:
        st.error("GROQ_API_KEY not set")

# ── Main form ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📎 Upload document", type=["txt", "pdf"],
    help="Plain text or PDF file to query against."
)
query = st.text_input("🔍 Your question", placeholder="e.g. Tell me about NIIT")
email = st.text_input("📧 Send result to", placeholder="you@example.com")

if st.button("▶ Run RAG Pipeline", type="primary", use_container_width=True):

    # Validate inputs
    errors = []
    if not uploaded_file:
        errors.append("Please upload a document.")
    if not query.strip():
        errors.append("Please enter a question.")
    if not email.strip():
        errors.append("Please enter a recipient email.")
    if not os.getenv("GROQ_API_KEY"):
        errors.append("GROQ_API_KEY is not configured. Add it in .env (local) or Streamlit secrets (cloud).")
    for e in errors:
        st.warning(f"⚠️ {e}")
    if errors:
        st.stop()

    # Run pipeline
    with st.spinner("Running pipeline…"):
        try:
            suffix = Path(uploaded_file.name).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            result = RAGMultiAgentSystem().process(
                file_path, query.strip(), email.strip()
            )

            st.success("✅ Done — result sent to your inbox!")
            st.text_area("📊 Output Preview", result, height=320)

        except EnvironmentError as exc:
            st.error(f"🔑 Configuration error: {exc}")

        except ValueError as exc:
            st.error(f"📄 Document error: {exc}")

        except Exception as exc:
            st.error(f"❌ Error: {exc}")
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())