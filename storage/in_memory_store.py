import os
import logging

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self.vector_store: FAISS | None = None

    def add_documents(self, documents) -> None:
        if not self.vector_store:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        logger.info("Vector store updated with %d documents.", len(documents))

    def get_retriever(self):
        if not self.vector_store:
            raise RuntimeError(
                "Vector store is empty. Call ingest() before querying."
            )
        return self.vector_store.as_retriever(
            search_kwargs={"k": int(os.getenv("TOP_K", 3))}
        )