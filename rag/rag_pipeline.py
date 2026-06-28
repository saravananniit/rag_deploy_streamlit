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