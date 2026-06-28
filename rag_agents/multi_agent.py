import logging

from rag.rag_pipeline import RAGPipeline
from utils.email_service import send_email

logger = logging.getLogger(__name__)


class RAGMultiAgentSystem:
    """
    Agent 1 — Ingest   : load & chunk document into vector store
    Agent 2 — Retrieve : run RAG chain, get real answer + scores
    Agent 3 — Format   : build readable output string
    Agent 4 — Deliver  : send result by email
    """

    def __init__(self):
        self.rag = RAGPipeline()

    def _agent_ingest(self, file_path: str) -> None:
        logger.info("[Agent 1] Ingesting: %s", file_path)
        self.rag.ingest(file_path)

    def _agent_query(self, question: str) -> dict:
        logger.info("[Agent 2] Querying: %s", question)
        return self.rag.query(question)

    def _agent_format(self, question: str, result: dict) -> str:
        logger.info("[Agent 3] Formatting result")
        return (
            "🧠 RAG QUERY RESULT\n\n"
            f"📌 Query:\n{question}\n\n"
            f"✅ Answer:\n{result['answer']}\n\n"
            "📊 Scores:\n"
            f"  • Relevance   : {result['relevance']}\n"
            f"  • Faithfulness: {result['faithfulness']}\n"
            f"  • Confidence  : {result['confidence']}\n"
        )

    def _agent_send(self, user_email: str, body: str) -> None:
        logger.info("[Agent 4] Sending to %s", user_email)
        send_email(user_email, "RAG Query Result", body)

    def process(self, file_path: str, query: str, user_email: str) -> str:
        self._agent_ingest(file_path)
        result = self._agent_query(query)
        formatted = self._agent_format(query, result)
        self._agent_send(user_email, formatted)
        return formatted