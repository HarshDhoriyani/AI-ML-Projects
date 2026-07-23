"""
src/rag_pipeline.py
---------------------
Combines the FAISS retriever with the LLM to answer questions grounded in
the ingested documents, and returns the source chunks used for citation.
"""

from typing import Dict, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

import config
from src.llm import get_llm
from src.utils import get_logger
from src.vectorstore import load_vectorstore

logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below.
If the answer cannot be found in the context, say "I don't have enough information in the \
provided documents to answer that." Do not make anything up.

Context:
{context}

Question: {question}

Answer clearly and concisely:"""


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


class RAGPipeline:
    """A retrieval-augmented generation pipeline over a FAISS vector store."""

    def __init__(self, vectorstore: FAISS = None, top_k: int = config.TOP_K):
        self.vectorstore = vectorstore or load_vectorstore()
        if self.vectorstore is None:
            raise RuntimeError(
                "Vector store not found. Run `python scripts/build_index.py` first."
            )
        self.top_k = top_k
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def query(self, question: str) -> Dict:
        """
        Answers `question` using retrieved context.
        Returns a dict with keys: 'answer' and 'sources' (list of Document).
        """
        retrieved_docs = self.retriever.invoke(question)
        context = _format_docs(retrieved_docs)

        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        return {
            "answer": answer.strip(),
            "sources": retrieved_docs,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() in ("quit", "exit"):
            break
        result = pipeline.query(q)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for doc in result["sources"]:
            print(" -", doc.metadata.get("source", "unknown"))
