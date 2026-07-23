"""
tests/test_rag_pipeline.py
-----------------------------
Tests the RAG pipeline's prompt/context assembly logic using mocked
vectorstore and LLM objects (no real model downloads required).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.rag_pipeline import RAGPipeline, _format_docs


def test_format_docs_includes_source_and_content():
    docs = [
        Document(page_content="Paris is the capital of France.", metadata={"source": "geo.txt"}),
        Document(page_content="The Eiffel Tower is in Paris.", metadata={"source": "geo.txt"}),
    ]
    formatted = _format_docs(docs)
    assert "geo.txt" in formatted
    assert "Paris is the capital of France." in formatted
    assert "The Eiffel Tower is in Paris." in formatted


@patch("src.rag_pipeline.get_llm")
def test_pipeline_query_returns_answer_and_sources(mock_get_llm):
    # Mock vectorstore + retriever
    mock_doc = Document(page_content="Test content.", metadata={"source": "test.txt"})
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_retriever

    # Mock LLM chain output
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    pipeline = RAGPipeline(vectorstore=mock_vectorstore, top_k=1)

    # Patch the chain's invoke to avoid needing a real LLM call
    with patch.object(pipeline, "llm", mock_llm):
        with patch(
            "src.rag_pipeline.StrOutputParser"
        ) as mock_parser_cls, patch(
            "src.rag_pipeline.ChatPromptTemplate.from_template"
        ) as mock_prompt_cls:
            fake_chain = MagicMock()
            fake_chain.invoke.return_value = "Mocked answer"

            # Build a fake pipeline: prompt | llm | parser -> our fake_chain
            mock_prompt_instance = MagicMock()
            mock_prompt_cls.return_value = mock_prompt_instance
            mock_prompt_instance.__or__ = MagicMock(return_value=fake_chain)
            fake_chain.__or__ = MagicMock(return_value=fake_chain)

            pipeline.prompt = mock_prompt_instance
            result = pipeline.query("What is this about?")

    assert "sources" in result
    assert result["sources"][0].metadata["source"] == "test.txt"
