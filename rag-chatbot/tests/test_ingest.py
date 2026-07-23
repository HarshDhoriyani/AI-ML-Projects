"""
tests/test_ingest.py
----------------------
Basic unit tests for the ingestion / chunking logic.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.ingest import split_documents


def test_split_documents_creates_chunks():
    long_text = "This is a sentence. " * 200  # ~4000 characters
    docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]

    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 500 + 50  # allow splitter slack
        assert chunk.metadata["source"] == "test.txt"


def test_split_documents_empty_input():
    assert split_documents([]) == []


def test_split_documents_preserves_metadata():
    docs = [
        Document(page_content="Short text.", metadata={"source": "a.txt", "page": 1})
    ]
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "a.txt"
    assert chunks[0].metadata["page"] == 1
