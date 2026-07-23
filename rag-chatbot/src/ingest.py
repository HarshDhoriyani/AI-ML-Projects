"""
src/ingest.py
--------------
Loads documents from `data/raw/` (PDF, DOCX, TXT, MD) and splits them into
overlapping text chunks ready for embedding.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

import config
from src.utils import get_logger

logger = get_logger(__name__)


def _load_single_file(file_path: Path) -> List[Document]:
    """Load a single file into LangChain Document objects based on its extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        logger.warning("Skipping unsupported file type: %s", file_path.name)
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.name
    return docs


def load_documents(raw_dir: Path = config.DATA_RAW_DIR) -> List[Document]:
    """Load every supported document found in `raw_dir`."""
    raw_dir = Path(raw_dir)
    all_docs: List[Document] = []

    files = [
        f
        for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(
            "No supported documents found in %s. Add .pdf/.docx/.txt/.md files "
            "before building the index.",
            raw_dir,
        )
        return all_docs

    for file_path in files:
        logger.info("Loading: %s", file_path.name)
        all_docs.extend(_load_single_file(file_path))

    logger.info("Loaded %d document(s) totaling %d page/section(s).", len(files), len(all_docs))
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """Split loaded documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(chunks))
    return chunks


def ingest(raw_dir: Path = config.DATA_RAW_DIR) -> List[Document]:
    """Full ingestion pipeline: load + split. Returns list of chunked Documents."""
    documents = load_documents(raw_dir)
    if not documents:
        return []
    return split_documents(documents)


if __name__ == "__main__":
    chunks = ingest()
    print(f"Produced {len(chunks)} chunks.")
    if chunks:
        print("\n--- Sample chunk ---")
        print(chunks[0].page_content[:300])
        print("Metadata:", chunks[0].metadata)
