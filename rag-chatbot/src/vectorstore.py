"""
src/vectorstore.py
--------------------
Builds, saves, and loads a FAISS vector index over document chunks.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

import config
from src.embeddings import get_embedding_model
from src.utils import get_logger

logger = get_logger(__name__)


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Build a fresh FAISS index from a list of document chunks."""
    if not chunks:
        raise ValueError(
            "No chunks provided. Add documents to data/raw/ and run ingestion first."
        )
    embeddings = get_embedding_model()
    logger.info("Embedding %d chunks and building FAISS index...", len(chunks))
    vs = FAISS.from_documents(chunks, embeddings)
    return vs


def save_vectorstore(vs: FAISS, path: Path = config.VECTORSTORE_DIR) -> None:
    """Persist the FAISS index to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path), index_name=config.FAISS_INDEX_NAME)
    logger.info("Saved FAISS index to %s", path)


def load_vectorstore(path: Path = config.VECTORSTORE_DIR) -> Optional[FAISS]:
    """Load a previously saved FAISS index from disk, if it exists."""
    path = Path(path)
    index_file = path / f"{config.FAISS_INDEX_NAME}.faiss"

    if not index_file.exists():
        logger.warning(
            "No FAISS index found at %s. Run `python scripts/build_index.py` first.",
            path,
        )
        return None

    embeddings = get_embedding_model()
    vs = FAISS.load_local(
        str(path),
        embeddings,
        index_name=config.FAISS_INDEX_NAME,
        allow_dangerous_deserialization=True,
    )
    logger.info("Loaded FAISS index from %s", path)
    return vs
