"""
src/embeddings.py
-------------------
Wraps the Sentence-Transformers embedding model used to turn text chunks
(and queries) into dense vectors for similarity search.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

import config
from src.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a cached HuggingFaceEmbeddings instance so the model is only
    loaded into memory once per process.
    """
    logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
