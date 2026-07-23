"""
scripts/build_index.py
------------------------
CLI entry point to (re)build the FAISS vector index from documents in data/raw/.

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

# Allow running this script directly (adds project root to sys.path)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from src.ingest import ingest
from src.utils import get_logger
from src.vectorstore import build_vectorstore, save_vectorstore

logger = get_logger(__name__)


def main():
    logger.info("Starting index build from: %s", config.DATA_RAW_DIR)

    chunks = ingest()
    if not chunks:
        logger.error(
            "No chunks produced. Add .pdf/.docx/.txt/.md files to data/raw/ and retry."
        )
        sys.exit(1)

    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)

    logger.info(
        "✅ Index built successfully with %d chunks. Saved to %s",
        len(chunks),
        config.VECTORSTORE_DIR,
    )


if __name__ == "__main__":
    main()
