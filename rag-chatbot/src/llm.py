"""
src/llm.py
-----------
Provides a single get_llm() factory that returns either:
  - a local, free HuggingFace text2text-generation model (default), or
  - an OpenAI chat model (if LLM_PROVIDER=openai and an API key is set)

Switch providers purely through environment variables — no code changes needed.
"""

from functools import lru_cache

import config
from src.utils import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """Return a LangChain-compatible LLM object based on config.LLM_PROVIDER."""

    if config.LLM_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing. "
                "Set it in your .env file."
            )
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI model: %s", config.OPENAI_MODEL)
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.2,
        )

    # Default: local, free HuggingFace model (no API key required)
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline

    logger.info("Using local HuggingFace model: %s", config.LOCAL_LLM_MODEL)
    hf_pipeline = pipeline(
        "text2text-generation",
        model=config.LOCAL_LLM_MODEL,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)
