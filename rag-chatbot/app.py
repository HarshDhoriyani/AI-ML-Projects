"""
app.py
-------
Streamlit chat UI for the RAG Chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st

import config
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 RAG Chatbot")
st.caption(
    "Chat with your own documents — answers are grounded in the files you added to `data/raw/`."
)


@st.cache_resource(show_spinner="Loading vector index and model...")
def load_pipeline():
    return RAGPipeline()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**LLM Provider:** `{config.LLM_PROVIDER}`")
    st.markdown(f"**Embedding model:** `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**Top-K retrieved chunks:** `{config.TOP_K}`")
    st.divider()
    st.markdown(
        "📁 Add documents to `data/raw/` then run:\n\n"
        "```bash\npython scripts/build_index.py\n```\n"
        "to (re)build the index, then refresh this page."
    )
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Load pipeline (fails gracefully if index doesn't exist yet)
# ---------------------------------------------------------------------------
try:
    pipeline = load_pipeline()
except RuntimeError as e:
    st.error(str(e))
    st.info(
        "👉 Add files to `data/raw/`, then run `python scripts/build_index.py` "
        "in your terminal, then refresh this page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = pipeline.query(question)
            answer = result["answer"]
            sources = sorted(
                {doc.metadata.get("source", "unknown") for doc in result["sources"]}
            )

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for src in sources:
                        st.markdown(f"- `{src}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
