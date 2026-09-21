import streamlit as st

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ui.errors import render_exception


try:
    settings = get_settings()
    configure_logging(settings.log_level)

except Exception as error:
    st.set_page_config(
        page_title="KnowledgeHub",
        page_icon="📚",
        layout="wide",
    )

    st.title("KnowledgeHub")

    render_exception(
        error,
        operation="application startup",
    )

    st.stop()


st.set_page_config(
    page_title=settings.app_name,
    page_icon="📚",
    layout="wide",
)

st.title("KnowledgeHub")

st.write(
    """
A local RAG-powered knowledge management
and technical interview preparation system.

Built with:

- Streamlit
- LangChain
- Ollama
- PostgreSQL
- pgvector
"""
)

st.info(
    """
Use the sidebar to open:

- Knowledge Base — upload and index documents
- Chat — ask questions using your knowledge base
- Interview — generate and evaluate interview questions
"""
)