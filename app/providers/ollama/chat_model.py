from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_ollama import ChatOllama
from app.core.config import Settings


def create_chat_model(
    settings: Settings,
) -> BaseChatModel:
    """
    Create the Ollama chat model used by
    RAG and Interview mode.
    """

    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,

        # Keep RAG answers relatively stable.
        temperature=0.2,
    )