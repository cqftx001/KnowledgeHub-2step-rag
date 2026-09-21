from langchain_core.embeddings import (
    Embeddings,
)
from langchain_ollama import (
    OllamaEmbeddings,
)

from app.core.config import Settings


def create_embedding_model(
    settings: Settings,
) -> Embeddings:
    """
    Create the embedding model used for
    both indexing and retrieval.
    """

    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,

        # Must match PGVector table dimension.
        dimensions=settings.embedding_dimensions,
    )