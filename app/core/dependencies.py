from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from app.core.config import (
    get_settings,
)
from app.database.vector_store import (
    VectorStoreService,
)
from app.providers.ollama.chat_model import (
    create_chat_model,
)
from app.providers.ollama.embedding_model import (
    create_embedding_model,
)
from app.services.interview_service import (
    InterviewService,
)
from app.services.knowledge_base_service import (
    KnowledgeBaseService,
)
from app.services.rag_service import RagService


@lru_cache
def get_chat_model() -> BaseChatModel:
    return create_chat_model(
        get_settings()
    )


@lru_cache
def get_embedding_model() -> Embeddings:
    return create_embedding_model(
        get_settings()
    )


@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService(
        settings=get_settings(),
        embedding_model=(
            get_embedding_model()
        ),
    )


@lru_cache
def get_knowledge_base_service(
) -> KnowledgeBaseService:
    return KnowledgeBaseService(
        settings=get_settings(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_rag_service() -> RagService:
    return RagService(
        settings=get_settings(),
        vector_store=get_vector_store(),
        chat_model=get_chat_model(),
    )


@lru_cache
def get_interview_service(
) -> InterviewService:
    return InterviewService(
        settings=get_settings(),
        vector_store=get_vector_store(),
        chat_model=get_chat_model(),
    )


def clear_dependency_caches() -> None:
    """
    Useful in tests and after configuration changes.
    """

    get_interview_service.cache_clear()
    get_rag_service.cache_clear()
    get_knowledge_base_service.cache_clear()
    get_vector_store.cache_clear()
    get_embedding_model.cache_clear()
    get_chat_model.cache_clear()
    get_settings.cache_clear()