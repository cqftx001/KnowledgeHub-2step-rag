from collections.abc import Iterator

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)
from langchain_core.messages import BaseMessage

from app.core.config import Settings
from app.core.exceptions import (
    GenerationError,
    InputValidationError,
    ModelUnavailableError,
)
from app.core.validation import (
    validate_knowledge_base_id,
    validate_required_text,
)
from app.database.vector_store import (
    VectorStoreService,
)
from app.prompts.rag_prompt import (
    MODE_INSTRUCTIONS,
    RAG_PROMPT,
    get_mode_instruction,
)


class RagService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        chat_model: BaseChatModel,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.chat_model = chat_model

    def ask(
        self,
        question: str,
        knowledge_base_id: str = "default",
        mode: str = "qa",
        history: list[BaseMessage] | None = None,
    ) -> tuple[
        Iterator[str],
        list[Document],
    ]:
        question = validate_required_text(
            question,
            field_name="Question",
            max_length=5_000,
        )

        knowledge_base_id = (
            validate_knowledge_base_id(
                knowledge_base_id
            )
        )

        if mode not in MODE_INSTRUCTIONS:
            raise InputValidationError(
                f"Unsupported RAG mode: {mode}"
            )

        documents = self.vector_store.search(
            query=question,
            knowledge_base_id=(
                knowledge_base_id
            ),
            k=self.settings.retrieval_top_k,
        )

        context = self._build_context(
            documents
        )

        messages = RAG_PROMPT.format_messages(
            mode_instruction=(
                get_mode_instruction(mode)
            ),
            context=context,
            history=history or [],
            question=question,
        )

        def generate() -> Iterator[str]:
            generated_text = False

            try:
                for chunk in (
                    self.chat_model.stream(
                        messages
                    )
                ):
                    content = chunk.content

                    if (
                        isinstance(content, str)
                        and content
                    ):
                        generated_text = True
                        yield content

            except Exception as exc:
                raise ModelUnavailableError(
                    "Ollama streaming failed."
                ) from exc

            if not generated_text:
                raise GenerationError(
                    "The model stream returned no text."
                )

        return generate(), documents

    @staticmethod
    def _build_context(
        documents: list[Document],
    ) -> str:
        if not documents:
            return (
                "No relevant knowledge was found."
            )

        sections: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            source = document.metadata.get(
                "source",
                "unknown",
            )

            page = document.metadata.get(
                "page"
            )

            source_info = (
                f"Source {index}: {source}"
            )

            if page is not None:
                source_info += (
                    f", page {page + 1}"
                )

            sections.append(
                (
                    f"[{source_info}]\n\n"
                    f"{document.page_content}"
                )
            )

        return "\n\n".join(sections)