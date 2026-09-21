from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

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
from app.prompts.interview_prompt import (
    INTERVIEW_EVALUATION_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
)


class InterviewService:
    ALLOWED_DIFFICULTIES = {
        "Easy",
        "Medium",
        "Hard",
    }

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
        chat_model: BaseChatModel,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.chat_model = chat_model

    def generate_question(
        self,
        topic: str,
        difficulty: str,
        knowledge_base_id: str = "default",
    ) -> str:
        topic = validate_required_text(
            topic,
            field_name="Topic",
            max_length=200,
        )

        knowledge_base_id = (
            validate_knowledge_base_id(
                knowledge_base_id
            )
        )

        if (
            difficulty
            not in self.ALLOWED_DIFFICULTIES
        ):
            raise InputValidationError(
                f"Unsupported difficulty: {difficulty}"
            )

        documents = self.vector_store.search(
            query=(
                f"{topic} core concepts "
                "technical interview"
            ),
            knowledge_base_id=(
                knowledge_base_id
            ),
            k=self.settings.retrieval_top_k,
        )

        context = self._build_context(
            documents
        )

        messages = (
            INTERVIEW_QUESTION_PROMPT
            .format_messages(
                topic=topic,
                difficulty=difficulty,
                context=context,
            )
        )

        return self._invoke_model(messages)

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        knowledge_base_id: str = "default",
    ) -> str:
        question = validate_required_text(
            question,
            field_name="Interview question",
            max_length=5_000,
        )

        answer = validate_required_text(
            answer,
            field_name="Answer",
            max_length=10_000,
        )

        knowledge_base_id = (
            validate_knowledge_base_id(
                knowledge_base_id
            )
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

        messages = (
            INTERVIEW_EVALUATION_PROMPT
            .format_messages(
                question=question,
                answer=answer,
                context=context,
            )
        )

        return self._invoke_model(messages)

    def _invoke_model(
        self,
        messages: list[Any],
    ) -> str:
        try:
            response = self.chat_model.invoke(
                messages
            )

        except Exception as exc:
            raise ModelUnavailableError(
                "Ollama invocation failed."
            ) from exc

        text = self._extract_text(
            response.content
        )

        if not text:
            raise GenerationError(
                "The model returned empty content."
            )

        return text

    @staticmethod
    def _extract_text(
        content: Any,
    ) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []

            for item in content:
                if isinstance(item, str):
                    parts.append(item)

                elif isinstance(item, dict):
                    text = item.get("text")

                    if isinstance(text, str):
                        parts.append(text)

            return "\n".join(parts).strip()

        return ""

    @staticmethod
    def _build_context(
        documents: list[Document],
    ) -> str:
        if not documents:
            return (
                "No relevant knowledge was found."
            )

        return "\n\n".join(
            document.page_content
            for document in documents
        )