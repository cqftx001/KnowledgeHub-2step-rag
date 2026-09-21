import re

from app.core.exceptions import InputValidationError


_KNOWLEDGE_BASE_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
)


def validate_knowledge_base_id(
    knowledge_base_id: str,
) -> str:
    value = knowledge_base_id.strip()

    if not _KNOWLEDGE_BASE_ID_PATTERN.fullmatch(value):
        raise InputValidationError(
            f"Invalid knowledge base ID: {value!r}",
            user_message=(
                "Knowledge Base must contain only letters, "
                "numbers, underscores, and hyphens."
            ),
        )

    return value


def validate_required_text(
    value: str,
    *,
    field_name: str,
    max_length: int = 10_000,
) -> str:
    normalized = value.strip()

    if not normalized:
        raise InputValidationError(
            f"{field_name} is empty.",
            user_message=f"{field_name} cannot be empty.",
        )

    if len(normalized) > max_length:
        raise InputValidationError(
            f"{field_name} exceeds {max_length} characters.",
            user_message=(
                f"{field_name} is too long. "
                f"The maximum length is {max_length} characters."
            ),
        )

    return normalized