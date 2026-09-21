import logging
from uuid import uuid4

import streamlit as st

from app.core.exceptions import KnowledgeHubError


logger = logging.getLogger(__name__)


def render_exception(
    error: Exception,
    *,
    operation: str,
) -> str:
    """
    Log technical details and display a safe message.

    Returns a short error reference ID.
    """

    error_id = uuid4().hex[:8]

    if isinstance(error, KnowledgeHubError):
        logger.warning(
            "%s failed. error_id=%s code=%s",
            operation,
            error_id,
            error.code,
            exc_info=True,
        )

        st.error(error.user_message)

        if error.retryable:
            st.info(
                "Check PostgreSQL and Ollama, "
                "then try the operation again."
            )

    else:
        logger.exception(
            "Unexpected error during %s. error_id=%s",
            operation,
            error_id,
        )

        st.error(
            "An unexpected error occurred. "
            f"Reference: {error_id}"
        )

    return error_id