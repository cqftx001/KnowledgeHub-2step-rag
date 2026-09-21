class KnowledgeHubError(Exception):
    """
    Base exception for all expected KnowledgeHub errors.

    message:
        Technical message written to application logs.

    user_message:
        Safe message displayed in the UI.
    """

    code = "KNOWLEDGEHUB_ERROR"
    default_user_message = "The operation could not be completed."
    retryable = False

    def __init__(
        self,
        message: str | None = None,
        *,
        user_message: str | None = None,
    ) -> None:
        self.user_message = (
            user_message or self.default_user_message
        )

        super().__init__(
            message or self.user_message
        )


class ConfigurationError(KnowledgeHubError):
    code = "CONFIGURATION_ERROR"
    default_user_message = (
        "KnowledgeHub is not configured correctly."
    )


class InputValidationError(KnowledgeHubError):
    code = "INVALID_INPUT"
    default_user_message = (
        "One or more input values are invalid."
    )


class DocumentError(KnowledgeHubError):
    code = "DOCUMENT_ERROR"
    default_user_message = (
        "The document could not be processed."
    )


class UnsupportedDocumentError(DocumentError):
    code = "UNSUPPORTED_DOCUMENT"
    default_user_message = (
        "This document type is not supported."
    )


class EmptyDocumentError(DocumentError):
    code = "EMPTY_DOCUMENT"
    default_user_message = (
        "The document does not contain readable text."
    )


class DocumentReadError(DocumentError):
    code = "DOCUMENT_READ_ERROR"
    default_user_message = (
        "The document could not be read."
    )


class DocumentStorageError(DocumentError):
    code = "DOCUMENT_STORAGE_ERROR"
    default_user_message = (
        "The document could not be saved."
    )


class DocumentProcessingError(DocumentError):
    code = "DOCUMENT_PROCESSING_ERROR"
    default_user_message = (
        "The document could not be split into searchable content."
    )


class VectorStoreError(KnowledgeHubError):
    code = "VECTOR_STORE_ERROR"
    default_user_message = (
        "The knowledge database operation failed."
    )


class VectorStoreUnavailableError(VectorStoreError):
    code = "VECTOR_STORE_UNAVAILABLE"
    default_user_message = (
        "The knowledge database is currently unavailable."
    )
    retryable = True


class ModelProviderError(KnowledgeHubError):
    code = "MODEL_PROVIDER_ERROR"
    default_user_message = (
        "The AI model operation failed."
    )


class ModelUnavailableError(ModelProviderError):
    code = "MODEL_UNAVAILABLE"
    default_user_message = (
        "The local AI model is currently unavailable."
    )
    retryable = True


class GenerationError(ModelProviderError):
    code = "GENERATION_ERROR"
    default_user_message = (
        "The AI model did not produce a valid response."
    )