import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.core.config import Settings
from app.core.exceptions import (
    DocumentProcessingError,
    DocumentStorageError,
    KnowledgeHubError,
    UnsupportedDocumentError,
)
from app.core.validation import (
    validate_knowledge_base_id,
)
from app.database.vector_store import (
    VectorStoreService,
)
from app.parser.document_parser import DocumentParser


logger = logging.getLogger(__name__)


class IngestStatus(StrEnum):
    INDEXED = "indexed"
    DUPLICATE = "duplicate"


@dataclass(frozen=True, slots=True)
class IngestResult:
    status: IngestStatus
    filename: str
    document_id: str
    chunk_count: int


class KnowledgeBaseService:
    """
    Validate -> Stage file -> Parse -> Split ->
    Add metadata -> Store file -> Index
    """

    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStoreService,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.parser = DocumentParser()

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=(
                    settings.chunk_overlap
                ),
                separators=[
                    "\n\n",
                    "\n",
                    "。",
                    "！",
                    "？",
                    ". ",
                    "! ",
                    "? ",
                    " ",
                    "",
                ],
            )
        )

    def ingest_upload(
        self,
        *,
        filename: str,
        content: bytes,
        knowledge_base_id: str = "default",
    ) -> IngestResult:
        knowledge_base_id = (
            validate_knowledge_base_id(
                knowledge_base_id
            )
        )

        safe_filename = self._safe_filename(
            filename
        )

        extension = Path(
            safe_filename
        ).suffix.lower()

        if (
            extension
            not in self.parser.SUPPORTED_TYPES
        ):
            raise UnsupportedDocumentError(
                f"Unsupported upload type: {extension}",
                user_message=(
                    "Only TXT, Markdown, and PDF "
                    "documents are supported."
                ),
            )

        if not content:
            raise DocumentProcessingError(
                "Uploaded file contains zero bytes.",
                user_message=(
                    "The uploaded file is empty."
                ),
            )

        if (
            len(content)
            > self.settings.max_upload_bytes
        ):
            max_megabytes = (
                self.settings.max_upload_bytes
                // (1024 * 1024)
            )

            raise DocumentProcessingError(
                f"Upload contains {len(content)} bytes.",
                user_message=(
                    "The uploaded file is too large. "
                    f"The maximum size is "
                    f"{max_megabytes} MB."
                ),
            )

        root = self._prepare_document_root()
        knowledge_base_directory = (
            root / knowledge_base_id
        )

        staging_directory = (
            root / ".staging" / knowledge_base_id
        )

        temporary_path = (
            staging_directory
            / f"{uuid4().hex}{extension}"
        )

        document_id = hashlib.sha256(
            content
        ).hexdigest()

        final_directory = (
            knowledge_base_directory / document_id
        )

        final_path = (
            final_directory / safe_filename
        )

        created_final_file = False
        vector_write_started = False

        try:
            staging_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_path.write_bytes(content)

            if self.vector_store.document_exists(
                knowledge_base_id,
                document_id,
            ):
                return IngestResult(
                    status=IngestStatus.DUPLICATE,
                    filename=safe_filename,
                    document_id=document_id,
                    chunk_count=0,
                )

            documents = self.parser.parse(
                temporary_path,
                source_name=safe_filename,
            )

            chunks = self._split_documents(
                documents
            )

            self._add_metadata(
                chunks=chunks,
                knowledge_base_id=(
                    knowledge_base_id
                ),
                document_id=document_id,
            )

            final_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not final_path.exists():
                temporary_path.replace(final_path)
                created_final_file = True

            vector_write_started = True

            self.vector_store.add_documents(
                chunks
            )

            return IngestResult(
                status=IngestStatus.INDEXED,
                filename=safe_filename,
                document_id=document_id,
                chunk_count=len(chunks),
            )

        except KnowledgeHubError:
            self._rollback_failed_ingest(
                knowledge_base_id=(
                    knowledge_base_id
                ),
                document_id=document_id,
                final_path=final_path,
                created_final_file=(
                    created_final_file
                ),
                vector_write_started=(
                    vector_write_started
                ),
            )

            raise

        except OSError as exc:
            self._rollback_failed_ingest(
                knowledge_base_id=(
                    knowledge_base_id
                ),
                document_id=document_id,
                final_path=final_path,
                created_final_file=(
                    created_final_file
                ),
                vector_write_started=(
                    vector_write_started
                ),
            )

            raise DocumentStorageError(
                f"Unable to store {safe_filename}."
            ) from exc

        finally:
            try:
                temporary_path.unlink(
                    missing_ok=True
                )
            except OSError:
                logger.warning(
                    "Unable to remove staging file: %s",
                    temporary_path,
                    exc_info=True,
                )

    def _split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        try:
            chunks = (
                self.splitter.split_documents(
                    documents
                )
            )

        except Exception as exc:
            raise DocumentProcessingError(
                "Document splitting failed."
            ) from exc

        chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content.strip()
        ]

        if not chunks:
            raise DocumentProcessingError(
                "Document produced zero chunks.",
                user_message=(
                    "The document did not contain "
                    "searchable text."
                ),
            )

        return chunks

    @staticmethod
    def _add_metadata(
        *,
        chunks: list[Document],
        knowledge_base_id: str,
        document_id: str,
    ) -> None:
        for index, chunk in enumerate(chunks):
            chunk.metadata.update(
                {
                    "knowledge_base_id":
                        knowledge_base_id,
                    "document_id":
                        document_id,
                    "content_hash":
                        document_id,
                    "chunk_index":
                        index,
                }
            )

            chunk.metadata.setdefault(
                "page",
                None,
            )

    def _prepare_document_root(self) -> Path:
        try:
            root = (
                self.settings.document_root
                .expanduser()
                .resolve()
            )

            root.mkdir(
                parents=True,
                exist_ok=True,
            )

            return root

        except OSError as exc:
            raise DocumentStorageError(
                "Unable to prepare document root."
            ) from exc

    def _rollback_failed_ingest(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
        final_path: Path,
        created_final_file: bool,
        vector_write_started: bool,
    ) -> None:
        if vector_write_started:
            try:
                self.vector_store.delete_document(
                    knowledge_base_id=(
                        knowledge_base_id
                    ),
                    document_id=document_id,
                )

            except KnowledgeHubError:
                logger.exception(
                    "Unable to clean partial vectors "
                    "for document_id=%s",
                    document_id,
                )

        if created_final_file:
            try:
                final_path.unlink(
                    missing_ok=True
                )

                final_path.parent.rmdir()

            except OSError:
                logger.warning(
                    "Unable to clean stored file: %s",
                    final_path,
                    exc_info=True,
                )

    @staticmethod
    def _safe_filename(
        filename: str,
    ) -> str:
        normalized = filename.replace(
            "\\",
            "/",
        )

        safe_filename = PurePosixPath(
            normalized
        ).name.strip()

        if (
            not safe_filename
            or safe_filename in {".", ".."}
        ):
            raise DocumentStorageError(
                f"Invalid filename: {filename!r}",
                user_message=(
                    "The uploaded file has an "
                    "invalid filename."
                ),
            )

        return safe_filename