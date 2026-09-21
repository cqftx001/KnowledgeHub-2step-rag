from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
)
from langchain_core.documents import Document

from app.core.exceptions import (
    DocumentReadError,
    EmptyDocumentError,
    KnowledgeHubError,
    UnsupportedDocumentError,
)


class DocumentParser:
    """
    Parse supported documents into LangChain Documents.
    """

    SUPPORTED_TYPES = {
        ".txt",
        ".md",
        ".pdf",
    }

    def parse(
        self,
        file_path: Path,
        *,
        source_name: str | None = None,
    ) -> list[Document]:
        extension = file_path.suffix.lower()

        if extension not in self.SUPPORTED_TYPES:
            raise UnsupportedDocumentError(
                f"Unsupported extension: {extension}",
                user_message=(
                    f"Files ending in {extension or '(none)'} "
                    "are not supported."
                ),
            )

        if not file_path.is_file():
            raise DocumentReadError(
                f"Document does not exist: {file_path}"
            )

        display_name = source_name or file_path.name

        try:
            if extension == ".pdf":
                documents = self._parse_pdf(
                    file_path,
                    source_name=display_name,
                )
            else:
                documents = self._parse_text(
                    file_path,
                    source_name=display_name,
                    file_type=(
                        "markdown"
                        if extension == ".md"
                        else "txt"
                    ),
                )

        except KnowledgeHubError:
            raise

        except (OSError, UnicodeError) as exc:
            raise DocumentReadError(
                f"Unable to read {file_path}."
            ) from exc

        except Exception as exc:
            raise DocumentReadError(
                f"Unable to parse {file_path}."
            ) from exc

        readable_documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not readable_documents:
            raise EmptyDocumentError(
                f"No readable text found in {file_path}."
            )

        return readable_documents

    @staticmethod
    def _parse_text(
        file_path: Path,
        *,
        source_name: str,
        file_type: str,
    ) -> list[Document]:
        content = file_path.read_text(
            encoding="utf-8-sig"
        ).strip()

        if not content:
            raise EmptyDocumentError(
                f"Document is empty: {file_path}"
            )

        return [
            Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "file_type": file_type,
                    "page": None,
                },
            )
        ]

    @staticmethod
    def _parse_pdf(
        file_path: Path,
        *,
        source_name: str,
    ) -> list[Document]:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        for document in documents:
            document.metadata["source"] = source_name
            document.metadata["file_type"] = "pdf"

        return documents