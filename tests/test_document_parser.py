from pathlib import Path

import pytest

from app.core.exceptions import (
    EmptyDocumentError,
    UnsupportedDocumentError,
)
from app.parser.document_parser import (
    DocumentParser,
)


def test_parse_text_document(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "example.txt"
    file_path.write_text(
        "Operating systems manage resources.",
        encoding="utf-8",
    )

    parser = DocumentParser()

    documents = parser.parse(file_path)

    assert len(documents) == 1
    assert (
        documents[0].page_content
        == "Operating systems manage resources."
    )
    assert (
        documents[0].metadata["source"]
        == "example.txt"
    )
    assert (
        documents[0].metadata["file_type"]
        == "txt"
    )


def test_empty_document_is_rejected(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "empty.md"
    file_path.write_text(
        "   ",
        encoding="utf-8",
    )

    parser = DocumentParser()

    with pytest.raises(
        EmptyDocumentError
    ):
        parser.parse(file_path)


def test_unsupported_document_is_rejected(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "example.csv"
    file_path.write_text(
        "name,value",
        encoding="utf-8",
    )

    parser = DocumentParser()

    with pytest.raises(
        UnsupportedDocumentError
    ):
        parser.parse(file_path)