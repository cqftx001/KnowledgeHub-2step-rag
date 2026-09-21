from threading import Lock
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import (
    Column,
    PGEngine,
    PGVectorStore,
)
from sqlalchemy import create_engine, inspect

from app.core.config import Settings
from app.core.exceptions import (
    VectorStoreError,
    VectorStoreUnavailableError,
)


class VectorStoreService:
    METADATA_COLUMNS = [
        "knowledge_base_id",
        "document_id",
        "source",
        "file_type",
        "content_hash",
        "chunk_index",
        "page",
    ]

    def __init__(
        self,
        settings: Settings,
        embedding_model: Embeddings,
    ) -> None:
        self.settings = settings
        self.embedding_model = embedding_model

        self._engine: PGEngine | None = None
        self._store: PGVectorStore | None = None
        self._initialization_lock = Lock()

    def _get_store(self) -> PGVectorStore:
        """
        Lazily initialize the database.

        Opening a Streamlit page no longer immediately
        connects to PostgreSQL or creates tables.
        """

        if self._store is not None:
            return self._store

        with self._initialization_lock:
            if self._store is not None:
                return self._store

            self._initialize()

        if self._store is None:
            raise VectorStoreUnavailableError(
                "Vector store initialization returned no store."
            )

        return self._store

    def _initialize(self) -> None:
        database_url = (
            self.settings.postgres_url.get_secret_value()
        )

        check_engine = None

        try:
            engine = PGEngine.from_connection_string(
                url=database_url
            )

            check_engine = create_engine(
                database_url,
                pool_pre_ping=True,
            )

            table_exists = inspect(
                check_engine
            ).has_table(
                self.settings.vector_table_name,
                schema=self.settings.vector_schema,
            )

            if not table_exists:
                engine.init_vectorstore_table(
                    table_name=(
                        self.settings.vector_table_name
                    ),
                    schema_name=(
                        self.settings.vector_schema
                    ),
                    vector_size=(
                        self.settings.embedding_dimensions
                    ),
                    metadata_columns=[
                        Column(
                            "knowledge_base_id",
                            "VARCHAR(100)",
                            nullable=False,
                        ),
                        Column(
                            "document_id",
                            "VARCHAR(64)",
                            nullable=False,
                        ),
                        Column(
                            "source",
                            "VARCHAR(512)",
                            nullable=False,
                        ),
                        Column(
                            "file_type",
                            "VARCHAR(32)",
                            nullable=False,
                        ),
                        Column(
                            "content_hash",
                            "VARCHAR(64)",
                            nullable=False,
                        ),
                        Column(
                            "chunk_index",
                            "INTEGER",
                            nullable=False,
                        ),
                        Column(
                            "page",
                            "INTEGER",
                            nullable=True,
                        ),
                    ],
                    store_metadata=True,
                )

            store = PGVectorStore.create_sync(
                engine=engine,
                table_name=(
                    self.settings.vector_table_name
                ),
                schema_name=(
                    self.settings.vector_schema
                ),
                embedding_service=self.embedding_model,
                metadata_columns=self.METADATA_COLUMNS,
            )

            self._engine = engine
            self._store = store

        except Exception as exc:
            raise VectorStoreUnavailableError(
                "Unable to initialize PostgreSQL "
                "vector storage."
            ) from exc

        finally:
            if check_engine is not None:
                check_engine.dispose()

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        if not documents:
            raise VectorStoreError(
                "Cannot add an empty document list."
            )

        ids = [
            str(uuid4())
            for _ in documents
        ]

        try:
            return self._get_store().add_documents(
                documents=documents,
                ids=ids,
            )

        except VectorStoreError:
            raise

        except Exception as exc:
            raise VectorStoreError(
                "Failed to add document chunks."
            ) from exc

    def search(
        self,
        query: str,
        knowledge_base_id: str,
        k: int,
    ) -> list[Document]:
        try:
            return self._get_store().similarity_search(
                query=query,
                k=k,
                filter={
                    "knowledge_base_id":
                        knowledge_base_id
                },
            )

        except VectorStoreError:
            raise

        except Exception as exc:
            raise VectorStoreError(
                "Vector similarity search failed."
            ) from exc

    def document_exists(
        self,
        knowledge_base_id: str,
        content_hash: str,
    ) -> bool:
        try:
            result = self._get_store().get(
                where={
                    "$and": [
                        {
                            "knowledge_base_id":
                                knowledge_base_id
                        },
                        {
                            "content_hash":
                                content_hash
                        },
                    ]
                },
                limit=1,
            )

            return bool(
                (result or {}).get(
                    "metadatas",
                    [],
                )
            )

        except VectorStoreError:
            raise

        except Exception as exc:
            raise VectorStoreError(
                "Duplicate document check failed."
            ) from exc

    def delete_document(
        self,
        *,
        knowledge_base_id: str,
        document_id: str,
    ) -> None:
        try:
            self._get_store().delete(
                filter={
                    "$and": [
                        {
                            "knowledge_base_id":
                                knowledge_base_id
                        },
                        {
                            "document_id":
                                document_id
                        },
                    ]
                }
            )

        except VectorStoreError:
            raise

        except Exception as exc:
            raise VectorStoreError(
                "Document cleanup failed."
            ) from exc