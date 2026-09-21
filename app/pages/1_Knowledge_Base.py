import streamlit as st

from app.core.dependencies import (
    get_knowledge_base_service,
)
from app.services.knowledge_base_service import (
    IngestStatus,
)
from app.ui.errors import render_exception


st.title("Knowledge Base")

knowledge_base_id = st.text_input(
    "Knowledge Base",
    value="default",
)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=[
        "txt",
        "md",
        "pdf",
    ],
    accept_multiple_files=True,
)

if st.button(
    "Add to Knowledge Base",
    type="primary",
):
    if not uploaded_files:
        st.warning(
            "Please upload at least one file."
        )

    else:
        try:
            service = (
                get_knowledge_base_service()
            )

        except Exception as error:
            render_exception(
                error,
                operation=(
                    "knowledge base initialization"
                ),
            )

            st.stop()

        for uploaded_file in uploaded_files:
            try:
                with st.spinner(
                    f"Indexing "
                    f"{uploaded_file.name}..."
                ):
                    result = (
                        service.ingest_upload(
                            filename=(
                                uploaded_file.name
                            ),
                            content=(
                                uploaded_file.getvalue()
                            ),
                            knowledge_base_id=(
                                knowledge_base_id
                            ),
                        )
                    )

            except Exception as error:
                render_exception(
                    error,
                    operation=(
                        f"indexing "
                        f"{uploaded_file.name}"
                    ),
                )

                continue

            if (
                result.status
                is IngestStatus.DUPLICATE
            ):
                st.warning(
                    f"{result.filename}: "
                    "already indexed."
                )

            else:
                st.success(
                    f"{result.filename}: "
                    f"{result.chunk_count} "
                    "chunks indexed."
                )