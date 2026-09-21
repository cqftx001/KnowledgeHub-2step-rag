import streamlit as st
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)

from app.core.dependencies import get_rag_service
from app.ui.errors import render_exception


st.title("Knowledge Chat")

knowledge_base_id = st.text_input(
    "Knowledge Base",
    value="default",
)

mode_map = {
    "Q&A": "qa",
    "Summary": "summary",
    "Concept": "concept",
}

selected_mode = st.selectbox(
    "Mode",
    options=list(mode_map),
)

mode = mode_map[selected_mode]

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if st.button("Clear Conversation"):
    st.session_state.chat_messages = []
    st.rerun()


for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources")

        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(source)


question = st.chat_input(
    "Ask your knowledge base..."
)

if question:
    history = []

    for message in (
        st.session_state.chat_messages
    ):
        if message["role"] == "user":
            history.append(
                HumanMessage(
                    content=message["content"]
                )
            )

        elif message["role"] == "assistant":
            history.append(
                AIMessage(
                    content=message["content"]
                )
            )

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            service = get_rag_service()

            stream, documents = service.ask(
                question=question,
                knowledge_base_id=(
                    knowledge_base_id
                ),
                mode=mode,
                history=history,
            )

            # Model failures can occur here while the
            # generator is being consumed.
            answer = st.write_stream(stream)

            sources: list[str] = []

            for document in documents:
                source = document.metadata.get(
                    "source",
                    "unknown",
                )

                page = document.metadata.get(
                    "page"
                )

                label = source

                if page is not None:
                    label += (
                        f" — page {page + 1}"
                    )

                if label not in sources:
                    sources.append(label)

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(source)

        except Exception as error:
            render_exception(
                error,
                operation="knowledge base chat",
            )

        else:
            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )