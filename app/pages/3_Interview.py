import streamlit as st

from app.core.dependencies import (
    get_interview_service,
)
from app.ui.errors import render_exception


st.title("Interview Mode")

knowledge_base_id = st.text_input(
    "Knowledge Base",
    value="default",
)

topic = st.text_input(
    "Topic",
    value="Java Concurrency",
)

difficulty = st.selectbox(
    "Difficulty",
    [
        "Easy",
        "Medium",
        "Hard",
    ],
)

if (
    "interview_question"
    not in st.session_state
):
    st.session_state.interview_question = None

if (
    "interview_feedback"
    not in st.session_state
):
    st.session_state.interview_feedback = None


if st.button(
    "Generate Question",
    type="primary",
):
    try:
        service = get_interview_service()

        with st.spinner(
            "Generating question..."
        ):
            question = (
                service.generate_question(
                    topic=topic,
                    difficulty=difficulty,
                    knowledge_base_id=(
                        knowledge_base_id
                    ),
                )
            )

    except Exception as error:
        render_exception(
            error,
            operation=(
                "interview question generation"
            ),
        )

    else:
        st.session_state.interview_question = (
            question
        )

        st.session_state.interview_feedback = (
            None
        )


if st.session_state.interview_question:
    st.subheader("Interview Question")

    st.markdown(
        st.session_state.interview_question
    )

    answer = st.text_area(
        "Your Answer",
        height=200,
    )

    if st.button("Submit Answer"):
        try:
            service = get_interview_service()

            with st.spinner("Evaluating..."):
                feedback = (
                    service.evaluate_answer(
                        question=(
                            st.session_state
                            .interview_question
                        ),
                        answer=answer,
                        knowledge_base_id=(
                            knowledge_base_id
                        ),
                    )
                )

        except Exception as error:
            render_exception(
                error,
                operation=(
                    "interview answer evaluation"
                ),
            )

        else:
            st.session_state.interview_feedback = (
                feedback
            )


if st.session_state.interview_feedback:
    st.subheader("Evaluation")

    st.markdown(
        st.session_state.interview_feedback
    )