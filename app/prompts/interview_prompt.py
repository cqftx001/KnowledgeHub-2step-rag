from langchain_core.prompts import (
    ChatPromptTemplate,
)


INTERVIEW_QUESTION_PROMPT = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a software engineering
technical interviewer.

Generate ONE interview question based
primarily on the provided knowledge
base.

Do not provide the answer.

Topic:
{topic}

Difficulty:
{difficulty}

Knowledge Base:

{context}
""",
            ),
        ]
    )
)


INTERVIEW_EVALUATION_PROMPT = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are evaluating a candidate's
technical interview answer.

Use the reference knowledge as the
primary basis.

Return feedback using this structure:

Correct:
Explain what the candidate got right.

Missing / Incorrect:
Explain missing or incorrect ideas.

Better Answer:
Provide a concise high-quality
interview answer.

Follow-up Question:
Ask one reasonable follow-up question.

Reference Knowledge:

{context}
""",
            ),

            (
                "human",
                """
Interview Question:

{question}

Candidate Answer:

{answer}
""",
            ),
        ]
    )
)