from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

MODE_INSTRUCTIONS = {

    "qa": """
Answer the user's question using the
knowledge base context.

Explain clearly and directly.
""",

    "summary": """
Summarize the relevant knowledge.

Organize the answer into:
1. Main idea
2. Important details
3. Key takeaways
""",

    "concept": """
Explain the concept as a structured
learning framework.

Organize the answer into:
1. Definition
2. Core mechanism
3. Important concepts
4. Relationships
5. Example
6. Common interview points
""",
}

RAG_PROMPT = (
    ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a personal knowledge assistant.

Use the provided knowledge base
context as the primary source.

If the context does not contain enough
information, explicitly say so.

Do not fabricate facts.

Mode instruction:

{mode_instruction}

Knowledge Base Context:

{context}
""",
            ),

            MessagesPlaceholder(
                variable_name="history"
            ),

            (
                "human",
                "{question}",
            ),
        ]
    )
)


def get_mode_instruction(
    mode: str,
) -> str:

    return MODE_INSTRUCTIONS.get(
        mode,
        MODE_INSTRUCTIONS["qa"],
    )