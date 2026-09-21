# KnowledgeHub

KnowledgeHub is a local RAG-powered knowledge management and
technical interview preparation application.

Users can upload documents, search them through semantic retrieval,
ask knowledge-grounded questions, and practice technical interviews.

## Features

- TXT, Markdown, and PDF document ingestion
- Recursive document chunking
- Ollama embeddings and chat generation
- PostgreSQL and pgvector semantic search
- SHA-256 document deduplication
- Source-aware RAG answers
- Q&A, summary, and concept explanation modes
- AI-generated technical interview questions
- Knowledge-grounded answer evaluation
- Structured error handling and safe UI messages

## Architecture

```text
Streamlit UI
    |
Application Services
    |
Document Parser / Ollama / PGVector
    |
PostgreSQL + pgvector
