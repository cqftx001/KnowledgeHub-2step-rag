# KnowledgeHub

A local RAG-powered knowledge management and technical interview
preparation system built with Streamlit, LangChain, Ollama,
PostgreSQL, and pgvector.

## Demo

### Document Upload & Indexing

![Document Upload Demo](./docs/assets/knowledge-base.gif)

### RAG Question Answering

![RAG Chat Demo](./docs/assets/rag-chat.gif)

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

## Tech Stack

- Streamlit
- LangChain
- Ollama
- PostgreSQL
- pgvector
