# AI Agent Plugin Platform

## Overview

This project is a plugin-based agent execution platform built with FastAPI and LangGraph.

## Project Layout

```text
agent_platform/
├── main.py                  # FastAPI application entrypoint
├── config/
│   └── settings.py          # Environment-driven configuration
├── core/
│   ├── database.py          # Database engine and schema migrations
│   ├── executor.py          # Tool execution engine
│   ├── graph.py             # LangGraph workflow
│   └── registry.py          # Tool registry
├── memory/
│   ├── short_term.py        # Short-term memory via LangGraph checkpointing
│   └── long_term.py         # Long-term memory via FAISS + Postgres metadata
├── api/
│   ├── routes.py            # HTTP routes
│   ├── schemas.py           # Request/response models
│   └── task_store.py        # Persistent task status store
├── services/
│   ├── agent_service.py     # Agent orchestration
│   └── task_runner.py       # Persistent async task worker
└── tools/                   # Auto-discovered tool modules
```

## Behavior

- `POST /api/execute` runs a task synchronously.
- `POST /api/execute/stream` streams task updates over SSE.
- `POST /api/execute/async` persists the task as `pending` and lets the startup worker pick it up.
- Short-term memory uses the LangGraph checkpointer configured in `core/graph.py`.
- Long-term memory is cached in-process and reuses the FAISS index instead of rebuilding it on every request.

## Configuration

Create a `.env` file with real values for your environment:

```bash
BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/agent_platform
CHECKPOINT_DATABASE_URL=postgresql://user:password@localhost:5432/agent_platform
JWT_SECRET_KEY=use-a-long-random-secret
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

In production, set `CORS_ORIGINS` explicitly. Wildcard CORS is intentionally not enabled.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## API Docs

Open `/docs` after the server starts.
