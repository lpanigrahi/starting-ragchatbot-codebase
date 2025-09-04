# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Key Commands

### Development and Running
- **Start the application**: `./run.sh` (or manually: `cd backend && uv run uvicorn app:app --reload --port 8000`)
- **Install dependencies**: `uv sync`
- **Environment setup**: Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- **Code quality checks**: `uv run python scripts/format.py` - runs Black, isort, flake8, and mypy
- **Format code**: `uv run black backend/ main.py` - automatic code formatting
- **Sort imports**: `uv run isort backend/ main.py` - organize import statements

### Access Points
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture Overview

This is a **Course Materials RAG System** - a Retrieval-Augmented Generation application that enables semantic search and AI-powered responses over course documents.

### Core Architecture Pattern
The system follows a **modular RAG architecture** with clear separation of concerns:

```
Frontend (HTML/CSS/JS) → FastAPI Backend → RAG Components → Vector Store & AI
```

### Key Backend Components

**Central Orchestrator**:
- `rag_system.py` - Main RAG orchestrator that coordinates all components
- `app.py` - FastAPI application with API endpoints (`/api/query`, `/api/courses`)

**Core RAG Pipeline**:
- `document_processor.py` - Processes course documents and creates text chunks
- `vector_store.py` - ChromaDB integration for semantic search
- `ai_generator.py` - Anthropic Claude integration for response generation
- `search_tools.py` - Tool-based search functionality with function calling

**Supporting Systems**:
- `session_manager.py` - Manages conversation history and context
- `models.py` - Pydantic models for Course, Lesson, CourseChunk
- `config.py` - Configuration management with environment variables

### Data Flow
1. Course documents in `/docs` are processed into chunks at startup
2. User queries trigger semantic search in ChromaDB vector store
3. Retrieved context + conversation history sent to Claude API
4. AI response with source citations returned to frontend

### Key Technologies
- **Vector Store**: ChromaDB with `all-MiniLM-L6-v2` embeddings
- **AI Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Backend**: FastAPI with async endpoints
- **Frontend**: Static HTML/CSS/JS served by FastAPI
- **Dependency Management**: uv for Python packages

### Configuration
All configuration is centralized in `config.py` with these key settings:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation history length

The system auto-loads documents from the `/docs` directory on startup and supports real-time querying with session-based conversation tracking.
- make sure uv to manage all dependencies
- don't run the server using ./run.sh , I will do it myself