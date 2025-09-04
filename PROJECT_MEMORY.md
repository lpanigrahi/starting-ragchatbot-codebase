# Project Memory - Course Materials RAG System

## Quick Reference

### Development Commands
- **Start server**: `./run.sh` or `cd backend && uv run uvicorn app:app --reload --port 8000`
- **Install dependencies**: `uv sync`
- **IMPORTANT**: Always use `uv` for Python package management, never use `pip` directly

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Environment Setup
- Copy `.env.example` to `.env`
- Add `ANTHROPIC_API_KEY=your-key-here` to `.env` file

## Architecture Summary

### Core System Type
**Retrieval-Augmented Generation (RAG) System** for course materials with conversational AI interface.

### Technology Stack
- **Backend**: FastAPI with Python 3.13+
- **Package Manager**: uv (required - do not use pip)
- **AI Model**: Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Vector Database**: ChromaDB with `all-MiniLM-L6-v2` embeddings
- **Frontend**: Static HTML/CSS/JS served by FastAPI

### Key Components

#### Backend Structure (`backend/`)
- `app.py` - FastAPI server with `/api/query` and `/api/courses` endpoints
- `rag_system.py` - Main orchestrator coordinating all RAG components
- `document_processor.py` - Processes course docs into searchable chunks
- `vector_store.py` - ChromaDB integration for semantic search
- `ai_generator.py` - Anthropic Claude API integration
- `search_tools.py` - Tool-based search with function calling
- `session_manager.py` - Conversation history management
- `models.py` - Pydantic models (Course, Lesson, CourseChunk)
- `config.py` - Configuration with environment variables

#### Frontend Structure (`frontend/`)
- `index.html` - Main chat interface
- `script.js` - Query handling and API communication
- `style.css` - UI styling

#### Data Directory
- `docs/` - Course materials (auto-processed on startup)
- Currently contains 4 course transcript files

### Document Processing Pipeline

1. **Structure Parsing**: Extracts course metadata (title, instructor, lessons)
2. **Text Chunking**: Intelligent sentence-based splitting (800 chars, 100 overlap)
3. **Context Enhancement**: Adds course/lesson context to each chunk
4. **Vector Storage**: Embeds chunks in ChromaDB for semantic search

### Query Flow Architecture

```
User Query → Frontend → FastAPI → RAG System → AI Generator → Claude API
                                      ↓
                                 Tool Manager → Vector Store → ChromaDB
```

### Key Configuration Settings
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation history length
- `CHROMA_PATH: "./chroma_db"` - Vector database location

### Important Development Notes

1. **Package Management**: Must use `uv` for all Python operations
2. **Document Loading**: System auto-loads documents from `/docs` on startup
3. **Session Management**: Maintains conversation context across queries
4. **Tool-Based Search**: Claude decides when to search course content
5. **Source Citations**: Automatic source tracking and display
6. **Error Handling**: Graceful fallbacks for missing API keys or documents

### API Endpoints
- `POST /api/query` - Process user queries with RAG
- `GET /api/courses` - Get course statistics and titles

### Startup Process
1. Creates necessary directories
2. Loads course documents into ChromaDB
3. Initializes RAG system components
4. Starts FastAPI server with frontend

### Common Development Tasks
- Add new courses: Drop files in `/docs` directory
- Modify AI behavior: Edit system prompt in `ai_generator.py`
- Adjust chunking: Update config values in `config.py`
- Frontend changes: Edit files in `/frontend` directory

### Dependencies (managed via uv)
- chromadb==1.0.15
- anthropic==0.58.2
- sentence-transformers==5.0.0
- fastapi==0.116.1
- uvicorn==0.35.0
- python-multipart==0.0.20
- python-dotenv==1.1.1