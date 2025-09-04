# RAG System Query Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend<br/>(script.js)
    participant API as FastAPI<br/>(app.py)
    participant RAG as RAG System<br/>(rag_system.py)
    participant SM as Session Manager<br/>(session_manager.py)
    participant AI as AI Generator<br/>(ai_generator.py)
    participant TM as Tool Manager<br/>(search_tools.py)
    participant VS as Vector Store<br/>(vector_store.py)
    participant Claude as Anthropic<br/>Claude API

    %% User initiates query
    U->>F: Types question & clicks send
    F->>F: Validate input & show loading
    F->>F: Add user message to chat

    %% Frontend to Backend
    F->>+API: POST /api/query<br/>{query, session_id}
    API->>API: Validate QueryRequest model
    API->>API: Create session if needed

    %% RAG System coordination
    API->>+RAG: query(user_query, session_id)
    RAG->>RAG: Format prompt for AI
    
    %% Get conversation history
    RAG->>+SM: get_conversation_history(session_id)
    SM-->>-RAG: Previous messages context

    %% AI Generation with tools
    RAG->>+AI: generate_response(query, history, tools)
    AI->>AI: Build system prompt + context
    AI->>+Claude: API call with tools enabled
    
    %% Claude decides to use search tool
    Claude->>Claude: Analyze query - needs course search
    Claude->>+TM: search_course_content(query, filters)
    
    %% Vector search execution
    TM->>+VS: search_similar_chunks(embedded_query)
    VS->>VS: Semantic search in ChromaDB
    VS-->>-TM: Relevant course chunks + metadata
    TM->>TM: Store sources for later retrieval
    TM-->>-Claude: Formatted search results

    %% Claude generates final response
    Claude->>Claude: Synthesize search results
    Claude-->>-AI: Generated response text
    AI-->>-RAG: Response string

    %% Get sources and update history
    RAG->>+TM: get_last_sources()
    TM-->>-RAG: Source citations list
    RAG->>+SM: add_exchange(session_id, query, response)
    SM-->>-RAG: Updated conversation history
    
    RAG-->>-API: (response, sources)

    %% Backend to Frontend
    API->>API: Create QueryResponse model
    API-->>-F: JSON response<br/>{answer, sources, session_id}

    %% Frontend displays response
    F->>F: Remove loading animation
    F->>F: Render AI response (markdown)
    F->>F: Add collapsible sources section
    F->>F: Update session ID & re-enable input
    F->>U: Display complete response

    %% Background: Document loading (startup)
    Note over VS: Documents loaded at startup:<br/>docs/ → chunks → ChromaDB embeddings
```

## Architecture Overview

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Web Interface<br/>HTML/CSS/JS]
    end

    subgraph "API Layer"
        FastAPI[FastAPI Server<br/>app.py]
    end

    subgraph "RAG Orchestration"
        RAG_SYS[RAG System<br/>rag_system.py]
        SESSION[Session Manager<br/>session_manager.py]
    end

    subgraph "AI Processing"
        AI_GEN[AI Generator<br/>ai_generator.py]
        CLAUDE[Anthropic<br/>Claude API]
    end

    subgraph "Search & Retrieval"
        TOOLS[Tool Manager<br/>search_tools.py]
        VECTOR[Vector Store<br/>vector_store.py]
        CHROMA[(ChromaDB<br/>Embeddings)]
    end

    subgraph "Document Processing"
        DOC_PROC[Document Processor<br/>document_processor.py]
        DOCS[Course Documents<br/>docs/]
    end

    %% Connections
    UI <--> FastAPI
    FastAPI <--> RAG_SYS
    RAG_SYS <--> SESSION
    RAG_SYS <--> AI_GEN
    AI_GEN <--> CLAUDE
    AI_GEN <--> TOOLS
    TOOLS <--> VECTOR
    VECTOR <--> CHROMA
    DOC_PROC --> VECTOR
    DOCS --> DOC_PROC

    %% Styling
    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef rag fill:#fff3e0
    classDef ai fill:#e8f5e8
    classDef search fill:#fce4ec
    classDef data fill:#f1f8e9

    class UI frontend
    class FastAPI api
    class RAG_SYS,SESSION rag
    class AI_GEN,CLAUDE ai
    class TOOLS,VECTOR,CHROMA search
    class DOC_PROC,DOCS data
```

## Data Flow Summary

1. **User Query** → Frontend validation & UI updates
2. **HTTP Request** → FastAPI endpoint with session management
3. **RAG Orchestration** → Coordinates components & manages context
4. **AI Processing** → Claude analyzes query & decides on tool usage
5. **Vector Search** → Semantic search through course embeddings
6. **Response Generation** → Claude synthesizes results into answer
7. **Source Tracking** → Citations collected from search tools
8. **Session Update** → Conversation history maintained for context
9. **Response Delivery** → Structured JSON back to frontend
10. **UI Rendering** → Markdown response with collapsible sources

The system maintains **conversational context** through session management and provides **accurate citations** through the tool-based search approach, creating a seamless educational assistant experience.