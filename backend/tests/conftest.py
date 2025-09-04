import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add backend to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from config import Config

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to Python", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="https://example.com/lesson2"),
        Lesson(lesson_number=3, title="Control Flow", lesson_link="https://example.com/lesson3")
    ]
    
    return Course(
        title="Python Programming Basics",
        course_link="https://example.com/course",
        instructor="John Doe",
        lessons=lessons
    )

@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = []
    
    # Lesson 1 chunks
    chunks.append(CourseChunk(
        content="Course Python Programming Basics Lesson 1 content: This is an introduction to Python programming. Python is a high-level programming language.",
        course_title=sample_course.title,
        lesson_number=1,
        chunk_index=0
    ))
    
    chunks.append(CourseChunk(
        content="Python is known for its simplicity and readability. It's used in web development, data science, and automation.",
        course_title=sample_course.title,
        lesson_number=1,
        chunk_index=1
    ))
    
    # Lesson 2 chunks
    chunks.append(CourseChunk(
        content="Course Python Programming Basics Lesson 2 content: Variables in Python store data values. Python has several data types including integers, strings, and lists.",
        course_title=sample_course.title,
        lesson_number=2,
        chunk_index=2
    ))
    
    return chunks

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    with patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
        mock_store = Mock(spec=VectorStore)
        
        # Default successful search result
        mock_store.search.return_value = SearchResults(
            documents=["Sample document content about Python programming"],
            metadata=[{"course_title": "Python Programming Basics", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        
        # Mock other methods
        mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
        mock_store._resolve_course_name.return_value = "Python Programming Basics"
        
        # Mock course catalog for outline tool tests
        mock_store.course_catalog = Mock()
        
        return mock_store

@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults.empty("Search error: ChromaDB connection failed")

@pytest.fixture
def successful_search_results():
    """Create successful search results for testing"""
    return SearchResults(
        documents=[
            "Python is a programming language that lets you work more quickly and integrate your systems more effectively.",
            "Variables in Python are used to store data values. Python has different data types for different kinds of data."
        ],
        metadata=[
            {"course_title": "Python Programming Basics", "lesson_number": 1},
            {"course_title": "Python Programming Basics", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock a successful response with tool use
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock content with tool use
    mock_tool_content = Mock()
    mock_tool_content.type = "tool_use"
    mock_tool_content.name = "search_course_content"
    mock_tool_content.id = "tool_123"
    mock_tool_content.input = {"query": "test query"}
    
    mock_response.content = [mock_tool_content]
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def mock_anthropic_text_response():
    """Create a mock Anthropic client that returns text responses"""
    mock_client = Mock()
    
    # Mock a text response
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    
    mock_text_content = Mock()
    mock_text_content.text = "This is a sample AI response about Python programming."
    
    mock_response.content = [mock_text_content]
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client

@pytest.fixture
def test_config():
    """Create a test configuration"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 20
    config.MAX_RESULTS = 3
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = ":memory:"  # Use in-memory database for tests
    return config

@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB instance for integration tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager for testing"""
    mock_manager = Mock()
    
    # Mock tool definitions
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }
    ]
    
    # Mock successful tool execution
    mock_manager.execute_tool.return_value = "Found relevant content about Python programming basics."
    
    # Mock sources
    mock_manager.get_last_sources.return_value = ["Python Programming Basics - Lesson 1|https://example.com/lesson1"]
    
    return mock_manager

@pytest.fixture
def sample_query_scenarios():
    """Provide various query scenarios for testing"""
    return {
        "simple_query": "What is Python?",
        "course_specific": "Tell me about variables in Python Programming Basics",
        "lesson_specific": "What's covered in lesson 1?",
        "complex_query": "How do I use variables and data types in Python programming?",
        "nonexistent_course": "Tell me about JavaScript basics",
        "empty_query": "",
        "very_long_query": "What is Python programming and how do I get started with learning Python programming language for web development and data science applications in the modern software development ecosystem?"
    }