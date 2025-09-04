import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk

class TestRAGSystem:
    """Integration test suite for RAGSystem"""
    
    @pytest.fixture
    def temp_docs_dir(self):
        """Create temporary directory with sample documents"""
        temp_dir = tempfile.mkdtemp()
        
        # Create a sample course document
        sample_doc_content = """Course Title: Python Programming Basics
Course Link: https://example.com/python-course
Course Instructor: Jane Smith

Lesson 1: Introduction to Python
Lesson Link: https://example.com/python-course/lesson1

Python is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development, as well as for use as a scripting or glue language to connect existing components together.

Python's simple, easy-to-learn syntax emphasizes readability and therefore reduces the cost of program maintenance. Python supports modules and packages, which encourages program modularity and code reuse.

Lesson 2: Variables and Data Types
Lesson Link: https://example.com/python-course/lesson2

In Python, variables are used to store data values. Unlike other programming languages, Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

Python has the following data types built-in by default:
- Text Type: str
- Numeric Types: int, float, complex
- Sequence Types: list, tuple, range
- Boolean Type: bool"""
        
        doc_path = os.path.join(temp_dir, "python_course.txt")
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(sample_doc_content)
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_chroma_db):
        """Create mock configuration"""
        config = Mock()
        config.CHUNK_SIZE = 200
        config.CHUNK_OVERLAP = 50
        config.CHROMA_PATH = temp_chroma_db
        config.EMBEDDING_MODEL = "test-model"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-3-sonnet"
        config.MAX_HISTORY = 2
        return config
    
    def test_rag_system_initialization(self, mock_config):
        """Test RAGSystem initializes all components correctly"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag_system = RAGSystem(mock_config)
            
            # Verify components are initialized
            assert rag_system.document_processor is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.session_manager is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None
            assert rag_system.outline_tool is not None
    
    def test_add_course_document_success(self, mock_config, temp_docs_dir):
        """Test successfully adding a course document"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            # Setup mocks
            sample_course = Course(
                title="Python Programming Basics",
                course_link="https://example.com/python-course",
                instructor="Jane Smith"
            )
            sample_chunks = [
                CourseChunk(content="Sample chunk", course_title="Python Programming Basics", chunk_index=0)
            ]
            
            mock_doc_processor = mock_doc_proc.return_value
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
            
            mock_store = mock_vector_store.return_value
            
            rag_system = RAGSystem(mock_config)
            
            # Execute
            course, chunk_count = rag_system.add_course_document(os.path.join(temp_docs_dir, "python_course.txt"))
            
            # Verify
            assert course.title == "Python Programming Basics"
            assert chunk_count == 1
            
            # Verify vector store operations
            mock_store.add_course_metadata.assert_called_once_with(sample_course)
            mock_store.add_course_content.assert_called_once_with(sample_chunks)
    
    def test_add_course_document_error(self, mock_config, temp_docs_dir):
        """Test error handling when adding course document fails"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            mock_doc_processor = mock_doc_proc.return_value
            mock_doc_processor.process_course_document.side_effect = Exception("Processing error")
            
            rag_system = RAGSystem(mock_config)
            
            # Execute
            course, chunk_count = rag_system.add_course_document("nonexistent_file.txt")
            
            # Verify
            assert course is None
            assert chunk_count == 0
    
    def test_add_course_folder_success(self, mock_config, temp_docs_dir):
        """Test adding all documents from a folder"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            # Setup mocks
            sample_course = Course(title="Python Programming Basics")
            sample_chunks = [CourseChunk(content="chunk", course_title="Python Programming Basics", chunk_index=0)]
            
            mock_doc_processor = mock_doc_proc.return_value
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
            
            mock_store = mock_vector_store.return_value
            mock_store.get_existing_course_titles.return_value = []
            
            rag_system = RAGSystem(mock_config)
            
            # Execute
            courses_added, chunks_added = rag_system.add_course_folder(temp_docs_dir)
            
            # Verify
            assert courses_added == 1
            assert chunks_added == 1
    
    def test_add_course_folder_skip_existing(self, mock_config, temp_docs_dir):
        """Test skipping existing courses when adding from folder"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            sample_course = Course(title="Python Programming Basics")
            sample_chunks = [CourseChunk(content="chunk", course_title="Python Programming Basics", chunk_index=0)]
            
            mock_doc_processor = mock_doc_proc.return_value
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
            
            mock_store = mock_vector_store.return_value
            # Simulate course already exists
            mock_store.get_existing_course_titles.return_value = ["Python Programming Basics"]
            
            rag_system = RAGSystem(mock_config)
            
            # Execute
            courses_added, chunks_added = rag_system.add_course_folder(temp_docs_dir)
            
            # Verify no courses were added
            assert courses_added == 0
            assert chunks_added == 0
    
    def test_query_without_session(self, mock_config):
        """Test query processing without session ID"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:
            
            # Setup mocks
            mock_generator = mock_ai_gen.return_value
            mock_generator.generate_response.return_value = "Python is a programming language."
            
            mock_tool_manager = Mock()
            mock_tool_manager.get_last_sources.return_value = ["Python Course - Lesson 1"]
            
            rag_system = RAGSystem(mock_config)
            rag_system.tool_manager = mock_tool_manager
            
            # Execute
            response, sources = rag_system.query("What is Python?")
            
            # Verify
            assert response == "Python is a programming language."
            assert sources == ["Python Course - Lesson 1"]
            
            # Verify AI generator was called correctly
            mock_generator.generate_response.assert_called_once()
            call_args = mock_generator.generate_response.call_args
            assert "Answer this question about course materials: What is Python?" in call_args[1]["query"]
            assert call_args[1]["conversation_history"] is None
            assert call_args[1]["tools"] is not None
            assert call_args[1]["tool_manager"] is not None
    
    def test_query_with_session(self, mock_config):
        """Test query processing with session ID"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:
            
            # Setup mocks
            mock_generator = mock_ai_gen.return_value
            mock_generator.generate_response.return_value = "Variables store data values."
            
            mock_session_manager = mock_session_mgr.return_value
            mock_session_manager.get_conversation_history.return_value = "Previous: What is Python?\nAssistant: Python is a language."
            
            mock_tool_manager = Mock()
            mock_tool_manager.get_last_sources.return_value = []
            
            rag_system = RAGSystem(mock_config)
            rag_system.tool_manager = mock_tool_manager
            
            # Execute
            response, sources = rag_system.query("What are variables?", session_id="test-session")
            
            # Verify
            assert response == "Variables store data values."
            
            # Verify session manager was used
            mock_session_manager.get_conversation_history.assert_called_once_with("test-session")
            mock_session_manager.add_exchange.assert_called_once_with(
                "test-session", 
                "What are variables?", 
                "Variables store data values."
            )
            
            # Verify conversation history was passed to AI
            call_args = mock_generator.generate_response.call_args
            assert call_args[1]["conversation_history"] == "Previous: What is Python?\nAssistant: Python is a language."
    
    def test_query_sources_reset(self, mock_config):
        """Test that sources are reset after query"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager'):
            
            mock_generator = mock_ai_gen.return_value
            mock_generator.generate_response.return_value = "Test response"
            
            mock_tool_manager = Mock()
            mock_tool_manager.get_last_sources.return_value = ["Source 1"]
            
            rag_system = RAGSystem(mock_config)
            rag_system.tool_manager = mock_tool_manager
            
            # Execute
            response, sources = rag_system.query("Test query")
            
            # Verify sources were retrieved and reset
            mock_tool_manager.get_last_sources.assert_called_once()
            mock_tool_manager.reset_sources.assert_called_once()
    
    def test_get_course_analytics(self, mock_config):
        """Test getting course analytics"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            mock_store = mock_vector_store.return_value
            mock_store.get_course_count.return_value = 3
            mock_store.get_existing_course_titles.return_value = ["Course 1", "Course 2", "Course 3"]
            
            rag_system = RAGSystem(mock_config)
            
            # Execute
            analytics = rag_system.get_course_analytics()
            
            # Verify
            assert analytics["total_courses"] == 3
            assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]
    
    def test_integration_with_real_components(self, test_config):
        """Integration test with real components (except Anthropic API)"""
        # This test uses real components to test integration
        # Only mocking the external Anthropic API call
        
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic, \
             patch('chromadb.PersistentClient') as mock_chroma:
            
            # Setup mock Anthropic response
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            mock_response = Mock()
            mock_response.stop_reason = "end_turn"
            mock_text_content = Mock()
            mock_text_content.text = "Python is a high-level programming language."
            mock_response.content = [mock_text_content]
            mock_client.messages.create.return_value = mock_response
            
            # Setup mock ChromaDB
            mock_chroma_client = Mock()
            mock_chroma.return_value = mock_chroma_client
            
            mock_collection = Mock()
            mock_chroma_client.get_or_create_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
            # Create RAG system with real components
            rag_system = RAGSystem(test_config)
            
            # Execute query
            response, sources = rag_system.query("What is Python?")
            
            # Verify response
            assert response == "Python is a high-level programming language."
            assert isinstance(sources, list)
    
    def test_error_propagation_in_query(self, mock_config):
        """Test error propagation during query processing"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager'):
            
            # Setup AI generator to raise exception
            mock_generator = mock_ai_gen.return_value
            mock_generator.generate_response.side_effect = Exception("API Error")
            
            rag_system = RAGSystem(mock_config)
            
            # Execute and expect exception
            with pytest.raises(Exception) as exc_info:
                rag_system.query("Test query")
            
            assert "API Error" in str(exc_info.value)
    
    def test_empty_query_handling(self, mock_config):
        """Test handling of empty queries"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager'):
            
            mock_generator = mock_ai_gen.return_value
            mock_generator.generate_response.return_value = "I need more information."
            
            mock_tool_manager = Mock()
            mock_tool_manager.get_last_sources.return_value = []
            
            rag_system = RAGSystem(mock_config)
            rag_system.tool_manager = mock_tool_manager
            
            # Execute with empty query
            response, sources = rag_system.query("")
            
            # Verify it still processes (doesn't prevent empty queries)
            assert response == "I need more information."
            assert sources == []
    
    def test_tool_manager_integration(self, mock_config):
        """Test tool manager integration in RAG system"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'):
            
            rag_system = RAGSystem(mock_config)
            
            # Verify tools are registered
            assert "search_course_content" in rag_system.tool_manager.tools
            assert "get_course_outline" in rag_system.tool_manager.tools
            
            # Verify tool definitions are available
            definitions = rag_system.tool_manager.get_tool_definitions()
            assert len(definitions) == 2
            
            tool_names = [def_["name"] for def_ in definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names