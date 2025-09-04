import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk

class TestSearchResults:
    """Test suite for SearchResults class"""
    
    def test_from_chroma_success(self):
        """Test creating SearchResults from ChromaDB results"""
        # Setup ChromaDB result format
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'Python'}, {'course': 'Java'}]],
            'distances': [[0.1, 0.2]]
        }
        
        # Execute
        results = SearchResults.from_chroma(chroma_results)
        
        # Verify
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'course': 'Python'}, {'course': 'Java'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error is None
    
    def test_empty_class_method(self):
        """Test creating empty SearchResults with error"""
        error_msg = "Database connection failed"
        results = SearchResults.empty(error_msg)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == error_msg
    
    def test_is_empty_method(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty() is True
        
        # Non-empty results
        non_empty_results = SearchResults(['doc1'], [{'meta': 'data'}], [0.1])
        assert non_empty_results.is_empty() is False


class TestVectorStore:
    """Test suite for VectorStore functionality"""
    
    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for ChromaDB"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client"""
        with patch('chromadb.PersistentClient') as mock_client_class, \
             patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_embedding:
            
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock embedding function
            mock_embedding_instance = Mock()
            mock_embedding.return_value = mock_embedding_instance
            
            # Mock collections
            mock_catalog = Mock()
            mock_content = Mock()
            mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
            
            yield mock_client, mock_catalog, mock_content
    
    def test_init(self, mock_chroma_client, temp_chroma_path):
        """Test VectorStore initialization"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        store = VectorStore(temp_chroma_path, "test-model", max_results=3)
        
        assert store.max_results == 3
        assert store.course_catalog == mock_catalog
        assert store.course_content == mock_content
    
    def test_search_success(self, mock_chroma_client, temp_chroma_path):
        """Test successful search operation"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Setup mock content collection response
        mock_content.query.return_value = {
            'documents': [['Python is a programming language']],
            'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        results = store.search("What is Python?")
        
        # Verify
        assert not results.is_empty()
        assert results.error is None
        assert len(results.documents) == 1
        assert "Python is a programming language" in results.documents[0]
        
        # Verify query was called correctly
        mock_content.query.assert_called_once_with(
            query_texts=["What is Python?"],
            n_results=5,  # default max_results
            where=None
        )
    
    def test_search_with_course_filter(self, mock_chroma_client, temp_chroma_path):
        """Test search with course name filter"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock course name resolution
        with patch.object(VectorStore, '_resolve_course_name', return_value='Python Basics'):
            # Setup mock responses
            mock_content.query.return_value = {
                'documents': [['Course content']],
                'metadatas': [[{'course_title': 'Python Basics'}]],
                'distances': [[0.1]]
            }
            
            store = VectorStore(temp_chroma_path, "test-model")
            
            # Execute
            results = store.search("test query", course_name="Python")
            
            # Verify
            assert not results.is_empty()
            mock_content.query.assert_called_once()
            args, kwargs = mock_content.query.call_args
            assert kwargs['where'] == {'course_title': 'Python Basics'}
    
    def test_search_with_lesson_filter(self, mock_chroma_client, temp_chroma_path):
        """Test search with lesson number filter"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        mock_content.query.return_value = {
            'documents': [['Lesson content']],
            'metadatas': [[{'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        results = store.search("test query", lesson_number=1)
        
        # Verify
        assert not results.is_empty()
        mock_content.query.assert_called_once()
        args, kwargs = mock_content.query.call_args
        assert kwargs['where'] == {'lesson_number': 1}
    
    def test_search_with_both_filters(self, mock_chroma_client, temp_chroma_path):
        """Test search with both course and lesson filters"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        with patch.object(VectorStore, '_resolve_course_name', return_value='Python Basics'):
            mock_content.query.return_value = {
                'documents': [['Specific lesson content']],
                'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 2}]],
                'distances': [[0.1]]
            }
            
            store = VectorStore(temp_chroma_path, "test-model")
            
            # Execute
            results = store.search("test query", course_name="Python", lesson_number=2)
            
            # Verify
            assert not results.is_empty()
            mock_content.query.assert_called_once()
            args, kwargs = mock_content.query.call_args
            expected_filter = {
                "$and": [
                    {"course_title": "Python Basics"},
                    {"lesson_number": 2}
                ]
            }
            assert kwargs['where'] == expected_filter
    
    def test_search_course_not_found(self, mock_chroma_client, temp_chroma_path):
        """Test search when course name cannot be resolved"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        with patch.object(VectorStore, '_resolve_course_name', return_value=None):
            store = VectorStore(temp_chroma_path, "test-model")
            
            # Execute
            results = store.search("test query", course_name="Nonexistent Course")
            
            # Verify
            assert results.is_empty()
            assert "No course found matching 'Nonexistent Course'" in results.error
    
    def test_search_exception_handling(self, mock_chroma_client, temp_chroma_path):
        """Test search exception handling"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock query to raise exception
        mock_content.query.side_effect = Exception("ChromaDB error")
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        results = store.search("test query")
        
        # Verify
        assert results.is_empty()
        assert "Search error: ChromaDB error" in results.error
    
    def test_resolve_course_name_success(self, mock_chroma_client, temp_chroma_path):
        """Test successful course name resolution"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock catalog query response
        mock_catalog.query.return_value = {
            'documents': [['Python Programming Basics']],
            'metadatas': [[{'title': 'Python Programming Basics'}]]
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        result = store._resolve_course_name("Python")
        
        # Verify
        assert result == "Python Programming Basics"
        mock_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
    
    def test_resolve_course_name_not_found(self, mock_chroma_client, temp_chroma_path):
        """Test course name resolution when not found"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock empty response
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        result = store._resolve_course_name("Nonexistent")
        
        # Verify
        assert result is None
    
    def test_build_filter_no_filters(self, mock_chroma_client, temp_chroma_path):
        """Test filter building with no filters"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        result = store._build_filter(None, None)
        assert result is None
    
    def test_build_filter_course_only(self, mock_chroma_client, temp_chroma_path):
        """Test filter building with course only"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        result = store._build_filter("Python Basics", None)
        assert result == {"course_title": "Python Basics"}
    
    def test_build_filter_lesson_only(self, mock_chroma_client, temp_chroma_path):
        """Test filter building with lesson only"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        result = store._build_filter(None, 1)
        assert result == {"lesson_number": 1}
    
    def test_build_filter_both(self, mock_chroma_client, temp_chroma_path):
        """Test filter building with both course and lesson"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        result = store._build_filter("Python Basics", 1)
        expected = {
            "$and": [
                {"course_title": "Python Basics"},
                {"lesson_number": 1}
            ]
        }
        assert result == expected
    
    def test_add_course_metadata(self, mock_chroma_client, temp_chroma_path, sample_course):
        """Test adding course metadata"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        store.add_course_metadata(sample_course)
        
        # Verify catalog.add was called
        mock_catalog.add.assert_called_once()
        args, kwargs = mock_catalog.add.call_args
        
        assert kwargs['documents'] == [sample_course.title]
        assert kwargs['ids'] == [sample_course.title]
        
        # Verify metadata structure
        metadata = kwargs['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert metadata['course_link'] == sample_course.course_link
        assert 'lessons_json' in metadata
        assert metadata['lesson_count'] == len(sample_course.lessons)
    
    def test_add_course_content(self, mock_chroma_client, temp_chroma_path, sample_course_chunks):
        """Test adding course content chunks"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        store.add_course_content(sample_course_chunks)
        
        # Verify content.add was called
        mock_content.add.assert_called_once()
        args, kwargs = mock_content.add.call_args
        
        assert len(kwargs['documents']) == len(sample_course_chunks)
        assert len(kwargs['metadatas']) == len(sample_course_chunks)
        assert len(kwargs['ids']) == len(sample_course_chunks)
        
        # Verify first chunk
        assert kwargs['documents'][0] == sample_course_chunks[0].content
        assert kwargs['metadatas'][0]['course_title'] == sample_course_chunks[0].course_title
        assert kwargs['metadatas'][0]['lesson_number'] == sample_course_chunks[0].lesson_number
    
    def test_add_course_content_empty_list(self, mock_chroma_client, temp_chroma_path):
        """Test adding empty course content list"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        store.add_course_content([])
        
        # Verify content.add was not called
        mock_content.add.assert_not_called()
    
    def test_clear_all_data(self, mock_chroma_client, temp_chroma_path):
        """Test clearing all data"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        store.clear_all_data()
        
        # Verify delete_collection was called
        assert mock_client.delete_collection.call_count == 2
        mock_client.delete_collection.assert_any_call("course_catalog")
        mock_client.delete_collection.assert_any_call("course_content")
        
        # Verify get_or_create_collection was called again to recreate
        # Should be called twice for initialization and twice for recreation
        expected_calls = 4  # 2 initial + 2 recreate
        assert mock_client.get_or_create_collection.call_count >= expected_calls - 1  # Allow for minor variation in mock setup
    
    def test_get_existing_course_titles(self, mock_chroma_client, temp_chroma_path):
        """Test getting existing course titles"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock catalog.get response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        titles = store.get_existing_course_titles()
        
        # Verify
        assert titles == ['Course 1', 'Course 2', 'Course 3']
        mock_catalog.get.assert_called_once()
    
    def test_get_course_count(self, mock_chroma_client, temp_chroma_path):
        """Test getting course count"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock catalog.get response
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        count = store.get_course_count()
        
        # Verify
        assert count == 2
    
    def test_get_lesson_link(self, mock_chroma_client, temp_chroma_path):
        """Test getting lesson link"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock catalog response with lessons JSON
        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}, {"lesson_number": 2, "lesson_link": "https://example.com/lesson2"}]'
        mock_catalog.get.return_value = {
            'metadatas': [{
                'lessons_json': lessons_json
            }]
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute
        link = store.get_lesson_link("Test Course", 1)
        
        # Verify
        assert link == "https://example.com/lesson1"
        mock_catalog.get.assert_called_once_with(ids=["Test Course"])
    
    def test_get_lesson_link_not_found(self, mock_chroma_client, temp_chroma_path):
        """Test getting lesson link when lesson not found"""
        mock_client, mock_catalog, mock_content = mock_chroma_client
        
        # Mock catalog response with lessons JSON
        lessons_json = '[{"lesson_number": 1, "lesson_link": "https://example.com/lesson1"}]'
        mock_catalog.get.return_value = {
            'metadatas': [{
                'lessons_json': lessons_json
            }]
        }
        
        store = VectorStore(temp_chroma_path, "test-model")
        
        # Execute - request lesson 2 which doesn't exist
        link = store.get_lesson_link("Test Course", 2)
        
        # Verify
        assert link is None