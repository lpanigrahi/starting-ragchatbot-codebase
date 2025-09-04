import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults

class TestCourseSearchTool:
    """Test suite for CourseSearchTool.execute() method"""
    
    def test_successful_search_with_results(self, mock_vector_store, successful_search_results):
        """Test successful search that returns results"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(query="What is Python?")
        
        # Verify
        assert result != ""
        assert "Python Programming Basics" in result
        assert "Lesson 1" in result
        assert "Python is a programming language" in result
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name=None,
            lesson_number=None
        )
        
        # Verify sources were stored
        assert len(tool.last_sources) == 2
        assert "Python Programming Basics - Lesson 1" in tool.last_sources[0]
        assert "Python Programming Basics - Lesson 2" in tool.last_sources[1]
    
    def test_search_with_course_filter(self, mock_vector_store, successful_search_results):
        """Test search with course name filter"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(
            query="What is Python?", 
            course_name="Python Programming"
        )
        
        # Verify
        assert result != ""
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name="Python Programming",
            lesson_number=None
        )
    
    def test_search_with_lesson_filter(self, mock_vector_store, successful_search_results):
        """Test search with lesson number filter"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(
            query="What is Python?",
            lesson_number=1
        )
        
        # Verify
        assert result != ""
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name=None,
            lesson_number=1
        )
    
    def test_search_with_both_filters(self, mock_vector_store, successful_search_results):
        """Test search with both course name and lesson number filters"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(
            query="What is Python?",
            course_name="Python Programming",
            lesson_number=2
        )
        
        # Verify
        assert result != ""
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name="Python Programming",
            lesson_number=2
        )
    
    def test_empty_search_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        # Setup
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(query="nonexistent topic")
        
        # Verify
        assert "No relevant content found" in result
        assert tool.last_sources == []
    
    def test_empty_results_with_course_filter(self, mock_vector_store, empty_search_results):
        """Test empty results with course filter shows filter info"""
        # Setup
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(
            query="nonexistent topic",
            course_name="Python Programming"
        )
        
        # Verify
        assert "No relevant content found in course 'Python Programming'" in result
    
    def test_empty_results_with_lesson_filter(self, mock_vector_store, empty_search_results):
        """Test empty results with lesson filter shows filter info"""
        # Setup
        mock_vector_store.search.return_value = empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(
            query="nonexistent topic",
            lesson_number=5
        )
        
        # Verify
        assert "No relevant content found in lesson 5" in result
    
    def test_search_error_handling(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        # Setup
        mock_vector_store.search.return_value = error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(query="test query")
        
        # Verify
        assert "Search error: ChromaDB connection failed" in result
        assert tool.last_sources == []
    
    def test_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        # Verify structure
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        
        # Verify schema
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]
    
    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting includes lesson links when available"""
        # Setup search results
        results = SearchResults(
            documents=["Sample content about Python"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1]
        )
        
        mock_vector_store.search.return_value = results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(query="Python")
        
        # Verify
        assert "[Python Basics - Lesson 1]" in result
        assert len(tool.last_sources) == 1
        assert "https://example.com/lesson1" in tool.last_sources[0]
    
    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test result formatting when lesson_number is None"""
        # Setup search results without lesson number
        results = SearchResults(
            documents=["Sample content about Python"],
            metadata=[{"course_title": "Python Basics", "lesson_number": None}],
            distances=[0.1]
        )
        
        mock_vector_store.search.return_value = results
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Execute
        result = tool.execute(query="Python")
        
        # Verify - should not include lesson info
        assert "[Python Basics]" in result
        assert "Lesson" not in result
    
    def test_sources_reset_between_searches(self, mock_vector_store, successful_search_results):
        """Test that sources are properly reset between searches"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        
        # First search
        tool.execute(query="First query")
        first_sources = tool.last_sources.copy()
        assert len(first_sources) > 0  # Verify we have sources from first search
        
        # Second search with empty results
        empty_results = SearchResults([], [], [])
        mock_vector_store.search.return_value = empty_results
        tool.execute(query="Second query")
        
        # Verify sources were reset (empty results should clear sources)
        assert tool.last_sources != first_sources
        assert tool.last_sources == []


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""
    
    def test_successful_outline_retrieval(self, mock_vector_store):
        """Test successful course outline retrieval"""
        # Setup mock responses
        mock_vector_store._resolve_course_name.return_value = "Python Programming Basics"
        
        # Mock course catalog response - ensure the get method returns the expected structure
        mock_results = {
            'metadatas': [{
                'course_link': 'https://example.com/course',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Introduction"}, {"lesson_number": 2, "lesson_title": "Variables"}]'
            }]
        }
        # Properly mock the course_catalog.get method
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        
        # Execute
        result = tool.execute(course_name="Python Programming")
        
        # Verify
        assert "**Course**: Python Programming Basics" in result
        assert "**Course Link**: https://example.com/course" in result
        assert "**Total Lessons**: 2" in result
        assert "1. Introduction" in result
        assert "2. Variables" in result
        
        # Verify the method calls were made correctly
        mock_vector_store._resolve_course_name.assert_called_once_with("Python Programming")
        mock_vector_store.course_catalog.get.assert_called_once_with(ids=["Python Programming Basics"])
    
    def test_course_not_found(self, mock_vector_store):
        """Test handling when course is not found"""
        # Setup
        mock_vector_store._resolve_course_name.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        
        # Execute
        result = tool.execute(course_name="Nonexistent Course")
        
        # Verify
        assert "No course found matching 'Nonexistent Course'" in result
    
    def test_outline_tool_definition(self, mock_vector_store):
        """Test that outline tool definition is properly formatted"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        # Verify structure
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        
        # Verify schema
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "course_name" in schema["properties"]
        assert schema["required"] == ["course_name"]


class TestToolManager:
    """Test suite for ToolManager"""
    
    def test_tool_registration(self, mock_vector_store):
        """Test tool registration functionality"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        # Register tool
        manager.register_tool(search_tool)
        
        # Verify registration
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == search_tool
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test retrieving all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        # Get definitions
        definitions = manager.get_tool_definitions()
        
        # Verify
        assert len(definitions) == 2
        tool_names = [def_["name"] for def_ in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names
    
    def test_execute_tool(self, mock_vector_store, successful_search_results):
        """Test tool execution through manager"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute
        result = manager.execute_tool("search_course_content", query="test")
        
        # Verify
        assert result != ""
        assert "Python Programming Basics" in result
    
    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing nonexistent tool returns error"""
        manager = ToolManager()
        
        # Execute nonexistent tool
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        # Verify
        assert "Tool 'nonexistent_tool' not found" in result
    
    def test_get_last_sources(self, mock_vector_store, successful_search_results):
        """Test retrieving sources from last search"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test")
        
        # Get sources
        sources = manager.get_last_sources()
        
        # Verify
        assert len(sources) > 0
        assert any("Python Programming Basics" in source for source in sources)
    
    def test_reset_sources(self, mock_vector_store, successful_search_results):
        """Test sources reset functionality"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)
        
        # Execute search to generate sources
        manager.execute_tool("search_course_content", query="test")
        
        # Verify sources exist
        assert len(manager.get_last_sources()) > 0
        
        # Reset sources
        manager.reset_sources()
        
        # Verify sources are cleared
        assert len(manager.get_last_sources()) == 0