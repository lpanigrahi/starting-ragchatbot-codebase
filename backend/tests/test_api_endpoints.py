import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

@pytest.mark.api
class TestAPIEndpoints:
    """Test the FastAPI endpoints for the RAG system"""
    
    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns success"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        assert response.json() == {"message": "RAG System API is running"}
    
    def test_query_endpoint_with_session(self, test_client, api_query_request, expected_query_response):
        """Test the /api/query endpoint with session ID"""
        response = test_client.post("/api/query", json=api_query_request)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "answer" in response_data
        assert "sources" in response_data
        assert "session_id" in response_data
        assert response_data["answer"] == expected_query_response["answer"]
        assert response_data["sources"] == expected_query_response["sources"]
        assert response_data["session_id"] == expected_query_response["session_id"]
    
    def test_query_endpoint_without_session(self, test_client, api_query_request_no_session):
        """Test the /api/query endpoint without session ID (should create one)"""
        response = test_client.post("/api/query", json=api_query_request_no_session)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "answer" in response_data
        assert "sources" in response_data
        assert "session_id" in response_data
        assert response_data["session_id"] == "test-session-123"  # Mock creates this ID
    
    def test_query_endpoint_invalid_request(self, test_client):
        """Test the /api/query endpoint with invalid request data"""
        invalid_request = {"invalid_field": "test"}
        
        response = test_client.post("/api/query", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
        
    def test_query_endpoint_empty_query(self, test_client):
        """Test the /api/query endpoint with empty query"""
        empty_request = {"query": ""}
        
        response = test_client.post("/api/query", json=empty_request)
        
        assert response.status_code == 200  # Should still process empty queries
        response_data = response.json()
        assert "answer" in response_data
        assert "sources" in response_data
        assert "session_id" in response_data
    
    def test_query_endpoint_rag_system_error(self, test_client):
        """Test the /api/query endpoint when RAG system raises an error"""
        # Configure mock to raise an exception
        test_client.app.state.mock_rag.query.side_effect = Exception("RAG system error")
        
        request_data = {"query": "test query"}
        response = test_client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
        
        # Reset mock for other tests
        test_client.app.state.mock_rag.query.side_effect = None
        test_client.app.state.mock_rag.query.return_value = ("Test response", ["Test source"])
    
    def test_courses_endpoint(self, test_client, expected_course_stats):
        """Test the /api/courses endpoint returns course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "total_courses" in response_data
        assert "course_titles" in response_data
        assert response_data["total_courses"] == expected_course_stats["total_courses"]
        assert response_data["course_titles"] == expected_course_stats["course_titles"]
    
    def test_courses_endpoint_rag_system_error(self, test_client):
        """Test the /api/courses endpoint when RAG system raises an error"""
        # Configure mock to raise an exception
        test_client.app.state.mock_rag.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]
        
        # Reset mock for other tests
        test_client.app.state.mock_rag.get_course_analytics.side_effect = None
        test_client.app.state.mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["Python Programming Basics", "Advanced Python"]
        }
    
    def test_clear_session_endpoint(self, test_client):
        """Test the DELETE /api/session/{session_id} endpoint"""
        session_id = "test-session-123"
        response = test_client.delete(f"/api/session/{session_id}")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["message"] == "Session cleared successfully"
        
        # Verify the mock was called with correct session ID
        test_client.app.state.mock_rag.session_manager.clear_session.assert_called_with(session_id)
    
    def test_clear_session_endpoint_error(self, test_client):
        """Test the DELETE /api/session/{session_id} endpoint when session manager raises an error"""
        # Configure mock to raise an exception
        test_client.app.state.mock_rag.session_manager.clear_session.side_effect = Exception("Session clear error")
        
        session_id = "test-session-123"
        response = test_client.delete(f"/api/session/{session_id}")
        
        assert response.status_code == 500
        assert "Session clear error" in response.json()["detail"]
        
        # Reset mock for other tests
        test_client.app.state.mock_rag.session_manager.clear_session.side_effect = None
        test_client.app.state.mock_rag.session_manager.clear_session.return_value = None

@pytest.mark.api
class TestAPIRequestValidation:
    """Test API request validation and error handling"""
    
    def test_query_request_missing_query(self, test_client):
        """Test query request with missing query field"""
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422
    
    def test_query_request_wrong_type(self, test_client):
        """Test query request with wrong data type"""
        response = test_client.post("/api/query", json={"query": 123})
        assert response.status_code == 422
    
    def test_query_request_invalid_json(self, test_client):
        """Test query request with invalid JSON"""
        response = test_client.post("/api/query", data="invalid json")
        assert response.status_code == 422
    
    def test_session_id_parameter_types(self, test_client):
        """Test different session ID parameter types"""
        # Valid string session ID
        response = test_client.post("/api/query", json={"query": "test", "session_id": "abc123"})
        assert response.status_code == 200
        
        # Null session ID (should be handled)
        response = test_client.post("/api/query", json={"query": "test", "session_id": None})
        assert response.status_code == 200
        
        # Invalid session ID type
        response = test_client.post("/api/query", json={"query": "test", "session_id": 123})
        assert response.status_code == 422

@pytest.mark.api
class TestAPIResponseFormats:
    """Test API response formats and structure"""
    
    def test_query_response_structure(self, test_client):
        """Test that query responses have the correct structure"""
        response = test_client.post("/api/query", json={"query": "test query"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Check sources structure
        for source in data["sources"]:
            assert isinstance(source, str)
    
    def test_courses_response_structure(self, test_client):
        """Test that courses responses have the correct structure"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields exist
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Check course titles structure
        for title in data["course_titles"]:
            assert isinstance(title, str)
    
    def test_error_response_structure(self, test_client):
        """Test that error responses have the correct structure"""
        # Configure mock to raise an exception
        test_client.app.state.mock_rag.query.side_effect = Exception("Test error")
        
        response = test_client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 500
        data = response.json()
        
        # Check error response structure
        assert "detail" in data
        assert isinstance(data["detail"], str)
        assert "Test error" in data["detail"]
        
        # Reset mock
        test_client.app.state.mock_rag.query.side_effect = None
        test_client.app.state.mock_rag.query.return_value = ("Test response", ["Test source"])

@pytest.mark.api
@pytest.mark.integration
class TestAPIIntegrationFlows:
    """Test complete API interaction flows"""
    
    def test_complete_query_session_flow(self, test_client):
        """Test a complete flow: query -> get courses -> clear session"""
        # Step 1: Make a query (creates session)
        query_response = test_client.post("/api/query", json={"query": "What is Python?"})
        assert query_response.status_code == 200
        
        query_data = query_response.json()
        session_id = query_data["session_id"]
        
        # Step 2: Get course statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200
        
        courses_data = courses_response.json()
        assert courses_data["total_courses"] > 0
        
        # Step 3: Make another query with the same session
        query2_response = test_client.post("/api/query", json={
            "query": "Tell me more about variables",
            "session_id": session_id
        })
        assert query2_response.status_code == 200
        
        query2_data = query2_response.json()
        assert query2_data["session_id"] == session_id
        
        # Step 4: Clear the session
        clear_response = test_client.delete(f"/api/session/{session_id}")
        assert clear_response.status_code == 200
        
        clear_data = clear_response.json()
        assert clear_data["message"] == "Session cleared successfully"
    
    def test_multiple_concurrent_sessions(self, test_client):
        """Test handling multiple concurrent sessions"""
        # Reset mock to return different session IDs
        session_ids = ["session-1", "session-2", "session-3"]
        test_client.app.state.mock_rag.session_manager.create_session.side_effect = session_ids
        
        # Create multiple queries without session IDs (should create new sessions)
        responses = []
        for i in range(3):
            response = test_client.post("/api/query", json={"query": f"Query {i+1}"})
            assert response.status_code == 200
            responses.append(response.json())
        
        # Each should have a different session ID
        retrieved_sessions = [resp["session_id"] for resp in responses]
        assert len(set(retrieved_sessions)) == 3  # All unique
        
        # Reset mock
        test_client.app.state.mock_rag.session_manager.create_session.side_effect = None
        test_client.app.state.mock_rag.session_manager.create_session.return_value = "test-session-123"