import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator

class TestAIGenerator:
    """Test suite for AIGenerator class"""
    
    @pytest.fixture
    def ai_generator(self):
        """Create AIGenerator instance for testing"""
        return AIGenerator("test-api-key", "claude-3-sonnet-20240229")
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            yield mock_client
    
    def test_init(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-key", "test-model")
        
        assert generator.model == "test-model"
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self, mock_anthropic_client):
        """Test response generation without tools"""
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "This is a test response about Python programming."
        mock_response.content = [mock_text_content]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Execute
        result = generator.generate_response("What is Python?")
        
        # Verify
        assert result == "This is a test response about Python programming."
        
        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args
        
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert len(call_args[1]["messages"]) == 1
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
    
    def test_generate_response_with_conversation_history(self, mock_anthropic_client):
        """Test response generation with conversation history"""
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "Continuing our Python discussion..."
        mock_response.content = [mock_text_content]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Execute with conversation history
        result = generator.generate_response(
            "Tell me more about variables",
            conversation_history="Previous: What is Python?\nAssistant: Python is a programming language."
        )
        
        # Verify
        assert result == "Continuing our Python discussion..."
        
        # Verify system content includes history
        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert "What is Python?" in system_content
    
    def test_generate_response_with_tools_but_no_tool_use(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation with tools available but not used"""
        # Setup mock response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "I can answer that without using tools."
        mock_response.content = [mock_text_content]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock tool definitions
        tools = [{"name": "search_course_content", "description": "Search courses"}]
        
        # Execute
        result = generator.generate_response(
            "What is programming?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify
        assert result == "I can answer that without using tools."
        
        # Verify tools were provided in API call
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation that uses tools (single round)"""
        # Setup initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "Python basics"}
        
        mock_initial_response.content = [mock_tool_content]
        
        # Setup final response after tool execution (no more tool use)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "Based on the search results, Python is a programming language..."
        mock_final_response.content = [mock_text_content]
        
        # Mock client to return different responses on subsequent calls
        mock_anthropic_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager.execute_tool.return_value = "Python is a high-level programming language."
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        tools = [{"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "What is Python?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify
        assert result == "Based on the search results, Python is a programming language..."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        
        # Verify two API calls were made (initial + final)
        assert mock_anthropic_client.messages.create.call_count == 2
        
        # Verify message structure
        final_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]
        # Should have: user query, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
    
    def test_handle_tool_execution_single_tool(self, mock_anthropic_client):
        """Test _handle_tool_execution with single tool"""
        # Setup
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test query"}
        mock_initial_response.content = [mock_tool_content]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        # Mock final response
        mock_final_response = Mock()
        mock_text_content = Mock()
        mock_text_content.text = "Final answer based on tool results"
        mock_final_response.content = [mock_text_content]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        # Base parameters
        base_params = {
            "messages": [{"role": "user", "content": "What is Python?"}],
            "system": "You are a helpful assistant.",
            "model": "test-model",
            "temperature": 0,
            "max_tokens": 800
        }
        
        # Execute
        result = generator._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)
        
        # Verify
        assert result == "Final answer based on tool results"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify final API call structure
        mock_anthropic_client.messages.create.assert_called_once()
        final_call_args = mock_anthropic_client.messages.create.call_args
        
        # Should have 3 messages: user query, assistant tool use, user tool results
        messages = final_call_args[1]["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Verify tool result structure
        tool_result = messages[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_123"
        assert tool_result["content"] == "Tool execution result"
    
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_client):
        """Test _handle_tool_execution with multiple tools"""
        # Setup
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Mock initial response with multiple tool uses
        mock_tool_1 = Mock()
        mock_tool_1.type = "tool_use"
        mock_tool_1.name = "search_course_content"
        mock_tool_1.id = "tool_1"
        mock_tool_1.input = {"query": "Python basics"}
        
        mock_tool_2 = Mock()
        mock_tool_2.type = "tool_use"
        mock_tool_2.name = "get_course_outline"
        mock_tool_2.id = "tool_2"
        mock_tool_2.input = {"course_name": "Python"}
        
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_1, mock_tool_2]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result for Python basics",
            "Course outline for Python"
        ]
        
        # Mock final response
        mock_final_response = Mock()
        mock_text_content = Mock()
        mock_text_content.text = "Combined response using both tools"
        mock_final_response.content = [mock_text_content]
        mock_anthropic_client.messages.create.return_value = mock_final_response
        
        base_params = {
            "messages": [{"role": "user", "content": "Tell me about Python"}],
            "system": "You are a helpful assistant.",
        }
        
        # Execute
        result = generator._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)
        
        # Verify
        assert result == "Combined response using both tools"
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify tool execution calls
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("search_course_content",)
        assert calls[0][1] == {"query": "Python basics"}
        assert calls[1][0] == ("get_course_outline",)
        assert calls[1][1] == {"course_name": "Python"}
        
        # Verify final message structure
        final_call_args = mock_anthropic_client.messages.create.call_args
        messages = final_call_args[1]["messages"]
        
        # Should have tool results for both tools
        tool_results = messages[2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[1]["tool_use_id"] == "tool_2"
    
    def test_system_prompt_content(self):
        """Test that the system prompt contains expected instructions"""
        # Verify system prompt has key components
        assert "search_course_content" in AIGenerator.SYSTEM_PROMPT
        assert "get_course_outline" in AIGenerator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in AIGenerator.SYSTEM_PROMPT
        assert "Brief, Concise and focused" in AIGenerator.SYSTEM_PROMPT
        assert "Educational" in AIGenerator.SYSTEM_PROMPT
    
    def test_base_params_structure(self):
        """Test that base parameters are properly structured"""
        generator = AIGenerator("test-key", "test-model")
        
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800
        assert len(generator.base_params) == 3
    
    def test_generate_response_api_exception(self, mock_anthropic_client):
        """Test handling of API exceptions during response generation"""
        # Setup mock to raise exception
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Execute and verify exception is raised
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")
        
        assert "API Error" in str(exc_info.value)
    
    def test_generate_response_empty_query(self, mock_anthropic_client):
        """Test response generation with empty query"""
        # Setup mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "I need more information to help you."
        mock_response.content = [mock_text_content]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        # Execute
        result = generator.generate_response("")
        
        # Verify it still processes (doesn't filter empty queries)
        assert result == "I need more information to help you."
    
    def test_tool_execution_without_tool_manager(self, mock_anthropic_client):
        """Test tool execution when tool_manager is None"""
        # Setup mock response with tool use
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.text = "I tried to use a tool but couldn't."
        mock_response.content = [mock_tool_content]
        
        mock_anthropic_client.messages.create.return_value = mock_response
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        tools = [{"name": "test_tool"}]
        
        # Execute with tool_manager=None
        result = generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=None
        )
        
        # Verify it returns the response directly (doesn't execute tools)
        # This tests the condition where stop_reason == "tool_use" but tool_manager is None
        assert result == "I tried to use a tool but couldn't."
        
        # Should only make one API call
        assert mock_anthropic_client.messages.create.call_count == 1
    
    def test_generate_response_two_tool_rounds(self, mock_anthropic_client, mock_tool_manager):
        """Test response generation with two rounds of tool calls"""
        # Setup first response with tool use
        mock_round1_response = Mock()
        mock_round1_response.stop_reason = "tool_use"
        mock_tool1_content = Mock()
        mock_tool1_content.type = "tool_use"
        mock_tool1_content.name = "get_course_outline"
        mock_tool1_content.id = "tool_1"
        mock_tool1_content.input = {"course_name": "Python Course"}
        mock_round1_response.content = [mock_tool1_content]
        
        # Setup second response with tool use
        mock_round2_response = Mock()
        mock_round2_response.stop_reason = "tool_use"
        mock_tool2_content = Mock()
        mock_tool2_content.type = "tool_use"
        mock_tool2_content.name = "search_course_content"
        mock_tool2_content.id = "tool_2"
        mock_tool2_content.input = {"query": "lesson 4 variables"}
        mock_round2_response.content = [mock_tool2_content]
        
        # Setup final response (no tool use)
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "Lesson 4 covers variables and data types in Python."
        mock_final_response.content = [mock_text_content]
        
        # Mock client to return responses in sequence
        mock_anthropic_client.messages.create.side_effect = [
            mock_round1_response, mock_round2_response, mock_final_response
        ]
        
        # Mock tool manager with different results for each tool
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 4 - Variables and Data Types",
            "Variables are used to store data in Python..."
        ]
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "What does lesson 4 of Python Course cover?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify final result
        assert result == "Lesson 4 covers variables and data types in Python."
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        calls = mock_tool_manager.execute_tool.call_args_list
        assert calls[0][0] == ("get_course_outline",)
        assert calls[0][1] == {"course_name": "Python Course"}
        assert calls[1][0] == ("search_course_content",)
        assert calls[1][1] == {"query": "lesson 4 variables"}
        
        # Verify three API calls were made (round1 + round2 + final)
        assert mock_anthropic_client.messages.create.call_count == 3
    
    def test_generate_response_max_rounds_reached(self, mock_anthropic_client, mock_tool_manager):
        """Test that max rounds (2) is enforced and final call made without tools"""
        # Setup responses that always want to use tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]
        
        # Final response without tools
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "Final answer after 2 tool rounds."
        mock_final_response.content = [mock_text_content]
        
        # Return tool use for first 2 calls, then final response
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response, mock_tool_response, mock_final_response
        ]
        
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        tools = [{"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify final result
        assert result == "Final answer after 2 tool rounds."
        
        # Verify exactly 2 tool executions (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify 3 API calls (2 tool rounds + final without tools)
        assert mock_anthropic_client.messages.create.call_count == 3
        
        # Verify final call had no tools parameter
        final_call_args = mock_anthropic_client.messages.create.call_args_list[2]
        assert "tools" not in final_call_args[1]
    
    def test_generate_response_tool_error_handling(self, mock_anthropic_client, mock_tool_manager):
        """Test graceful handling of tool execution errors"""
        # Setup response with tool use
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_content = Mock()
        mock_tool_content.type = "tool_use"
        mock_tool_content.name = "search_course_content"
        mock_tool_content.id = "tool_123"
        mock_tool_content.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_content]
        
        # Final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_text_content = Mock()
        mock_text_content.text = "I apologize, there was an error with the search."
        mock_final_response.content = [mock_text_content]
        
        mock_anthropic_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
        
        # Mock tool manager to raise exception
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        generator = AIGenerator("test-key", "test-model")
        generator.client = mock_anthropic_client
        
        tools = [{"name": "search_course_content"}]
        
        # Execute
        result = generator.generate_response(
            "Search for something",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Should complete successfully despite tool error
        assert result == "I apologize, there was an error with the search."
        
        # Verify tool execution was attempted
        mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify conversation continued with error in tool results
        final_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        messages = final_call_args[1]["messages"]
        tool_result_msg = messages[2]["content"][0]
        assert tool_result_msg["type"] == "tool_result"
        assert "Error executing tool" in tool_result_msg["content"]
        assert tool_result_msg["is_error"] is True