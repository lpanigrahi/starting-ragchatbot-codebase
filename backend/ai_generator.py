import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
- **search_course_content**: Search specific course content and detailed educational materials
- **get_course_outline**: Get complete course outlines including title, course link, and full lesson lists

Tool Usage Guidelines:
- **Content questions**: Use search_course_content for specific course materials and lessons
- **Outline questions**: Use get_course_outline for course structure, lesson lists, and overview information
- **Sequential tool usage**: You may use tools multiple times to gather comprehensive information
- **Tool reasoning**: After each tool result, decide if additional tools would help provide a better answer
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content first, then answer
- **Course outline questions**: Use get_course_outline first, then answer
- **Complex questions**: Use multiple tools as needed to gather complete information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

When responding to outline queries, always include:
- Course title
- Course link 
- Complete lesson list with numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with support for sequential tool calls (up to max_rounds).
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool execution rounds (default: 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation state
        messages = [{"role": "user", "content": query}]
        current_round = 0
        
        # Main execution loop
        while current_round < max_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content
            }
            
            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            response = self._make_api_call(api_params)
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use" and tool_manager and tools:
                # Execute tools and update conversation state
                messages = self._execute_tools_and_update_conversation(
                    response, messages, tool_manager
                )
                current_round += 1
                # Continue loop for next round
            else:
                # No tool use - return final response
                return response.content[0].text
        
        # Reached max rounds - make final call without tools
        return self._make_final_call_without_tools(messages, system_content)
    
    def _make_api_call(self, api_params: Dict[str, Any]):
        """Make API call with error handling"""
        try:
            return self.client.messages.create(**api_params)
        except Exception as e:
            # Log error and re-raise for now
            # Could implement retry logic here in future
            raise e
    
    def _execute_tools_and_update_conversation(self, response, messages: List, tool_manager) -> List:
        """
        Execute all tool calls from response and update conversation history.
        
        Args:
            response: Claude's response containing tool use blocks
            messages: Current conversation messages
            tool_manager: Manager to execute tools
            
        Returns:
            Updated messages list with tool execution results
        """
        # Add Claude's tool use response to conversation
        messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        return messages
    
    def _make_final_call_without_tools(self, messages: List, system_content: str) -> str:
        """
        Make final API call without tools when max rounds reached.
        
        Args:
            messages: Complete conversation history
            system_content: System prompt content
            
        Returns:
            Final response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
            # Explicitly no tools parameter
        }
        
        final_response = self._make_api_call(final_params)
        return final_response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text