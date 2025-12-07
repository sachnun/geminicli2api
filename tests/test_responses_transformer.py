"""Tests for Responses API transformer module."""

import json
import pytest

from src.routes.transformers.responses import (
    responses_request_to_gemini,
    gemini_response_to_responses,
    gemini_stream_chunk_to_responses_events,
    _transform_responses_tools_to_gemini,
    _transform_tool_choice_to_gemini,
    _build_thinking_config,
    _process_input_item,
    _extract_content_and_function_calls,
)
from src.schemas.responses import ResponsesRequest


class TestTransformResponsesToolsToGemini:
    """Tests for _transform_responses_tools_to_gemini function."""

    def test_function_tool_flat_structure(self):
        """Function tool with flat structure should be transformed."""
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        func_decls, has_web_search = _transform_responses_tools_to_gemini(tools)
        assert len(func_decls) == 1
        assert func_decls[0]["name"] == "get_weather"
        assert func_decls[0]["description"] == "Get weather info"
        assert not has_web_search

    def test_function_tool_nested_structure(self):
        """Function tool with OpenAI nested structure should be transformed."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        func_decls, has_web_search = _transform_responses_tools_to_gemini(tools)
        assert len(func_decls) == 1
        assert func_decls[0]["name"] == "get_weather"
        assert not has_web_search

    def test_web_search_tool(self):
        """Web search tool should be detected."""
        tools = [{"type": "web_search"}]
        func_decls, has_web_search = _transform_responses_tools_to_gemini(tools)
        assert len(func_decls) == 0
        assert has_web_search

    def test_web_search_preview_tool(self):
        """Web search preview tool should be detected."""
        tools = [{"type": "web_search_preview"}]
        func_decls, has_web_search = _transform_responses_tools_to_gemini(tools)
        assert len(func_decls) == 0
        assert has_web_search

    def test_mixed_tools(self):
        """Mixed tools should be processed correctly."""
        tools = [
            {"type": "web_search"},
            {"type": "function", "name": "test_func", "description": "Test"},
        ]
        func_decls, has_web_search = _transform_responses_tools_to_gemini(tools)
        assert len(func_decls) == 1
        assert has_web_search

    def test_empty_tools(self):
        """Empty tools list should return empty results."""
        func_decls, has_web_search = _transform_responses_tools_to_gemini([])
        assert len(func_decls) == 0
        assert not has_web_search

    def test_tool_without_name(self):
        """Tool without name should be skipped."""
        tools = [{"type": "function", "description": "No name function"}]
        func_decls, has_web_search = _transform_responses_tools_to_gemini(tools)
        assert len(func_decls) == 0


class TestTransformToolChoiceToGemini:
    """Tests for _transform_tool_choice_to_gemini function."""

    def test_auto_choice(self):
        """Auto choice should map to AUTO mode."""
        result = _transform_tool_choice_to_gemini("auto")
        assert result == {"mode": "AUTO"}

    def test_none_choice(self):
        """None choice should map to NONE mode."""
        result = _transform_tool_choice_to_gemini("none")
        assert result == {"mode": "NONE"}

    def test_required_choice(self):
        """Required choice should map to ANY mode."""
        result = _transform_tool_choice_to_gemini("required")
        assert result == {"mode": "ANY"}

    def test_specific_function_choice(self):
        """Specific function choice should map correctly."""
        choice = {"type": "function", "function": {"name": "my_func"}}
        result = _transform_tool_choice_to_gemini(choice)
        assert result == {"mode": "ANY", "allowedFunctionNames": ["my_func"]}

    def test_null_choice(self):
        """Null choice should return None."""
        result = _transform_tool_choice_to_gemini(None)
        assert result is None


class TestBuildThinkingConfig:
    """Tests for _build_thinking_config function."""

    def test_default_thinking(self):
        """Default should return auto thinking budget."""
        result = _build_thinking_config("gemini-2.5-pro", None)
        assert result is not None
        assert "thinkingBudget" in result
        assert "includeThoughts" in result

    def test_high_reasoning_effort(self):
        """High reasoning effort should use max budget."""
        reasoning = {"effort": "high"}
        result = _build_thinking_config("gemini-2.5-pro", reasoning)
        assert result is not None
        assert result["thinkingBudget"] > 0

    def test_disabled_reasoning(self):
        """Disabled reasoning should use minimal budget."""
        reasoning = {"effort": "none"}
        result = _build_thinking_config("gemini-2.5-flash", reasoning)
        assert result is not None
        assert result["thinkingBudget"] == 0

    def test_medium_reasoning(self):
        """Medium reasoning should use default budget."""
        reasoning = {"effort": "medium"}
        result = _build_thinking_config("gemini-2.5-pro", reasoning)
        assert result is not None
        # Default budget is -1 (auto)
        assert result["thinkingBudget"] == -1

    def test_flash_image_model(self):
        """Flash image model should return None."""
        reasoning = {"effort": "high"}
        result = _build_thinking_config("gemini-2.5-flash-image", reasoning)
        assert result is None


class TestProcessInputItem:
    """Tests for _process_input_item function."""

    def test_user_message(self):
        """User message should be processed correctly."""
        item = {"type": "message", "role": "user", "content": "Hello!"}
        result = _process_input_item(item)
        assert result is not None
        assert result["role"] == "user"
        assert result["parts"][0]["text"] == "Hello!"

    def test_assistant_message(self):
        """Assistant message should be processed with model role."""
        item = {"type": "message", "role": "assistant", "content": "Hi there!"}
        result = _process_input_item(item)
        assert result is not None
        assert result["role"] == "model"

    def test_function_call_output(self):
        """Function call output should return function response."""
        item = {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": '{"result": "success"}',
            "name": "test_func",
        }
        result = _process_input_item(item)
        assert result is not None
        assert result["role"] == "user"
        assert "functionResponse" in result["parts"][0]
        assert result["parts"][0]["functionResponse"]["name"] == "test_func"

    def test_message_with_list_content(self):
        """Message with list content should be processed."""
        item = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        }
        result = _process_input_item(item)
        assert result is not None
        assert len(result["parts"]) == 2

    def test_message_with_image_url(self):
        """Message with image URL should process inline data."""
        item = {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                }
            ],
        }
        result = _process_input_item(item)
        assert result is not None
        assert "inlineData" in result["parts"][0]

    def test_empty_content(self):
        """Empty content should return empty text part."""
        item = {"type": "message", "role": "user", "content": ""}
        result = _process_input_item(item)
        assert result is not None
        assert result["parts"][0]["text"] == ""


class TestExtractContentAndFunctionCalls:
    """Tests for _extract_content_and_function_calls function."""

    def test_text_only(self):
        """Text-only parts should return content."""
        parts = [{"text": "Hello"}, {"text": "World"}]
        content, reasoning, func_calls = _extract_content_and_function_calls(parts)
        assert content == "Hello\n\nWorld"
        assert reasoning == ""
        assert func_calls == []

    def test_thought_parts(self):
        """Thought parts should be extracted as reasoning."""
        parts = [
            {"text": "Let me think...", "thought": True},
            {"text": "The answer is 42"},
        ]
        content, reasoning, func_calls = _extract_content_and_function_calls(parts)
        assert content == "The answer is 42"
        assert reasoning == "Let me think..."

    def test_function_call_parts(self):
        """Function call parts should be extracted."""
        parts = [{"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}]
        content, reasoning, func_calls = _extract_content_and_function_calls(parts)
        assert content == ""
        assert len(func_calls) == 1
        assert func_calls[0]["type"] == "function_call"
        assert func_calls[0]["name"] == "get_weather"
        assert json.loads(func_calls[0]["arguments"]) == {"city": "Tokyo"}

    def test_inline_image_data(self):
        """Inline image data should be converted to markdown."""
        parts = [{"inlineData": {"mimeType": "image/png", "data": "abc123"}}]
        content, reasoning, func_calls = _extract_content_and_function_calls(parts)
        assert "![image](data:image/png;base64,abc123)" in content


class TestResponsesRequestToGemini:
    """Tests for responses_request_to_gemini function."""

    def test_string_input(self):
        """String input should be transformed correctly."""
        request = ResponsesRequest(model="gemini-2.5-pro", input="Hello!")
        result = responses_request_to_gemini(request)
        assert result["model"] == "gemini-2.5-pro"
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello!"

    def test_list_input(self):
        """List input should be transformed correctly."""
        request = ResponsesRequest(
            model="gemini-2.5-pro",
            input=[{"type": "message", "role": "user", "content": "Hello!"}],
        )
        result = responses_request_to_gemini(request)
        assert len(result["contents"]) == 1
        assert result["contents"][0]["parts"][0]["text"] == "Hello!"

    def test_with_instructions(self):
        """Instructions should be included as system instruction."""
        request = ResponsesRequest(
            model="gemini-2.5-pro",
            input="Hi",
            instructions="You are a helpful assistant.",
        )
        result = responses_request_to_gemini(request)
        assert "systemInstruction" in result
        assert (
            result["systemInstruction"]["parts"][0]["text"]
            == "You are a helpful assistant."
        )

    def test_with_generation_params(self):
        """Generation parameters should be included."""
        request = ResponsesRequest(
            model="gemini-2.5-pro",
            input="Hi",
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=2048,
        )
        result = responses_request_to_gemini(request)
        assert result["generationConfig"]["temperature"] == 0.7
        assert result["generationConfig"]["topP"] == 0.9
        assert result["generationConfig"]["maxOutputTokens"] == 2048

    def test_with_tools(self):
        """Tools should be included."""
        request = ResponsesRequest(
            model="gemini-2.5-pro",
            input="Hi",
            tools=[{"type": "function", "name": "test", "description": "Test func"}],
        )
        result = responses_request_to_gemini(request)
        assert "tools" in result
        assert "functionDeclarations" in result["tools"][0]

    def test_with_web_search(self):
        """Web search should be included."""
        request = ResponsesRequest(
            model="gemini-2.5-pro", input="Hi", tools=[{"type": "web_search"}]
        )
        result = responses_request_to_gemini(request)
        assert "tools" in result
        assert "googleSearch" in result["tools"][0]

    def test_with_reasoning(self):
        """Reasoning config should be included."""
        request = ResponsesRequest(
            model="gemini-2.5-pro",
            input="Hi",
            reasoning={"effort": "high"},
        )
        result = responses_request_to_gemini(request)
        assert "thinkingConfig" in result["generationConfig"]


class TestGeminiResponseToResponses:
    """Tests for gemini_response_to_responses function."""

    def test_basic_response(self):
        """Basic response should be transformed correctly."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello! How can I help?"}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30,
            },
        }
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-pro")
        assert result["object"] == "response"
        assert result["model"] == "gemini-2.5-pro"
        assert result["status"] == "completed"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output_text"] == "Hello! How can I help?"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20

    def test_response_with_reasoning(self):
        """Response with reasoning should include reasoning item."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Let me think...", "thought": True},
                            {"text": "The answer is 42"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-pro")
        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "reasoning"
        assert result["output"][1]["type"] == "message"

    def test_response_with_function_call(self):
        """Response with function call should include function_call item."""
        gemini_response = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"city": "Tokyo"},
                                }
                            }
                        ],
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-pro")
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "function_call"
        assert result["output"][0]["name"] == "get_weather"

    def test_empty_response(self):
        """Empty response should have empty output."""
        gemini_response = {"candidates": []}
        result = gemini_response_to_responses(gemini_response, "gemini-2.5-pro")
        assert result["output"] == []
        assert result["output_text"] is None


class TestGeminiStreamChunkToResponsesEvents:
    """Tests for gemini_stream_chunk_to_responses_events function."""

    def test_text_chunk(self):
        """Text chunk should produce output_text delta event."""
        chunk = {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}]
        }
        events = gemini_stream_chunk_to_responses_events(
            chunk, "gemini-2.5-pro", "resp_123"
        )
        assert len(events) == 1
        assert events[0]["type"] == "response.output_text.delta"
        assert events[0]["delta"] == "Hello"

    def test_reasoning_chunk(self):
        """Reasoning chunk should produce reasoning delta event."""
        chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Thinking...", "thought": True}],
                    }
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            chunk, "gemini-2.5-pro", "resp_123"
        )
        assert len(events) == 1
        assert events[0]["type"] == "response.reasoning.delta"
        assert events[0]["delta"] == "Thinking..."

    def test_function_call_chunk(self):
        """Function call chunk should produce item added and arguments done events."""
        chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "test_func", "args": {}}}],
                    }
                }
            ]
        }
        events = gemini_stream_chunk_to_responses_events(
            chunk, "gemini-2.5-pro", "resp_123"
        )
        assert len(events) == 2
        assert events[0]["type"] == "response.output_item.added"
        assert events[1]["type"] == "response.function_call_arguments.done"

    def test_empty_chunk(self):
        """Empty chunk should produce no events."""
        chunk = {"candidates": []}
        events = gemini_stream_chunk_to_responses_events(
            chunk, "gemini-2.5-pro", "resp_123"
        )
        assert len(events) == 0
