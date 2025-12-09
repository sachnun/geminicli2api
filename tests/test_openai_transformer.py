"""Tests for OpenAI transformer module."""

import json
import pytest

from src.routes.transformers.openai import (
    openai_request_to_gemini,
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
    _parse_data_uri,
    _extract_images_from_text,
    _extract_content_and_reasoning,
    _map_finish_reason,
    _process_message_content,
    _transform_openai_tools_to_gemini,
    _transform_tool_choice_to_gemini,
    _build_thinking_config,
)
from src.schemas import ChatCompletionRequest, ChatMessage


class TestParseDataUri:
    """Tests for _parse_data_uri function."""

    def test_valid_image_data_uri(self):
        """Valid image data URI should return inline data dict."""
        uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        result = _parse_data_uri(uri)
        assert result is not None
        assert "inlineData" in result
        assert result["inlineData"]["mimeType"] == "image/png"
        assert result["inlineData"]["data"] == "iVBORw0KGgoAAAANSUhEUg=="

    def test_valid_jpeg_data_uri(self):
        """Valid JPEG data URI should return inline data dict."""
        uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
        result = _parse_data_uri(uri)
        assert result is not None
        assert result["inlineData"]["mimeType"] == "image/jpeg"

    def test_non_image_data_uri(self):
        """Non-image data URI should return None."""
        uri = "data:text/plain;base64,SGVsbG8gV29ybGQ="
        result = _parse_data_uri(uri)
        assert result is None

    def test_non_data_uri(self):
        """Non-data URI should return None."""
        result = _parse_data_uri("https://example.com/image.png")
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        result = _parse_data_uri("")
        assert result is None

    def test_malformed_data_uri(self):
        """Malformed data URI should return None."""
        result = _parse_data_uri("data:malformed")
        assert result is None


class TestExtractImagesFromText:
    """Tests for _extract_images_from_text function."""

    def test_plain_text(self):
        """Plain text should return single text part."""
        result = _extract_images_from_text("Hello, world!")
        assert len(result) == 1
        assert result[0] == {"text": "Hello, world!"}

    def test_text_with_data_uri_image(self):
        """Text with data URI image should extract inline data."""
        text = "Here is an image: ![alt](data:image/png;base64,abc123)"
        result = _extract_images_from_text(text)
        assert len(result) == 2
        assert result[0] == {"text": "Here is an image: "}
        assert "inlineData" in result[1]

    def test_text_with_url_image(self):
        """Text with URL image should keep as text."""
        text = "![alt](https://example.com/image.png)"
        result = _extract_images_from_text(text)
        assert len(result) == 1
        assert result[0]["text"] == "![alt](https://example.com/image.png)"

    def test_empty_string(self):
        """Empty string should return text part with empty string."""
        result = _extract_images_from_text("")
        assert result == [{"text": ""}]

    def test_none_like_input(self):
        """Empty string should return text part with empty string."""
        result = _extract_images_from_text("")
        assert result == [{"text": ""}]


class TestExtractContentAndReasoning:
    """Tests for _extract_content_and_reasoning function."""

    def test_text_only(self):
        """Text-only parts should return content."""
        parts = [{"text": "Hello"}, {"text": "World"}]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == "Hello\n\nWorld"
        assert reasoning == ""
        assert tool_calls == []

    def test_thought_parts(self):
        """Thought parts should be extracted as reasoning."""
        parts = [
            {"text": "Let me think...", "thought": True},
            {"text": "The answer is 42"},
        ]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == "The answer is 42"
        assert reasoning == "Let me think..."
        assert tool_calls == []

    def test_function_call_parts(self):
        """Function call parts should be extracted as tool_calls."""
        parts = [{"functionCall": {"name": "get_weather", "args": {"city": "Tokyo"}}}]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_inline_image_data(self):
        """Inline image data should be converted to markdown."""
        parts = [{"inlineData": {"mimeType": "image/png", "data": "abc123"}}]
        content, reasoning, tool_calls = _extract_content_and_reasoning(parts)
        assert "![image](data:image/png;base64,abc123)" in content

    def test_empty_parts(self):
        """Empty parts list should return empty values."""
        content, reasoning, tool_calls = _extract_content_and_reasoning([])
        assert content == ""
        assert reasoning == ""
        assert tool_calls == []


class TestMapFinishReason:
    """Tests for _map_finish_reason function."""

    def test_stop_reason(self):
        """STOP should map to stop."""
        assert _map_finish_reason("STOP") == "stop"

    def test_max_tokens_reason(self):
        """MAX_TOKENS should map to length."""
        assert _map_finish_reason("MAX_TOKENS") == "length"

    def test_safety_reason(self):
        """SAFETY should map to content_filter."""
        assert _map_finish_reason("SAFETY") == "content_filter"

    def test_recitation_reason(self):
        """RECITATION should map to content_filter."""
        assert _map_finish_reason("RECITATION") == "content_filter"

    def test_none_reason(self):
        """None should return None."""
        assert _map_finish_reason(None) is None

    def test_unknown_reason(self):
        """Unknown reason should return None."""
        assert _map_finish_reason("UNKNOWN") is None


class TestProcessMessageContent:
    """Tests for _process_message_content function."""

    def test_string_content(self):
        """String content should be processed."""
        result = _process_message_content("Hello, world!")
        assert len(result) == 1
        assert result[0] == {"text": "Hello, world!"}

    def test_list_content_with_text(self):
        """List content with text type should be processed."""
        content = [{"type": "text", "text": "Hello"}]
        result = _process_message_content(content)
        assert len(result) == 1
        assert result[0] == {"text": "Hello"}

    def test_list_content_with_image_url(self):
        """List content with image_url should be processed."""
        content = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc123"},
            }
        ]
        result = _process_message_content(content)
        assert len(result) == 1
        assert "inlineData" in result[0]
        assert result[0]["inlineData"]["mimeType"] == "image/png"

    def test_none_content(self):
        """None content should return empty list."""
        result = _process_message_content(None)
        assert result == []

    def test_non_list_non_string_content(self):
        """Non-list non-string content should be converted to string."""
        result = _process_message_content(123)
        assert len(result) == 1
        assert result[0] == {"text": "123"}


class TestTransformOpenaiToolsToGemini:
    """Tests for _transform_openai_tools_to_gemini function."""

    def test_function_tool(self):
        """Function tool should be transformed."""
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
        func_decls, has_web_search = _transform_openai_tools_to_gemini(tools)
        assert len(func_decls) == 1
        assert func_decls[0]["name"] == "get_weather"
        assert func_decls[0]["description"] == "Get weather info"
        assert not has_web_search

    def test_web_search_tool(self):
        """Web search tool should be detected."""
        tools = [{"type": "web_search"}]
        func_decls, has_web_search = _transform_openai_tools_to_gemini(tools)
        assert len(func_decls) == 0
        assert has_web_search

    def test_mixed_tools(self):
        """Mixed tools should be processed correctly."""
        tools = [
            {"type": "web_search"},
            {
                "type": "function",
                "function": {"name": "test_func", "description": "Test"},
            },
        ]
        func_decls, has_web_search = _transform_openai_tools_to_gemini(tools)
        assert len(func_decls) == 1
        assert has_web_search

    def test_empty_tools(self):
        """Empty tools list should return empty results."""
        func_decls, has_web_search = _transform_openai_tools_to_gemini([])
        assert len(func_decls) == 0
        assert not has_web_search


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

    def test_unknown_choice(self):
        """Unknown choice should return None."""
        result = _transform_tool_choice_to_gemini("unknown")
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
        result = _build_thinking_config("gemini-2.5-pro", "high")
        assert result is not None
        assert result["thinkingBudget"] > 0

    def test_disabled_reasoning(self):
        """Disabled reasoning should use minimal budget."""
        result = _build_thinking_config("gemini-2.5-flash", "none")
        assert result is not None
        assert result["thinkingBudget"] == 0

    def test_flash_image_model(self):
        """Flash image model should return None."""
        result = _build_thinking_config("gemini-2.5-flash-image", "high")
        assert result is None


class TestOpenaiRequestToGemini:
    """Tests for openai_request_to_gemini function."""

    def test_basic_request(self):
        """Basic request should be transformed correctly."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Hello!")],
        )
        result = openai_request_to_gemini(request)
        assert result["model"] == "gemini-2.5-pro"
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello!"

    def test_request_with_system_message(self):
        """System message should be converted to user role."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hi"),
            ],
        )
        result = openai_request_to_gemini(request)
        assert len(result["contents"]) == 2
        # System message is mapped to user role
        assert result["contents"][0]["role"] == "user"

    def test_request_with_temperature(self):
        """Temperature should be included in generation config."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Hello!")],
            temperature=0.7,
        )
        result = openai_request_to_gemini(request)
        assert result["generationConfig"]["temperature"] == 0.7

    def test_request_with_max_tokens(self):
        """Max tokens should be included in generation config."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Hello!")],
            max_tokens=1024,
        )
        result = openai_request_to_gemini(request)
        assert result["generationConfig"]["maxOutputTokens"] == 1024

    def test_request_with_json_format(self):
        """JSON response format should set mime type."""
        request = ChatCompletionRequest(
            model="gemini-2.5-pro",
            messages=[ChatMessage(role="user", content="Hello!")],
            response_format={"type": "json_object"},
        )
        result = openai_request_to_gemini(request)
        assert result["generationConfig"]["responseMimeType"] == "application/json"


class TestGeminiResponseToOpenai:
    """Tests for gemini_response_to_openai function."""

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
            ]
        }
        result = gemini_response_to_openai(gemini_response, "gemini-2.5-pro")
        assert result["object"] == "chat.completion"
        assert result["model"] == "gemini-2.5-pro"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "Hello! How can I help?"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_response_with_thinking(self):
        """Response with thinking should include reasoning_content."""
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
        result = gemini_response_to_openai(gemini_response, "gemini-2.5-pro")
        assert result["choices"][0]["message"]["content"] == "The answer is 42"
        assert result["choices"][0]["message"]["reasoning_content"] == "Let me think..."

    def test_response_with_function_call(self):
        """Response with function call should include tool_calls."""
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
        result = gemini_response_to_openai(gemini_response, "gemini-2.5-pro")
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1
        assert (
            result["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "get_weather"
        )
        assert result["choices"][0]["finish_reason"] == "tool_calls"


class TestGeminiStreamChunkToOpenai:
    """Tests for gemini_stream_chunk_to_openai function."""

    def test_basic_chunk(self):
        """Basic chunk should be transformed correctly."""
        chunk = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    }
                }
            ]
        }
        result = gemini_stream_chunk_to_openai(chunk, "gemini-2.5-pro", "chatcmpl-123")
        assert result["object"] == "chat.completion.chunk"
        assert result["id"] == "chatcmpl-123"
        assert result["choices"][0]["delta"]["content"] == "Hello"

    def test_chunk_with_finish_reason(self):
        """Chunk with finish reason should be included."""
        chunk = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Done"}]},
                    "finishReason": "STOP",
                }
            ]
        }
        result = gemini_stream_chunk_to_openai(chunk, "gemini-2.5-pro", "chatcmpl-123")
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_chunk_with_reasoning(self):
        """Chunk with reasoning should include reasoning_content."""
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
        result = gemini_stream_chunk_to_openai(chunk, "gemini-2.5-pro", "chatcmpl-123")
        assert result["choices"][0]["delta"]["reasoning_content"] == "Thinking..."
