"""
Anthropic Format Transformers - Handles conversion between Anthropic/Claude and Gemini API formats.
This module contains all the logic for transforming requests and responses between the two formats.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .anthropic_stream import AnthropicStreamProcessor, format_sse_event
from ...schemas import AnthropicMessagesRequest
from ...config import (
    DEFAULT_SAFETY_SETTINGS,
    THINKING_MAX_BUDGETS,
    THINKING_MINIMAL_BUDGETS,
    THINKING_DEFAULT_BUDGET,
)
from ...models import (
    get_base_model_name,
    should_include_thoughts,
)


def _process_text_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a text content block into Gemini format."""
    return {"text": block.get("text", "")}


def _process_image_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process an image content block into Gemini format."""
    source = block.get("source", {})
    source_type = source.get("type", "")

    if source_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"inlineData": {"mimeType": media_type, "data": data}}

    if source_type == "url":
        url = source.get("url", "")
        return {"text": f"[Image: {url}]"}

    return None


def _process_tool_use_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Process a tool_use content block into Gemini functionCall format."""
    return {
        "functionCall": {
            "name": block.get("name", ""),
            "args": block.get("input", {}),
        }
    }


def _process_tool_result_block(
    block: Dict[str, Any], tool_use_id_to_name: Dict[str, str]
) -> Dict[str, Any]:
    """Process a tool_result content block into Gemini functionResponse format."""
    tool_use_id = block.get("tool_use_id", "")
    result_content = block.get("content", "")

    # Convert content to string if it's a list
    if isinstance(result_content, list):
        text_parts = [
            item.get("text", "")
            for item in result_content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        result_content = "\n".join(text_parts)

    # Look up the actual function name from tool_use_id
    func_name = tool_use_id_to_name.get(tool_use_id, tool_use_id)

    return {
        "functionResponse": {
            "name": func_name,
            "response": {
                "result": result_content,
                "is_error": block.get("is_error", False),
            },
        }
    }


def _process_thinking_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Process a thinking content block into Gemini format."""
    return {"text": block.get("thinking", ""), "thought": True}


def _process_content_block(
    block: Any, tool_use_id_to_name: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Process a single content block and return Gemini-formatted part.

    Args:
        block: Content block (dict or Pydantic model)
        tool_use_id_to_name: Mapping of tool_use_id to function name

    Returns:
        Gemini-formatted part dict, or None if block should be skipped
    """
    if not isinstance(block, dict):
        # Handle Pydantic models or other objects
        block_type = getattr(block, "type", None)
        if block_type == "text":
            return {"text": getattr(block, "text", "")}
        return None

    block_type = block.get("type", "text")

    if block_type == "text":
        return _process_text_block(block)

    if block_type == "image":
        return _process_image_block(block)

    if block_type == "tool_use":
        return _process_tool_use_block(block)

    if block_type == "tool_result":
        return _process_tool_result_block(block, tool_use_id_to_name)

    if block_type == "thinking":
        return _process_thinking_block(block)

    return None


def anthropic_request_to_gemini(
    anthropic_request: AnthropicMessagesRequest,
) -> Dict[str, Any]:
    """
    Transform an Anthropic messages request to Gemini format.

    Args:
        anthropic_request: Anthropic format request

    Returns:
        Dictionary in Gemini API format
    """
    contents = []

    # Build a mapping of tool_use_id -> function_name from assistant messages
    # This is needed to resolve function names when processing tool_result blocks
    tool_use_id_to_name: Dict[str, str] = {}
    for message in anthropic_request.messages:
        if message.role == "assistant" and isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id", "")
                    tool_name = block.get("name", "")
                    if tool_id and tool_name:
                        tool_use_id_to_name[tool_id] = tool_name

    # Handle system message/prompt
    system_instruction = None
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            system_instruction = {"parts": [{"text": anthropic_request.system}]}
        elif isinstance(anthropic_request.system, list):
            # System can be a list of content blocks
            system_parts = []
            for block in anthropic_request.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_parts.append({"text": block.get("text", "")})
            if system_parts:
                system_instruction = {"parts": system_parts}

    # Process each message in the conversation
    for message in anthropic_request.messages:
        role = message.role

        # Map Anthropic roles to Gemini roles
        if role == "assistant":
            role = "model"
        # "user" stays as "user"

        # Handle different content types
        parts = []
        content = message.content

        if isinstance(content, str):
            # Simple text content
            parts.append({"text": content})
        elif isinstance(content, list):
            # List of content blocks - dispatch to helper functions
            for block in content:
                part = _process_content_block(block, tool_use_id_to_name)
                if part is not None:
                    parts.append(part)

        if parts:
            contents.append({"role": role, "parts": parts})

    # Map Anthropic generation parameters to Gemini format
    generation_config = {}

    if anthropic_request.max_tokens is not None:
        generation_config["maxOutputTokens"] = anthropic_request.max_tokens

    if anthropic_request.temperature is not None:
        generation_config["temperature"] = anthropic_request.temperature

    if anthropic_request.top_p is not None:
        generation_config["topP"] = anthropic_request.top_p

    if anthropic_request.top_k is not None:
        generation_config["topK"] = anthropic_request.top_k

    if anthropic_request.stop_sequences:
        generation_config["stopSequences"] = anthropic_request.stop_sequences

    # Build the request payload
    request_payload = {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": get_base_model_name(anthropic_request.model),
    }

    # Add system instruction if present
    if system_instruction:
        request_payload["systemInstruction"] = system_instruction

    # Handle tools - check for web_search and function tools
    tools_list: List[Dict[str, Any]] = []
    has_web_search = False

    # Handle tools (function declarations and web_search)
    if anthropic_request.tools:
        for tool in anthropic_request.tools:
            # Check for web_search tool (custom extension)
            tool_type = getattr(tool, "type", None)
            tool_name = getattr(tool, "name", None)

            if tool_type in ("web_search", "web_search_preview"):
                has_web_search = True
                continue
            elif tool_name in ("web_search", "web_search_preview"):
                has_web_search = True
                continue

            # Regular function tool
            tool_def = {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": {
                    "type": "object",
                    "properties": tool.input_schema.properties or {},
                    "required": tool.input_schema.required or [],
                },
            }
            tools_list.append({"functionDeclarations": [tool_def]})

    # Add googleSearch tool if web_search was requested
    if has_web_search:
        tools_list.insert(0, {"googleSearch": {}})

    if tools_list:
        request_payload["tools"] = tools_list

    # Add thinking configuration
    if "gemini-2.5-flash-image" not in anthropic_request.model:
        thinking_budget = None

        # Check if thinking config is provided in request
        if anthropic_request.thinking:
            if (
                anthropic_request.thinking.type == "enabled"
                and anthropic_request.thinking.budget_tokens
            ):
                thinking_budget = anthropic_request.thinking.budget_tokens
            elif anthropic_request.thinking.type == "disabled":
                # Use minimal thinking from centralized config
                base_model = get_base_model_name(anthropic_request.model)
                for model_key, budget in THINKING_MINIMAL_BUDGETS.items():
                    if model_key in base_model:
                        thinking_budget = budget
                        break
                else:
                    # Default minimal budget if no match
                    thinking_budget = 0
        else:
            # Default: auto thinking
            thinking_budget = THINKING_DEFAULT_BUDGET

        if thinking_budget is not None:
            request_payload["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": thinking_budget,
                "includeThoughts": should_include_thoughts(anthropic_request.model),
            }

    return request_payload


def gemini_response_to_anthropic(
    gemini_response: Dict[str, Any],
    model: str,
    input_tokens: int = 0,
    include_thinking: bool = False,
) -> Dict[str, Any]:
    """
    Transform a Gemini API response to Anthropic messages format.

    Args:
        gemini_response: Response from Gemini API
        model: Model name to include in response
        input_tokens: Estimated input token count
        include_thinking: Whether to include thinking blocks in response (default False)

    Returns:
        Dictionary in Anthropic messages format
    """
    content_blocks = []
    stop_reason = "end_turn"
    output_tokens = 0

    for candidate in gemini_response.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])

        for part in parts:
            # Handle text parts
            if part.get("text") is not None:
                if part.get("thought", False):
                    # This is thinking content - only include if explicitly requested
                    if include_thinking:
                        content_blocks.append(
                            {"type": "thinking", "thinking": part.get("text", "")}
                        )
                    # Skip thinking blocks if not requested
                else:
                    # Regular text content
                    content_blocks.append(
                        {"type": "text", "text": part.get("text", "")}
                    )

            # Handle function calls (tool use)
            elif part.get("functionCall"):
                func_call = part["functionCall"]
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": func_call.get("name", ""),
                        "input": func_call.get("args", {}),
                    }
                )
                stop_reason = "tool_use"

            # Handle inline data (images in response)
            elif part.get("inlineData"):
                inline = part["inlineData"]
                mime = inline.get("mimeType", "image/png")
                data = inline.get("data", "")
                # Convert to markdown image reference in text
                content_blocks.append(
                    {
                        "type": "text",
                        "text": f"![Generated Image](data:{mime};base64,{data})",
                    }
                )

        # Map finish reason
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "STOP":
            stop_reason = "end_turn"
        elif finish_reason == "MAX_TOKENS":
            stop_reason = "max_tokens"
        elif finish_reason in ["SAFETY", "RECITATION"]:
            stop_reason = "end_turn"  # Anthropic doesn't have content_filter

    # Get token usage from response metadata if available
    usage_metadata = gemini_response.get("usageMetadata", {})
    input_tokens = usage_metadata.get("promptTokenCount", input_tokens)
    output_tokens = usage_metadata.get(
        "candidatesTokenCount", len(str(content_blocks)) // 4
    )

    # Ensure we have at least one content block
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def create_anthropic_stream_message_start(
    model: str, message_id: str
) -> Dict[str, Any]:
    """
    Create the initial message_start event for streaming.

    Args:
        model: Model name
        message_id: Unique message ID

    Returns:
        message_start event dictionary
    """
    return {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }


def create_anthropic_content_block_start(
    index: int, block_type: str = "text"
) -> Dict[str, Any]:
    """
    Create a content_block_start event.

    Args:
        index: Content block index
        block_type: Type of content block ("text", "thinking", "tool_use")

    Returns:
        content_block_start event dictionary
    """
    if block_type == "text":
        content_block = {"type": "text", "text": ""}
    elif block_type == "thinking":
        content_block = {"type": "thinking", "thinking": ""}
    elif block_type == "tool_use":
        content_block = {
            "type": "tool_use",
            "id": f"toolu_{uuid.uuid4().hex[:24]}",
            "name": "",
            "input": {},
        }
    else:
        content_block = {"type": block_type}

    return {
        "type": "content_block_start",
        "index": index,
        "content_block": content_block,
    }


def create_anthropic_content_block_delta(
    index: int, delta_type: str, content: str
) -> Dict[str, Any]:
    """
    Create a content_block_delta event.

    Args:
        index: Content block index
        delta_type: Type of delta ("text_delta", "thinking_delta", "input_json_delta")
        content: The delta content

    Returns:
        content_block_delta event dictionary
    """
    if delta_type == "text_delta":
        delta = {"type": "text_delta", "text": content}
    elif delta_type == "thinking_delta":
        delta = {"type": "thinking_delta", "thinking": content}
    elif delta_type == "input_json_delta":
        delta = {"type": "input_json_delta", "partial_json": content}
    else:
        delta = {"type": delta_type, "text": content}

    return {"type": "content_block_delta", "index": index, "delta": delta}


def create_anthropic_content_block_stop(index: int) -> Dict[str, Any]:
    """
    Create a content_block_stop event.

    Args:
        index: Content block index

    Returns:
        content_block_stop event dictionary
    """
    return {"type": "content_block_stop", "index": index}


def create_anthropic_message_delta(
    stop_reason: str = "end_turn", output_tokens: int = 0
) -> Dict[str, Any]:
    """
    Create a message_delta event.

    Args:
        stop_reason: Reason for stopping
        output_tokens: Total output tokens

    Returns:
        message_delta event dictionary
    """
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


def create_anthropic_message_stop() -> Dict[str, Any]:
    """
    Create a message_stop event.

    Returns:
        message_stop event dictionary
    """
    return {"type": "message_stop"}


def create_anthropic_ping() -> Dict[str, Any]:
    """
    Create a ping event for keeping connection alive.

    Returns:
        ping event dictionary
    """
    return {"type": "ping"}


def create_anthropic_error(error_type: str, message: str) -> Dict[str, Any]:
    """
    Create an error event.

    Args:
        error_type: Type of error (e.g., "invalid_request_error", "api_error")
        message: Error message

    Returns:
        error event dictionary
    """
    return {"type": "error", "error": {"type": error_type, "message": message}}
