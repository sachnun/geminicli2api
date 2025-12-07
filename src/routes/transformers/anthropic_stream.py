"""
Anthropic Stream Processor - Handles Gemini to Anthropic SSE streaming conversion.

This module contains the AnthropicStreamProcessor class for converting Gemini streaming
chunks to Anthropic Server-Sent Events (SSE) format, maintaining state across multiple
chunks to properly emit events.
"""

import json
import uuid
from typing import Any, Dict, List


def _create_stream_message_start(model: str, message_id: str) -> Dict[str, Any]:
    """Create the initial message_start event for streaming."""
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


def _create_content_block_start(index: int, block_type: str = "text") -> Dict[str, Any]:
    """Create a content_block_start event."""
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


def _create_content_block_delta(
    index: int, delta_type: str, content: str
) -> Dict[str, Any]:
    """Create a content_block_delta event."""
    if delta_type == "text_delta":
        delta = {"type": "text_delta", "text": content}
    elif delta_type == "thinking_delta":
        delta = {"type": "thinking_delta", "thinking": content}
    elif delta_type == "input_json_delta":
        delta = {"type": "input_json_delta", "partial_json": content}
    else:
        delta = {"type": delta_type, "text": content}

    return {"type": "content_block_delta", "index": index, "delta": delta}


def _create_content_block_stop(index: int) -> Dict[str, Any]:
    """Create a content_block_stop event."""
    return {"type": "content_block_stop", "index": index}


def _create_message_delta(
    stop_reason: str = "end_turn", output_tokens: int = 0
) -> Dict[str, Any]:
    """Create a message_delta event."""
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }


def _create_message_stop() -> Dict[str, Any]:
    """Create a message_stop event."""
    return {"type": "message_stop"}


class AnthropicStreamProcessor:
    """
    Processor for converting Gemini streaming chunks to Anthropic SSE format.
    Maintains state across multiple chunks to properly emit events.
    """

    def __init__(self, model: str, include_thinking: bool = False):
        self.model = model
        self.include_thinking = include_thinking
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.current_block_index = -1
        self.current_block_type = None
        self.is_thinking = False
        self.has_started = False
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self.pending_tool_call = None

    def process_chunk(self, gemini_chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a Gemini streaming chunk and return Anthropic SSE events.

        Args:
            gemini_chunk: A chunk from Gemini streaming response

        Returns:
            List of Anthropic SSE events to emit
        """
        events = []

        # Emit message_start on first chunk
        if not self.has_started:
            events.append(_create_stream_message_start(self.model, self.message_id))
            self.has_started = True

        for candidate in gemini_chunk.get("candidates", []):
            parts = candidate.get("content", {}).get("parts", [])

            for part in parts:
                # Handle text/thinking parts
                if part.get("text") is not None:
                    is_thought = part.get("thought", False)
                    text = part.get("text", "")

                    if not text:
                        continue

                    # Skip thinking blocks if not requested
                    if is_thought and not self.include_thinking:
                        continue

                    # Determine block type
                    block_type = "thinking" if is_thought else "text"
                    delta_type = "thinking_delta" if is_thought else "text_delta"

                    # Check if we need to start a new block
                    if self.current_block_type != block_type:
                        # Close previous block if any
                        if self.current_block_index >= 0:
                            events.append(
                                _create_content_block_stop(self.current_block_index)
                            )

                        # Start new block
                        self.current_block_index += 1
                        self.current_block_type = block_type
                        events.append(
                            _create_content_block_start(
                                self.current_block_index, block_type
                            )
                        )

                    # Emit content delta
                    events.append(
                        _create_content_block_delta(
                            self.current_block_index, delta_type, text
                        )
                    )

                    self.output_tokens += len(text) // 4  # Rough estimate

                # Handle function calls
                elif part.get("functionCall"):
                    func_call = part["functionCall"]

                    # Close previous block if any
                    if (
                        self.current_block_index >= 0
                        and self.current_block_type != "tool_use"
                    ):
                        events.append(
                            _create_content_block_stop(self.current_block_index)
                        )

                    # Start tool_use block
                    self.current_block_index += 1
                    self.current_block_type = "tool_use"

                    tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    events.append(
                        {
                            "type": "content_block_start",
                            "index": self.current_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": func_call.get("name", ""),
                                "input": {},
                            },
                        }
                    )

                    # Emit input as delta
                    input_json = json.dumps(func_call.get("args", {}))
                    events.append(
                        _create_content_block_delta(
                            self.current_block_index, "input_json_delta", input_json
                        )
                    )

                    self.stop_reason = "tool_use"

            # Check finish reason
            finish_reason = candidate.get("finishReason", "")
            if finish_reason == "STOP":
                self.stop_reason = "end_turn"
            elif finish_reason == "MAX_TOKENS":
                self.stop_reason = "max_tokens"

        # Update token count from usage metadata
        usage_metadata = gemini_chunk.get("usageMetadata", {})
        if usage_metadata.get("candidatesTokenCount"):
            self.output_tokens = usage_metadata["candidatesTokenCount"]

        return events

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Generate final events to close the stream.

        Returns:
            List of final Anthropic SSE events
        """
        events = []

        # Close last content block if any
        if self.current_block_index >= 0:
            events.append(_create_content_block_stop(self.current_block_index))

        # Emit message_delta with final stats
        events.append(_create_message_delta(self.stop_reason, self.output_tokens))

        # Emit message_stop
        events.append(_create_message_stop())

        return events


def format_sse_event(event: Dict[str, Any]) -> str:
    """
    Format an event as an SSE data line.

    Args:
        event: Event dictionary

    Returns:
        Formatted SSE string
    """
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
