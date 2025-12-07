"""Shared utility functions for route handlers."""

import json
import logging
from typing import Any, Callable, Union

from fastapi import Response

from ..services.gemini_client import send_gemini_request

logger = logging.getLogger(__name__)


def create_route_error_response(
    message: str, error_type: str = "api_error", code: str | None = None
) -> dict[str, Any]:
    """Create a standardized error response for route handlers."""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }


def parse_response_body(body: bytes) -> dict[str, Any] | None:
    """Parse JSON response body, returning None on failure."""
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def decode_chunk(chunk: bytes | memoryview | str) -> str:
    """Decode a streaming chunk to string."""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", "ignore")
    elif isinstance(chunk, memoryview):
        return bytes(chunk).decode("utf-8", "ignore")
    else:
        return str(chunk)


def determine_error_type(status_code: int) -> str:
    """Determine error type based on HTTP status code."""
    if status_code == 404:
        return "invalid_request_error"
    return "api_error"


def _parse_response_body_from_response(response: Response) -> dict[str, Any]:
    """Parse response body from a Response object to dict."""
    body = response.body
    if isinstance(body, bytes):
        body_str = body.decode("utf-8", "ignore")
    elif isinstance(body, memoryview):
        body_str = bytes(body).decode("utf-8", "ignore")
    else:
        body_str = str(body)
    return json.loads(body_str)


def _create_json_error_response(
    message: str, status_code: int, error_type: str = "api_error"
) -> Response:
    """Create a JSON error response."""
    from ..config import create_error_response

    return Response(
        content=json.dumps(create_error_response(message, error_type, status_code)),
        status_code=status_code,
        media_type="application/json",
    )


async def handle_non_streaming_response(
    gemini_payload: dict[str, Any],
    model: str,
    transformer_func: Callable[[dict[str, Any], str], dict[str, Any]],
    log_prefix: str = "Processed response",
) -> Union[dict[str, Any], Response]:
    """
    Generic handler for non-streaming Gemini requests.

    Args:
        gemini_payload: Prepared Gemini request payload
        model: Model name for response
        transformer_func: Function to transform Gemini response to target format
        log_prefix: Prefix for success log message

    Returns:
        Transformed response dict or error Response
    """
    response = await send_gemini_request(gemini_payload, is_streaming=False)

    # Handle error responses
    if isinstance(response, Response) and response.status_code != 200:
        logger.error(f"Gemini API error: {response.status_code}")

        try:
            error_data = _parse_response_body_from_response(response)
            if "error" in error_data:
                error = error_data["error"]
                error_type = determine_error_type(response.status_code)
                return _create_json_error_response(
                    error.get("message", f"API error: {response.status_code}"),
                    response.status_code,
                    error.get("type", error_type),
                )
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        error_type = determine_error_type(response.status_code)
        return _create_json_error_response(
            f"API error: {response.status_code}", response.status_code, error_type
        )

    # Parse and transform response
    try:
        gemini_response = _parse_response_body_from_response(response)
        transformed_response = transformer_func(gemini_response, model)
        logger.info(f"{log_prefix} for model: {model}")
        return transformed_response
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"Failed to parse response: {e}")
        return _create_json_error_response(f"Failed to process response: {e}", 500)
