"""
Middleware module for FastAPI authentication.
"""

import base64
import logging

from fastapi import Request, HTTPException

from ...config import GEMINI_AUTH_PASSWORD

logger = logging.getLogger(__name__)


def authenticate_user(request: Request) -> str:
    """
    Authenticate the user with multiple methods.

    Args:
        request: FastAPI request object

    Returns:
        Username/identifier of authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    # If no password is set, skip authentication
    if not GEMINI_AUTH_PASSWORD:
        return "anonymous"

    # Check API key in query parameters
    api_key = request.query_params.get("key")
    if api_key and api_key == GEMINI_AUTH_PASSWORD:
        return "api_key_user"

    # Check x-goog-api-key header
    goog_api_key = request.headers.get("x-goog-api-key", "")
    if goog_api_key and goog_api_key == GEMINI_AUTH_PASSWORD:
        return "goog_api_key_user"

    # Check Bearer token
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        bearer_token = auth_header[7:]
        if bearer_token == GEMINI_AUTH_PASSWORD:
            return "bearer_user"

    # Check Basic auth
    if auth_header.startswith("Basic "):
        try:
            encoded = auth_header[6:]
            decoded = base64.b64decode(encoded).decode("utf-8", "ignore")
            _, password = decoded.split(":", 1)
            if password == GEMINI_AUTH_PASSWORD:
                return "basic_user"
        except (ValueError, UnicodeDecodeError) as e:
            logger.debug(f"Basic auth decode failed: {e}")

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Basic"},
    )
