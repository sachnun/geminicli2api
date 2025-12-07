"""
OAuth module for handling OAuth2 authentication flow with Google APIs.
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import urlparse, parse_qs

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from ...config import (
    CLIENT_ID,
    CLIENT_SECRET,
    SCOPES,
    TOKEN_URI,
    AUTH_URI,
    OAUTH_REDIRECT_URI,
    OAUTH_CALLBACK_PORT,
)

logger = logging.getLogger(__name__)

# --- Global State with Thread Safety ---
_state_lock = threading.Lock()
_credentials_from_env: bool = False


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    auth_code: Optional[str] = None

    def do_GET(self) -> None:
        query_components = parse_qs(urlparse(self.path).query)
        code = query_components.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.auth_code = code
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<h1>OAuth authentication successful!</h1>"
                b"<p>You can close this window.</p>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Authentication failed.</h1><p>Please try again.</p>")

    def log_message(self, format: str, *args) -> None:
        """Suppress default HTTP server logging."""
        pass


def _run_oauth_flow() -> Optional[Credentials]:
    """Run interactive OAuth flow."""
    global _credentials_from_env

    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
        }
    }

    flow = Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=OAUTH_REDIRECT_URI
    )
    flow.oauth2session.scope = SCOPES

    auth_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent", include_granted_scopes="true"
    )

    print(f"\n{'=' * 60}")
    print("AUTHENTICATION REQUIRED")
    print(f"{'=' * 60}")
    print(f"Please open this URL in your browser:\n{auth_url}")
    print(f"{'=' * 60}\n")
    logger.info(f"OAuth URL: {auth_url}")

    server = HTTPServer(("", OAUTH_CALLBACK_PORT), _OAuthCallbackHandler)
    server.handle_request()

    auth_code = _OAuthCallbackHandler.auth_code
    if not auth_code:
        return None

    # Patch oauthlib to ignore scope warnings
    import oauthlib.oauth2.rfc6749.parameters as oauth_params

    original_validate = oauth_params.validate_token_parameters
    oauth_params.validate_token_parameters = lambda p: None

    try:
        flow.fetch_token(code=auth_code)
        creds = flow.credentials
        # flow.credentials returns OAuth2 Credentials for installed app flow
        if not isinstance(creds, Credentials):
            logger.error("Unexpected credentials type from OAuth flow")
            return None
        with _state_lock:
            _credentials_from_env = False

        # Import save_credentials here to avoid circular import
        from .credentials import save_credentials

        save_credentials(creds)
        logger.info("Authentication successful!")
        return creds
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return None
    finally:
        oauth_params.validate_token_parameters = original_validate


def refresh_access_token(creds: Credentials) -> bool:
    """
    Refresh the access token using the refresh token.

    Args:
        creds: Credentials object with refresh_token

    Returns:
        True if refresh succeeded, False otherwise
    """
    from google.auth.transport.requests import Request as GoogleAuthRequest

    if not creds.refresh_token:
        logger.warning("No refresh token available")
        return False

    try:
        creds.refresh(GoogleAuthRequest())
        logger.info("Access token refreshed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to refresh access token: {e}")
        return False


def get_oauth_token(allow_interactive: bool = True) -> Optional[Credentials]:
    """
    Get OAuth token, optionally running interactive flow.

    Args:
        allow_interactive: Whether to run interactive OAuth if no token exists

    Returns:
        Credentials object or None
    """
    from .credentials import get_credentials

    return get_credentials(allow_oauth_flow=allow_interactive)
