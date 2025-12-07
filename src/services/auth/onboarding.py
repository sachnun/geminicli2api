"""
Onboarding module for Code Assist user onboarding.
"""

import json
import logging
import os
import threading
import time
from typing import Optional

import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

from ...config import (
    CODE_ASSIST_ENDPOINT,
    CREDENTIAL_FILE,
    ONBOARD_POLL_INTERVAL,
    ONBOARD_MAX_RETRIES,
)
from ...utils import get_user_agent, get_client_metadata

logger = logging.getLogger(__name__)

# --- Global State with Thread Safety ---
_state_lock = threading.Lock()
_user_project_id: Optional[str] = None
_onboarding_complete: bool = False


def onboard_user(creds: Credentials, project_id: str) -> None:
    """
    Ensure user is onboarded to Code Assist.

    Args:
        creds: Valid credentials
        project_id: Google Cloud project ID

    Raises:
        Exception: If onboarding fails
    """
    global _onboarding_complete

    with _state_lock:
        if _onboarding_complete:
            return

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            from .credentials import save_credentials

            save_credentials(creds)
        except Exception as e:
            raise Exception(f"Failed to refresh credentials: {e}")

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }

    try:
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        load_data = resp.json()

        # Get tier info
        tier = load_data.get("currentTier")
        if not tier:
            for allowed_tier in load_data.get("allowedTiers", []):
                if allowed_tier.get("isDefault"):
                    tier = allowed_tier
                    break
            if not tier:
                tier = {"id": "legacy-tier", "userDefinedCloudaicompanionProject": True}

        if tier.get("userDefinedCloudaicompanionProject") and not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable required.")

        if load_data.get("currentTier"):
            with _state_lock:
                _onboarding_complete = True
            return

        # Run onboarding
        onboard_payload = {
            "tierId": tier.get("id"),
            "cloudaicompanionProject": project_id,
            "metadata": get_client_metadata(project_id),
        }

        for _ in range(ONBOARD_MAX_RETRIES):
            onboard_resp = requests.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                json=onboard_payload,
                headers=headers,
            )
            onboard_resp.raise_for_status()
            lro_data = onboard_resp.json()

            if lro_data.get("done"):
                with _state_lock:
                    _onboarding_complete = True
                return

            time.sleep(ONBOARD_POLL_INTERVAL)

        raise Exception("Onboarding timed out")

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        raise Exception(f"Onboarding failed: {error_text}")


def discover_project_id_for_credential(creds: Credentials) -> Optional[str]:
    """
    Discover project ID for a specific credential via API.

    Unlike get_user_project_id(), this function does NOT use global cache
    and always queries the API. Used for multi-credential mode where each
    credential has its own project ID.

    Args:
        creds: Valid credentials

    Returns:
        Project ID or None if discovery fails
    """
    # Refresh credentials if needed
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")
            return None

    if not creds.token:
        logger.error("No valid access token for project ID discovery")
        return None

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    try:
        logger.info("Discovering project ID via API for credential...")
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json={"metadata": get_client_metadata()},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        discovered = data.get("cloudaicompanionProject")

        # If no project ID, try auto-onboarding to free tier
        if not discovered:
            logger.info("No project ID in response, attempting auto-onboard...")
            discovered = _auto_onboard_to_free_tier(creds, headers)

        if discovered:
            logger.info(f"Discovered project ID: {discovered}")

        return discovered

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        logger.error(f"Failed to discover project ID: {error_text}")
        return None
    except Exception as e:
        logger.error(f"Error discovering project ID: {e}")
        return None


def _auto_onboard_to_free_tier(creds: Credentials, headers: dict) -> Optional[str]:
    """
    Automatically onboard user to free tier and return project ID.

    This is called when loadCodeAssist doesn't return a cloudaicompanionProject,
    which happens for accounts that haven't selected a tier yet.

    Args:
        creds: Valid credentials
        headers: HTTP headers for API calls

    Returns:
        Project ID if onboarding succeeds, None otherwise
    """
    logger.info("Auto-onboarding to free tier...")

    try:
        # Start onboarding to free-tier
        onboard_resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
            json={"tierId": "free-tier", "metadata": get_client_metadata()},
            headers=headers,
        )
        onboard_resp.raise_for_status()
        logger.debug(f"Onboarding initiated: {onboard_resp.json()}")

        # Poll loadCodeAssist until we get project ID
        for attempt in range(ONBOARD_MAX_RETRIES):
            time.sleep(ONBOARD_POLL_INTERVAL)

            resp = requests.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                json={"metadata": get_client_metadata()},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            project_id = data.get("cloudaicompanionProject")
            if project_id:
                logger.info(f"Auto-onboarding successful, project ID: {project_id}")
                return project_id

            # Check if we have currentTier but no project ID yet
            if data.get("currentTier"):
                logger.debug(f"Waiting for project ID (attempt {attempt + 1})...")
            else:
                logger.debug(
                    f"Waiting for onboarding to complete (attempt {attempt + 1})..."
                )

        logger.warning("Auto-onboarding timed out waiting for project ID")
        return None

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        logger.error(f"Auto-onboarding failed: {error_text}")
        return None
    except Exception as e:
        logger.error(f"Auto-onboarding error: {e}")
        return None


def get_user_project_id(creds: Credentials) -> str:
    """
    Get the user's Google Cloud project ID.

    If the account hasn't been onboarded to Gemini Code Assist yet,
    this will automatically onboard to the free tier.

    Args:
        creds: Valid credentials

    Returns:
        Project ID string

    Raises:
        Exception: If project ID cannot be determined
    """
    global _user_project_id

    # Priority 1: Environment variable
    env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project_id:
        logger.info(f"Using project ID from env: {env_project_id}")
        with _state_lock:
            _user_project_id = env_project_id
        from .credentials import save_credentials

        save_credentials(creds, env_project_id)
        return env_project_id

    # Priority 2: Cached value
    with _state_lock:
        cached_project_id = _user_project_id
    if cached_project_id:
        logger.info(f"Using cached project ID: {cached_project_id}")
        return cached_project_id

    # Priority 3: Credential file
    if os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                creds_data = json.load(f)
                cached = creds_data.get("project_id")
                if cached:
                    logger.info(f"Using project ID from file: {cached}")
                    with _state_lock:
                        _user_project_id = cached
                    return cached
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read project_id from file: {e}")

    # Priority 4: API discovery
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GoogleAuthRequest())
            from .credentials import save_credentials

            save_credentials(creds)
        except Exception as e:
            logger.error(f"Failed to refresh credentials: {e}")

    if not creds.token:
        raise Exception("No valid access token for project ID discovery")

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    try:
        logger.info("Discovering project ID via API...")
        resp = requests.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            json={"metadata": get_client_metadata()},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        discovered = data.get("cloudaicompanionProject")

        # If no project ID, try auto-onboarding to free tier
        if not discovered:
            logger.info("No project ID in response, attempting auto-onboard...")
            discovered = _auto_onboard_to_free_tier(creds, headers)

        if not discovered:
            raise ValueError(
                "Could not get project ID. The account may need manual onboarding. "
                "Try visiting https://codeassist.google.com/ to set up your account."
            )

        logger.info(f"Discovered project ID: {discovered}")
        with _state_lock:
            _user_project_id = discovered
        from .credentials import save_credentials

        save_credentials(creds, discovered)
        return discovered

    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if hasattr(e, "response") else str(e)
        raise Exception(f"Failed to discover project ID: {error_text}")
