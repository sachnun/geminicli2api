"""
Credentials module for loading, saving, and managing OAuth credentials.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

from ...config import (
    CLIENT_ID,
    CLIENT_SECRET,
    SCOPES,
    CREDENTIAL_FILE,
    TOKEN_URI,
    ISO_DATE_FORMAT,
)

logger = logging.getLogger(__name__)

# --- Global State with Thread Safety ---
_state_lock = threading.Lock()
_credentials: Optional[Credentials] = None
_user_project_id: Optional[str] = None
_onboarding_complete: bool = False
_credentials_from_env: bool = False


@dataclass
class CredentialEntry:
    """Represents a single credential with its metadata."""

    credentials: Credentials
    project_id: Optional[str] = None
    source: str = "unknown"
    index: int = 0
    failed_at: Optional[float] = None
    onboarding_complete: bool = False
    from_env: bool = False


def _parse_expiry(expiry_str: str) -> Optional[str]:
    """
    Parse and normalize expiry string to ISO format.

    Args:
        expiry_str: Expiry timestamp string in various formats

    Returns:
        Normalized expiry string or None if parsing fails
    """
    if not isinstance(expiry_str, str):
        return None

    try:
        if "+00:00" in expiry_str:
            parsed_expiry = datetime.fromisoformat(expiry_str)
        elif expiry_str.endswith("Z"):
            parsed_expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
        else:
            parsed_expiry = datetime.fromisoformat(expiry_str)

        timestamp = parsed_expiry.timestamp()
        return datetime.utcfromtimestamp(timestamp).strftime(ISO_DATE_FORMAT)
    except (ValueError, OSError) as e:
        logger.warning(f"Could not parse expiry format '{expiry_str}': {e}")
        return None


def _normalize_credentials_data(creds_data: dict) -> dict:
    """
    Normalize credential data to standard format.

    Args:
        creds_data: Raw credentials dictionary

    Returns:
        Normalized credentials dictionary
    """
    normalized = creds_data.copy()

    # Handle access_token -> token mapping
    if "access_token" in normalized and "token" not in normalized:
        normalized["token"] = normalized["access_token"]

    # Handle scope -> scopes mapping
    if "scope" in normalized and "scopes" not in normalized:
        normalized["scopes"] = normalized["scope"].split()

    # Handle expiry format
    if "expiry" in normalized:
        parsed_expiry = _parse_expiry(normalized["expiry"])
        if parsed_expiry:
            normalized["expiry"] = parsed_expiry
        else:
            del normalized["expiry"]

    return normalized


def _create_credentials_from_data(
    creds_data: dict, source: str = "unknown"
) -> Optional[Credentials]:
    """
    Create Credentials object from data dictionary.

    Args:
        creds_data: Credentials data dictionary
        source: Source identifier for logging

    Returns:
        Credentials object or None if creation fails
    """
    global _user_project_id

    try:
        normalized = _normalize_credentials_data(creds_data)
        credentials = Credentials.from_authorized_user_info(normalized, SCOPES)

        # Extract project_id if available (thread-safe)
        if "project_id" in creds_data:
            with _state_lock:
                _user_project_id = creds_data["project_id"]
            logger.info(
                f"Extracted project_id from {source}: {creds_data['project_id']}"
            )

        return credentials
    except Exception as e:
        logger.warning(f"Failed to create credentials from {source}: {e}")
        return None


def _create_minimal_credentials(creds_data: dict) -> Optional[Credentials]:
    """
    Create minimal credentials with just refresh token.

    Args:
        creds_data: Raw credentials data with refresh_token

    Returns:
        Credentials object or None if creation fails
    """
    try:
        minimal_data = {
            "client_id": creds_data.get("client_id", CLIENT_ID),
            "client_secret": creds_data.get("client_secret", CLIENT_SECRET),
            "refresh_token": creds_data["refresh_token"],
            "token_uri": TOKEN_URI,
        }
        return Credentials.from_authorized_user_info(minimal_data, SCOPES)
    except Exception as e:
        logger.error(f"Failed to create minimal credentials: {e}")
        return None


def _refresh_credentials(creds: Credentials) -> bool:
    """
    Attempt to refresh credentials.

    Args:
        creds: Credentials object to refresh

    Returns:
        True if refresh succeeded, False otherwise
    """
    if not creds.refresh_token:
        logger.warning("No refresh token available")
        return False

    try:
        creds.refresh(GoogleAuthRequest())
        logger.info("Credentials refreshed successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to refresh credentials: {e}")
        return False


def save_credentials(creds: Credentials, project_id: Optional[str] = None) -> None:
    """
    Save credentials to file.

    Args:
        creds: Credentials object to save
        project_id: Optional project ID to save
    """
    global _credentials_from_env

    with _state_lock:
        from_env = _credentials_from_env

    if from_env:
        # Only update project_id in existing file if needed
        if project_id and os.path.exists(CREDENTIAL_FILE):
            try:
                with open(CREDENTIAL_FILE, "r") as f:
                    existing_data = json.load(f)
                if "project_id" not in existing_data:
                    existing_data["project_id"] = project_id
                    with open(CREDENTIAL_FILE, "w") as f:
                        json.dump(existing_data, f, indent=2)
                    logger.info(f"Added project_id {project_id} to credential file")
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not update project_id in credential file: {e}")
        return

    creds_data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "token_uri": TOKEN_URI,
    }

    if creds.expiry:
        expiry_utc = creds.expiry
        if expiry_utc.tzinfo is None:
            expiry_utc = expiry_utc.replace(tzinfo=timezone.utc)
        creds_data["expiry"] = expiry_utc.isoformat()

    # Preserve or set project_id
    if project_id:
        creds_data["project_id"] = project_id
    elif os.path.exists(CREDENTIAL_FILE):
        try:
            with open(CREDENTIAL_FILE, "r") as f:
                existing_data = json.load(f)
                if "project_id" in existing_data:
                    creds_data["project_id"] = existing_data["project_id"]
        except (IOError, json.JSONDecodeError):
            pass

    with open(CREDENTIAL_FILE, "w") as f:
        json.dump(creds_data, f, indent=2)


def _load_credentials_from_env() -> Optional[Credentials]:
    """Load credentials from GEMINI_CREDENTIALS environment variable."""
    global _credentials_from_env

    env_creds_json = os.getenv("GEMINI_CREDENTIALS")
    if not env_creds_json:
        return None

    try:
        raw_data = json.loads(env_creds_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse GEMINI_CREDENTIALS JSON: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning("No refresh_token in environment credentials")
        return None

    logger.info("Loading credentials from environment variable")

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, "environment")
    if creds:
        with _state_lock:
            _credentials_from_env = True
        if creds.expired:
            _refresh_credentials(creds)
        return creds

    # Try minimal credentials as fallback
    creds = _create_minimal_credentials(raw_data)
    if creds:
        with _state_lock:
            _credentials_from_env = True
        _refresh_credentials(creds)
        return creds

    return None


def _load_credentials_from_file() -> Optional[Credentials]:
    """Load credentials from credential file."""
    global _credentials_from_env

    if not os.path.exists(CREDENTIAL_FILE):
        return None

    try:
        with open(CREDENTIAL_FILE, "r") as f:
            raw_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read credentials file: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning("No refresh_token in credentials file")
        return None

    logger.info("Loading credentials from file")

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, "file")
    if creds:
        with _state_lock:
            _credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        if creds.expired and _refresh_credentials(creds):
            save_credentials(creds)
        return creds

    # Try minimal credentials as fallback
    creds = _create_minimal_credentials(raw_data)
    if creds:
        with _state_lock:
            _credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
        if _refresh_credentials(creds):
            save_credentials(creds)
        return creds

    return None


# --- Multi-Credential Loading Functions ---


def _load_credential_entry_from_json(
    json_str: str, source: str
) -> Optional[CredentialEntry]:
    """
    Create a CredentialEntry from JSON string.

    Args:
        json_str: JSON string containing credentials
        source: Source identifier for logging

    Returns:
        CredentialEntry or None if parsing fails
    """
    try:
        raw_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse credentials JSON from {source}: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning(f"No refresh_token in credentials from {source}")
        return None

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, source)
    if not creds:
        # Try minimal credentials as fallback
        creds = _create_minimal_credentials(raw_data)

    if not creds:
        return None

    if creds.expired:
        _refresh_credentials(creds)

    project_id = raw_data.get("project_id")

    return CredentialEntry(
        credentials=creds,
        project_id=project_id,
        source=source,
        from_env=True,
    )


def _load_credential_entry_from_file(
    file_path: str, source: str
) -> Optional[CredentialEntry]:
    """
    Create a CredentialEntry from a file.

    Args:
        file_path: Path to the credentials file
        source: Source identifier for logging

    Returns:
        CredentialEntry or None if loading fails
    """
    if not os.path.exists(file_path):
        logger.warning(f"Credentials file not found: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            raw_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read credentials file {file_path}: {e}")
        return None

    if not raw_data.get("refresh_token"):
        logger.warning(f"No refresh_token in credentials file {file_path}")
        return None

    # Try normal parsing first
    creds = _create_credentials_from_data(raw_data, source)
    if not creds:
        # Try minimal credentials as fallback
        creds = _create_minimal_credentials(raw_data)

    if not creds:
        return None

    if creds.expired:
        _refresh_credentials(creds)

    project_id = raw_data.get("project_id")

    return CredentialEntry(
        credentials=creds,
        project_id=project_id,
        source=source,
        from_env=False,
    )


def _load_multiple_credentials_from_env() -> List[CredentialEntry]:
    """
    Load multiple credentials from environment variables.
    Supports patterns: GEMINI_CREDENTIALS_1, GEMINI_CREDENTIALS_2, etc.

    Returns:
        List of CredentialEntry objects
    """
    entries: List[CredentialEntry] = []

    # Find all GEMINI_CREDENTIALS_N environment variables
    pattern = re.compile(r"^GEMINI_CREDENTIALS_(\d+)$")
    indexed_vars: List[Tuple[int, str, str]] = []

    for key, value in os.environ.items():
        match = pattern.match(key)
        if match:
            index = int(match.group(1))
            indexed_vars.append((index, key, value))

    # Sort by index to maintain order
    indexed_vars.sort(key=lambda x: x[0])

    for index, key, value in indexed_vars:
        entry = _load_credential_entry_from_json(value, f"env:{key}")
        if entry:
            entries.append(entry)

    if entries:
        logger.info(f"Loaded {len(entries)} credentials from indexed env vars")

    return entries


def _load_multiple_credentials_from_files() -> List[CredentialEntry]:
    """
    Load multiple credentials from files specified in GEMINI_CREDENTIAL_FILES.
    Format: comma-separated list of file paths.

    Returns:
        List of CredentialEntry objects
    """
    entries: List[CredentialEntry] = []

    files_str = os.getenv("GEMINI_CREDENTIAL_FILES", "")
    if not files_str:
        return entries

    file_paths = [f.strip() for f in files_str.split(",") if f.strip()]

    for i, file_path in enumerate(file_paths):
        # Handle relative paths
        if not os.path.isabs(file_path):
            from ...config import SCRIPT_DIR

            file_path = os.path.join(SCRIPT_DIR, file_path)

        entry = _load_credential_entry_from_file(file_path, f"file:{file_path}")
        if entry:
            entries.append(entry)

    if entries:
        logger.info(f"Loaded {len(entries)} credentials from files")

    return entries


def initialize_credential_pool():
    """
    Initialize the credential pool with all available credentials.
    Loads from multiple sources in order of priority.

    Returns:
        Initialized CredentialPool
    """
    from .credential_pool import (
        CredentialPool,
        set_credential_pool,
        get_credential_pool,
    )
    from .credential_pool import _credential_pool

    # Check if already initialized
    if _credential_pool is not None and not _credential_pool.is_empty():
        return _credential_pool

    pool = CredentialPool()

    # Priority 1: Multiple indexed environment variables (GEMINI_CREDENTIALS_N)
    env_entries = _load_multiple_credentials_from_env()
    for entry in env_entries:
        pool.add(entry)

    # Priority 2: Multiple files (GEMINI_CREDENTIAL_FILES)
    file_entries = _load_multiple_credentials_from_files()
    for entry in file_entries:
        pool.add(entry)

    # Priority 3: Single GEMINI_CREDENTIALS env var (backward compatibility)
    if pool.is_empty():
        single_creds_json = os.getenv("GEMINI_CREDENTIALS")
        if single_creds_json:
            entry = _load_credential_entry_from_json(
                single_creds_json, "env:GEMINI_CREDENTIALS"
            )
            if entry:
                pool.add(entry)

    # Priority 4: Default credential file (backward compatibility)
    if pool.is_empty():
        entry = _load_credential_entry_from_file(
            CREDENTIAL_FILE, f"file:{CREDENTIAL_FILE}"
        )
        if entry:
            pool.add(entry)

    stats = pool.get_stats()
    logger.info(
        f"Credential pool initialized: {stats['total']} total, "
        f"{stats['available']} available"
    )

    set_credential_pool(pool)
    return pool


def get_credentials(allow_oauth_flow: bool = True) -> Optional[Credentials]:
    """
    Load or obtain OAuth credentials (backward compatible single-credential mode).

    Args:
        allow_oauth_flow: Whether to run interactive OAuth if needed

    Returns:
        Credentials object or None if unavailable
    """
    global _credentials

    with _state_lock:
        if _credentials and _credentials.token:
            return _credentials

    # Try environment variable first
    creds = _load_credentials_from_env()
    if creds:
        with _state_lock:
            _credentials = creds
        return creds

    # Try credentials file
    creds = _load_credentials_from_file()
    if creds:
        with _state_lock:
            _credentials = creds
        return creds

    # Run OAuth flow if allowed
    if allow_oauth_flow:
        from .oauth import _run_oauth_flow

        creds = _run_oauth_flow()
        if creds:
            with _state_lock:
                _credentials = creds
        return creds

    logger.info("OAuth flow not allowed, returning None")
    return None


def get_next_credential(
    allow_oauth_flow: bool = True,
) -> Optional[Tuple[Credentials, int]]:
    """
    Get the next available credential from the pool using round-robin selection.

    Args:
        allow_oauth_flow: Whether to run interactive OAuth if pool is empty

    Returns:
        Tuple of (Credentials, credential_index) or None if unavailable
    """
    from .credential_pool import get_credential_pool

    pool = get_credential_pool()

    if pool.is_empty():
        # Fallback to single credential mode / OAuth flow
        creds = get_credentials(allow_oauth_flow)
        if creds:
            return (creds, -1)  # -1 indicates single credential mode
        return None

    result = pool.get_next()
    if result:
        entry, idx = result
        # Refresh if needed
        if entry.credentials.expired and entry.credentials.refresh_token:
            try:
                entry.credentials.refresh(GoogleAuthRequest())
                logger.debug(f"Refreshed credential #{idx + 1}")
            except Exception as e:
                logger.warning(f"Failed to refresh credential #{idx + 1}: {e}")
                pool.mark_failed(idx)
                # Try to get fallback
                fallback = pool.get_fallback(idx)
                if fallback:
                    return (fallback[0].credentials, fallback[1])
                return None

        return (entry.credentials, idx)

    return None


def get_fallback_credential(exclude_index: int) -> Optional[Tuple[Credentials, int]]:
    """
    Get a fallback credential when the current one fails.

    Args:
        exclude_index: Index of credential to skip

    Returns:
        Tuple of (Credentials, credential_index) or None if no fallback available
    """
    if exclude_index == -1:
        # Single credential mode, no fallback
        return None

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    result = pool.get_fallback(exclude_index)

    if result:
        entry, idx = result
        # Refresh if needed
        if entry.credentials.expired and entry.credentials.refresh_token:
            try:
                entry.credentials.refresh(GoogleAuthRequest())
            except Exception as e:
                logger.warning(f"Failed to refresh fallback credential #{idx + 1}: {e}")
                pool.mark_failed(idx)
                return None

        return (entry.credentials, idx)

    return None


def mark_credential_failed(index: int) -> None:
    """
    Mark a credential as failed.

    Args:
        index: Index of the failed credential (-1 for single credential mode)
    """
    if index == -1:
        return  # Single credential mode, nothing to mark

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    pool.mark_failed(index)


def mark_credential_success(index: int) -> None:
    """
    Mark a credential as successful (clear failed status).

    Args:
        index: Index of the successful credential
    """
    if index == -1:
        return  # Single credential mode

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    pool.mark_success(index)


def get_credential_project_id(index: int) -> Optional[str]:
    """
    Get the project ID associated with a credential.

    Args:
        index: Credential index

    Returns:
        Project ID or None
    """
    if index == -1:
        return None  # Use global project ID discovery

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    entry = pool.get_entry(index)
    return entry.project_id if entry else None


def set_credential_project_id(index: int, project_id: str) -> None:
    """
    Set the project ID for a credential.

    Args:
        index: Credential index
        project_id: Project ID to set
    """
    if index == -1:
        return  # Single credential mode uses global

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    pool.update_project_id(index, project_id)


def is_credential_onboarded(index: int) -> bool:
    """
    Check if a credential has completed onboarding.

    Args:
        index: Credential index

    Returns:
        True if onboarded
    """
    if index == -1:
        with _state_lock:
            return _onboarding_complete

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    entry = pool.get_entry(index)
    return entry.onboarding_complete if entry else False


def set_credential_onboarded(index: int) -> None:
    """
    Mark a credential as having completed onboarding.

    Args:
        index: Credential index
    """
    global _onboarding_complete

    if index == -1:
        with _state_lock:
            _onboarding_complete = True
        return

    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    pool.set_onboarding_complete(index)


def get_pool_stats() -> dict:
    """
    Get credential pool statistics.

    Returns:
        Dictionary with pool statistics
    """
    from .credential_pool import get_credential_pool

    pool = get_credential_pool()
    return pool.get_stats()
