"""
Authentication module for Geminicli2api.
Handles OAuth2 authentication with Google APIs.
Supports multiple credentials with round-robin selection and automatic fallback.

This module is split into focused submodules:
- credential_pool: CredentialPool class and pool management
- credentials: Credential loading/saving logic
- oauth: OAuth flow and token refresh
- middleware: FastAPI authentication dependency
- onboarding: User onboarding to Code Assist
"""

# Re-export all public APIs for backward compatibility
from .credential_pool import (
    CredentialPool,
    get_credential_pool,
)

from .credentials import (
    CredentialEntry,
    get_credentials,
    save_credentials,
    get_next_credential,
    get_fallback_credential,
    mark_credential_failed,
    mark_credential_success,
    get_credential_project_id,
    set_credential_project_id,
    is_credential_onboarded,
    set_credential_onboarded,
    get_pool_stats,
    initialize_credential_pool,
    _load_credential_entry_from_json,
    _load_credential_entry_from_file,
    _load_multiple_credentials_from_env,
    _load_multiple_credentials_from_files,
)

from .oauth import (
    refresh_access_token,
    get_oauth_token,
)

from .middleware import (
    authenticate_user,
)

from .onboarding import (
    onboard_user,
    get_user_project_id,
    discover_project_id_for_credential,
)

__all__ = [
    # CredentialPool
    "CredentialPool",
    "get_credential_pool",
    # Credentials
    "CredentialEntry",
    "get_credentials",
    "save_credentials",
    "get_next_credential",
    "get_fallback_credential",
    "mark_credential_failed",
    "mark_credential_success",
    "get_credential_project_id",
    "set_credential_project_id",
    "is_credential_onboarded",
    "set_credential_onboarded",
    "get_pool_stats",
    "initialize_credential_pool",
    "_load_credential_entry_from_json",
    "_load_credential_entry_from_file",
    "_load_multiple_credentials_from_env",
    "_load_multiple_credentials_from_files",
    # OAuth
    "refresh_access_token",
    "get_oauth_token",
    # Middleware
    "authenticate_user",
    # Onboarding
    "onboard_user",
    "get_user_project_id",
    "discover_project_id_for_credential",
]
