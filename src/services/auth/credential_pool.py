"""
Credential Pool module for managing multiple Google OAuth credentials.
Provides round-robin selection and automatic fallback on failure.
"""

import logging
import threading
import time
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .credentials import CredentialEntry

from ...config import CREDENTIAL_RECOVERY_TIME

logger = logging.getLogger(__name__)

# --- Global State with Thread Safety ---
_state_lock = threading.Lock()
_credential_pool: Optional["CredentialPool"] = None


class CredentialPool:
    """
    Manages multiple Google OAuth credentials with round-robin selection
    and automatic fallback on failure.
    """

    def __init__(self):
        self._entries: List["CredentialEntry"] = []
        self._current_index: int = 0
        self._lock = threading.Lock()

    def add(self, entry: "CredentialEntry") -> None:
        """Add a credential entry to the pool."""
        with self._lock:
            entry.index = len(self._entries)
            self._entries.append(entry)
            logger.info(f"Added credential #{entry.index + 1} from {entry.source}")

    def size(self) -> int:
        """Return the number of credentials in the pool."""
        return len(self._entries)

    def is_empty(self) -> bool:
        """Check if the pool has no credentials."""
        return len(self._entries) == 0

    def get_next(self) -> Optional[Tuple["CredentialEntry", int]]:
        """
        Get the next available credential using round-robin selection.
        Skips credentials that are temporarily marked as failed.

        Returns:
            Tuple of (CredentialEntry, index) or None if no credentials available
        """
        with self._lock:
            if not self._entries:
                return None

            current_time = time.time()
            attempts = 0
            total = len(self._entries)

            while attempts < total:
                entry = self._entries[self._current_index]
                idx = self._current_index
                self._current_index = (self._current_index + 1) % total

                # Check if credential has recovered from failure
                if entry.failed_at is not None:
                    if current_time - entry.failed_at >= CREDENTIAL_RECOVERY_TIME:
                        entry.failed_at = None
                        logger.info(f"Credential #{idx + 1} recovered, now available")
                    else:
                        attempts += 1
                        continue

                logger.debug(f"Selected credential #{idx + 1} (round-robin)")
                return (entry, idx)

            # All credentials failed, try the least recently failed one
            logger.warning("All credentials marked as failed, trying oldest failure")
            oldest_entry = min(self._entries, key=lambda e: e.failed_at or 0)
            oldest_entry.failed_at = None
            return (oldest_entry, oldest_entry.index)

    def get_fallback(
        self, exclude_index: int
    ) -> Optional[Tuple["CredentialEntry", int]]:
        """
        Get a fallback credential, excluding the specified index.

        Args:
            exclude_index: Index of credential to skip

        Returns:
            Tuple of (CredentialEntry, index) or None if no fallback available
        """
        with self._lock:
            if len(self._entries) <= 1:
                return None

            current_time = time.time()

            for i, entry in enumerate(self._entries):
                if i == exclude_index:
                    continue

                # Check if recovered or never failed
                if entry.failed_at is None:
                    logger.info(f"Using fallback credential #{i + 1}")
                    return (entry, i)
                elif current_time - entry.failed_at >= CREDENTIAL_RECOVERY_TIME:
                    entry.failed_at = None
                    logger.info(f"Using recovered fallback credential #{i + 1}")
                    return (entry, i)

            return None

    def mark_failed(self, index: int) -> None:
        """
        Mark a credential as temporarily failed.

        Args:
            index: Index of the failed credential
        """
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].failed_at = time.time()
                logger.warning(
                    f"Credential #{index + 1} marked as failed, "
                    f"will retry in {CREDENTIAL_RECOVERY_TIME}s"
                )

    def mark_success(self, index: int) -> None:
        """
        Mark a credential as successful (clear failed status).

        Args:
            index: Index of the successful credential
        """
        with self._lock:
            if 0 <= index < len(self._entries):
                if self._entries[index].failed_at is not None:
                    self._entries[index].failed_at = None
                    logger.info(f"Credential #{index + 1} marked as recovered")

    def get_entry(self, index: int) -> Optional["CredentialEntry"]:
        """Get a specific credential entry by index."""
        with self._lock:
            if 0 <= index < len(self._entries):
                return self._entries[index]
            return None

    def update_project_id(self, index: int, project_id: str) -> None:
        """Update the project ID for a credential entry."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].project_id = project_id

    def set_onboarding_complete(self, index: int) -> None:
        """Mark onboarding as complete for a credential entry."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].onboarding_complete = True

    def get_stats(self) -> dict:
        """Get pool statistics for debugging."""
        with self._lock:
            current_time = time.time()
            return {
                "total": len(self._entries),
                "available": sum(
                    1
                    for e in self._entries
                    if e.failed_at is None
                    or current_time - e.failed_at >= CREDENTIAL_RECOVERY_TIME
                ),
                "failed": sum(
                    1
                    for e in self._entries
                    if e.failed_at is not None
                    and current_time - e.failed_at < CREDENTIAL_RECOVERY_TIME
                ),
                "current_index": self._current_index,
            }


def get_credential_pool() -> CredentialPool:
    """
    Get or initialize the credential pool.

    Returns:
        The global CredentialPool instance
    """
    global _credential_pool

    if _credential_pool is None:
        # Import here to avoid circular imports
        from .credentials import initialize_credential_pool

        return initialize_credential_pool()

    return _credential_pool


def set_credential_pool(pool: Optional[CredentialPool]) -> None:
    """
    Set the global credential pool instance.

    Args:
        pool: CredentialPool instance or None to reset
    """
    global _credential_pool
    _credential_pool = pool
