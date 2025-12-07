"""
Helper functions for working with Gemini models.
"""

from typing import Optional, Set

from .gemini import SUPPORTED_MODELS


# Build set of valid model names for fast lookup
_VALID_MODEL_NAMES: Set[str] = set()


def _get_valid_model_names() -> Set[str]:
    """
    Get the set of valid model names (lazy initialization).

    Returns:
        Set of valid model name strings
    """
    global _VALID_MODEL_NAMES

    if not _VALID_MODEL_NAMES:
        for model in SUPPORTED_MODELS:
            full_name = model["name"]  # e.g., "models/gemini-2.5-flash"
            # Add both full name and short name
            _VALID_MODEL_NAMES.add(full_name)
            if full_name.startswith("models/"):
                short_name = full_name[7:]  # Remove "models/" prefix
                _VALID_MODEL_NAMES.add(short_name)

    return _VALID_MODEL_NAMES


def is_valid_model(model_name: str) -> bool:
    """
    Check if a model name is valid/supported.

    Args:
        model_name: Model name to validate

    Returns:
        True if model is supported
    """
    valid_names = _get_valid_model_names()
    return model_name in valid_names


def validate_model(model_name: str) -> Optional[str]:
    """
    Validate model name and return error message if invalid.

    Args:
        model_name: Model name to validate

    Returns:
        Error message if invalid, None if valid
    """
    if not model_name:
        return "Model name is required"

    if not is_valid_model(model_name):
        valid_models = sorted(
            [m["name"].replace("models/", "") for m in SUPPORTED_MODELS]
        )
        return (
            f"Model '{model_name}' is not supported. "
            f"Available models: {', '.join(valid_models[:5])}..."
        )

    return None


def get_base_model_name(model_name: str) -> str:
    """Get the base model name."""
    return model_name


def should_include_thoughts(model_name: str) -> bool:
    """Check if thoughts should be included in the response."""
    # Always include thoughts for thinking-capable models
    return True
