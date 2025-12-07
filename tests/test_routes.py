"""Tests for route handlers (OpenAI, Anthropic, Responses)."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.routes.openai import router as openai_router
from src.routes.anthropic import router as anthropic_router, authenticate_anthropic
from src.routes.responses import router as responses_router


# --- Test setup ---


def create_test_app_openai():
    """Create a FastAPI app with OpenAI router for testing."""
    app = FastAPI()
    app.include_router(openai_router)
    return app


def create_test_app_anthropic():
    """Create a FastAPI app with Anthropic router for testing."""
    app = FastAPI()
    app.include_router(anthropic_router)
    return app


def create_test_app_responses():
    """Create a FastAPI app with Responses router for testing."""
    app = FastAPI()
    app.include_router(responses_router)
    return app


# --- OpenAI Route Tests ---


class TestOpenAIListModels:
    """Tests for /v1/models endpoint."""

    def test_list_models_success(self):
        """List models should return supported models."""
        app = create_test_app_openai()
        client = TestClient(app)

        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        # Check model structure
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert model["owned_by"] == "google"


class TestOpenAIChatCompletions:
    """Tests for /v1/chat/completions endpoint."""

    @patch("src.routes.openai.authenticate_user")
    def test_invalid_model(self, mock_auth):
        """Invalid model should return 400 error."""
        mock_auth.return_value = "test_user"
        app = create_test_app_openai()
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid-model-name",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    @patch("src.services.gemini_client.send_gemini_request")
    @patch("src.routes.openai.authenticate_user")
    def test_chat_completion_non_streaming(self, mock_auth, mock_send):
        """Non-streaming chat completion should work."""
        mock_auth.return_value = "test_user"

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello! How can I help?"}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ).encode()
        mock_send.return_value = mock_response

        app = create_test_app_openai()
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-pro",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "gemini-2.5-pro"
        assert len(data["choices"]) == 1
        # Verify response has message content structure
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]


# --- Anthropic Route Tests ---


class TestAnthropicAuthentication:
    """Tests for Anthropic authentication."""

    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_password")
    def test_valid_x_api_key(self):
        """Valid x-api-key should authenticate."""
        result = authenticate_anthropic(authorization=None, x_api_key="test_password")
        assert result == "anthropic_user"

    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_password")
    def test_valid_sk_prefixed_key(self):
        """sk- prefixed key should authenticate."""
        result = authenticate_anthropic(
            authorization=None, x_api_key="sk-test_password"
        )
        assert result == "anthropic_user"

    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_password")
    def test_valid_bearer_token(self):
        """Bearer token should authenticate."""
        result = authenticate_anthropic(
            authorization="Bearer test_password", x_api_key=None
        )
        assert result == "anthropic_user"

    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_password")
    def test_invalid_key_raises_401(self):
        """Invalid key should raise 401."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            authenticate_anthropic(authorization=None, x_api_key="wrong_password")
        assert exc_info.value.status_code == 401


class TestAnthropicMessages:
    """Tests for /v1/messages endpoint."""

    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_key")
    def test_invalid_model(self):
        """Invalid model should return 400 error."""
        app = create_test_app_anthropic()
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test_key"},
            json={
                "model": "invalid-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"

    @patch("src.routes.anthropic.send_gemini_request")
    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_key")
    def test_messages_non_streaming(self, mock_send):
        """Non-streaming messages should work."""
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello! How can I help?"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
            }
        ).encode()
        mock_send.return_value = mock_response

        app = create_test_app_anthropic()
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            headers={"x-api-key": "test_key"},
            json={
                "model": "gemini-2.5-pro",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) == 1
        assert data["content"][0]["text"] == "Hello! How can I help?"

    @patch("src.routes.anthropic.GEMINI_AUTH_PASSWORD", "test_key")
    def test_missing_auth_returns_401(self):
        """Missing authentication should return 401."""
        app = create_test_app_anthropic()
        client = TestClient(app)

        response = client.post(
            "/v1/messages",
            json={
                "model": "gemini-2.5-pro",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert response.status_code == 401


# --- Responses Route Tests ---


class TestResponsesCreate:
    """Tests for /v1/responses endpoint."""

    @patch("src.routes.responses.authenticate_user")
    def test_invalid_model(self, mock_auth):
        """Invalid model should return 400 error."""
        mock_auth.return_value = "test_user"
        app = create_test_app_responses()
        client = TestClient(app)

        response = client.post(
            "/v1/responses",
            json={"model": "invalid-model", "input": "Hello"},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    @patch("src.services.gemini_client.send_gemini_request")
    @patch("src.routes.responses.authenticate_user")
    def test_string_input(self, mock_auth, mock_send):
        """String input should work."""
        mock_auth.return_value = "test_user"

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello! How can I help?"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                },
            }
        ).encode()
        mock_send.return_value = mock_response

        app = create_test_app_responses()
        client = TestClient(app)

        response = client.post(
            "/v1/responses",
            json={"model": "gemini-2.5-pro", "input": "Hello", "stream": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"
        assert data["model"] == "gemini-2.5-pro"
        assert data["status"] == "completed"
        # Verify output_text is present
        assert "output_text" in data

    @patch("src.services.gemini_client.send_gemini_request")
    @patch("src.routes.responses.authenticate_user")
    def test_list_input(self, mock_auth, mock_send):
        """List input should work."""
        mock_auth.return_value = "test_user"

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Response text"}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ).encode()
        mock_send.return_value = mock_response

        app = create_test_app_responses()
        client = TestClient(app)

        response = client.post(
            "/v1/responses",
            json={
                "model": "gemini-2.5-pro",
                "input": [{"type": "message", "role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"

    @patch("src.services.gemini_client.send_gemini_request")
    @patch("src.routes.responses.authenticate_user")
    def test_with_instructions(self, mock_auth, mock_send):
        """Request with instructions should work."""
        mock_auth.return_value = "test_user"

        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.body = json.dumps(
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Helpful response"}],
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ).encode()
        mock_send.return_value = mock_response

        app = create_test_app_responses()
        client = TestClient(app)

        response = client.post(
            "/v1/responses",
            json={
                "model": "gemini-2.5-pro",
                "input": "Hello",
                "instructions": "You are a helpful assistant.",
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "response"
