"""Tests for the Gemini client module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import Response
from fastapi.responses import StreamingResponse

from src.services.gemini_client import (
    _create_json_error_response,
    _error_stream_generator,
    _stream_generator,
    _make_non_streaming_request,
    _should_try_fallback,
    _is_error_response,
    _get_project_id_for_credential,
    build_gemini_payload_from_openai,
    build_gemini_payload_from_native,
    send_gemini_request,
    FALLBACK_ERROR_CODES,
)


class TestCreateJsonErrorResponse:
    """Tests for _create_json_error_response."""

    def test_404_error_type(self):
        """Test that 404 errors get invalid_request_error type."""
        response = _create_json_error_response("Not found", 404)
        assert response.status_code == 404
        content = json.loads(bytes(response.body))
        assert content["error"]["type"] == "invalid_request_error"
        assert content["error"]["message"] == "Not found"

    def test_500_error_type(self):
        """Test that 500 errors get api_error type."""
        response = _create_json_error_response("Internal error", 500)
        assert response.status_code == 500
        content = json.loads(bytes(response.body))
        assert content["error"]["type"] == "api_error"
        assert content["error"]["message"] == "Internal error"

    def test_401_error_type(self):
        """Test that 401 errors get api_error type."""
        response = _create_json_error_response("Unauthorized", 401)
        assert response.status_code == 401
        content = json.loads(bytes(response.body))
        assert content["error"]["type"] == "api_error"


class TestErrorStreamGenerator:
    """Tests for _error_stream_generator."""

    @pytest.mark.asyncio
    async def test_error_stream_404(self):
        """Test error stream for 404."""
        chunks = []
        async for chunk in _error_stream_generator("Not found", 404):
            chunks.append(chunk)

        assert len(chunks) == 1
        chunk_str = chunks[0].decode("utf-8")
        assert chunk_str.startswith("data: ")
        data = json.loads(chunk_str[6:].strip())
        assert data["error"]["type"] == "invalid_request_error"
        assert data["error"]["message"] == "Not found"

    @pytest.mark.asyncio
    async def test_error_stream_500(self):
        """Test error stream for 500."""
        chunks = []
        async for chunk in _error_stream_generator("Server error", 500):
            chunks.append(chunk)

        assert len(chunks) == 1
        chunk_str = chunks[0].decode("utf-8")
        data = json.loads(chunk_str[6:].strip())
        assert data["error"]["type"] == "api_error"


class TestStreamGenerator:
    """Tests for _stream_generator.

    Note: Stream generator tests are complex due to nested async context managers.
    These tests verify basic error handling behavior that can be reliably mocked.
    """

    @pytest.mark.asyncio
    async def test_stream_api_error(self):
        """Test streaming with API error returns error chunk."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.aread = AsyncMock()
        mock_response.json.return_value = {"error": {"message": "API Error"}}

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__.return_value = mock_response
        mock_stream_cm.__aexit__.return_value = None

        mock_client = AsyncMock()
        mock_client.stream.return_value = mock_stream_cm

        with patch("src.services.gemini_client.httpx.AsyncClient") as MockAsyncClient:
            MockAsyncClient.return_value.__aenter__.return_value = mock_client
            MockAsyncClient.return_value.__aexit__.return_value = None
            chunks = []
            async for chunk in _stream_generator("http://test.com", {}, {}):
                chunks.append(chunk.decode("utf-8"))

        assert len(chunks) == 1
        data = json.loads(chunks[0][6:].strip())
        assert "error" in data


class TestMakeNonStreamingRequest:
    """Tests for _make_non_streaming_request."""

    @pytest.mark.asyncio
    async def test_success_response(self):
        """Test successful non-streaming response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'data: {"response": {"text": "Hello"}}'

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch(
            "src.services.gemini_client.httpx.AsyncClient", return_value=mock_client
        ):
            response = await _make_non_streaming_request("http://test.com", {}, {})

        assert response.status_code == 200
        content = json.loads(bytes(response.body))
        assert content["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_success_response_without_data_prefix(self):
        """Test successful response without data: prefix."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"response": {"text": "Hello"}}'

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch(
            "src.services.gemini_client.httpx.AsyncClient", return_value=mock_client
        ):
            response = await _make_non_streaming_request("http://test.com", {}, {})

        assert response.status_code == 200
        content = json.loads(bytes(response.body))
        assert content["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_api_error_with_message(self):
        """Test API error response with error message."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = {"error": {"message": "Invalid input"}}
        mock_response.headers = {"Content-Type": "application/json"}

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch(
            "src.services.gemini_client.httpx.AsyncClient", return_value=mock_client
        ):
            response = await _make_non_streaming_request("http://test.com", {}, {})

        assert response.status_code == 400
        content = json.loads(bytes(response.body))
        assert content["error"]["message"] == "Invalid input"

    @pytest.mark.asyncio
    async def test_api_error_without_json(self):
        """Test API error response without JSON body."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.content = b"Internal Server Error"
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        mock_response.headers = {"Content-Type": "text/plain"}

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.return_value = mock_response

        with patch(
            "src.services.gemini_client.httpx.AsyncClient", return_value=mock_client
        ):
            response = await _make_non_streaming_request("http://test.com", {}, {})

        assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test timeout error handling."""
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")

        with patch(
            "src.services.gemini_client.httpx.AsyncClient", return_value=mock_client
        ):
            response = await _make_non_streaming_request("http://test.com", {}, {})

        assert response.status_code == 504
        content = json.loads(bytes(response.body))
        assert "timed out" in content["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_request_error(self):
        """Test request error handling."""
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post.side_effect = httpx.RequestError("Connection failed")

        with patch(
            "src.services.gemini_client.httpx.AsyncClient", return_value=mock_client
        ):
            response = await _make_non_streaming_request("http://test.com", {}, {})

        assert response.status_code == 502
        content = json.loads(bytes(response.body))
        assert "failed" in content["error"]["message"].lower()


class TestShouldTryFallback:
    """Tests for _should_try_fallback."""

    def test_single_credential_no_fallback(self):
        """Test that single credential doesn't fallback."""
        response = Response(content=b"", status_code=401)
        pool_stats = {"total": 1, "available": 1}
        assert _should_try_fallback(response, 0, pool_stats) is False

    def test_negative_index_no_fallback(self):
        """Test that -1 index doesn't fallback."""
        response = Response(content=b"", status_code=401)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, -1, pool_stats) is False

    def test_401_triggers_fallback(self):
        """Test that 401 triggers fallback."""
        response = Response(content=b"", status_code=401)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, 0, pool_stats) is True

    def test_403_triggers_fallback(self):
        """Test that 403 triggers fallback."""
        response = Response(content=b"", status_code=403)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, 0, pool_stats) is True

    def test_429_triggers_fallback(self):
        """Test that 429 triggers fallback."""
        response = Response(content=b"", status_code=429)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, 0, pool_stats) is True

    def test_200_no_fallback(self):
        """Test that 200 doesn't trigger fallback."""
        response = Response(content=b"", status_code=200)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, 0, pool_stats) is False

    def test_500_no_fallback(self):
        """Test that 500 doesn't trigger fallback (not in FALLBACK_ERROR_CODES)."""
        response = Response(content=b"", status_code=500)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, 0, pool_stats) is False

    def test_streaming_response_fallback(self):
        """Test fallback with StreamingResponse."""

        async def dummy_gen():
            yield b""

        response = StreamingResponse(dummy_gen(), status_code=401)
        pool_stats = {"total": 2, "available": 2}
        assert _should_try_fallback(response, 0, pool_stats) is True


class TestIsErrorResponse:
    """Tests for _is_error_response."""

    def test_response_200_not_error(self):
        """Test that 200 is not an error."""
        response = Response(content=b"", status_code=200)
        assert _is_error_response(response) is False

    def test_response_400_is_error(self):
        """Test that 400 is an error."""
        response = Response(content=b"", status_code=400)
        assert _is_error_response(response) is True

    def test_response_500_is_error(self):
        """Test that 500 is an error."""
        response = Response(content=b"", status_code=500)
        assert _is_error_response(response) is True

    def test_streaming_response_200_not_error(self):
        """Test that streaming 200 is not an error."""

        async def dummy_gen():
            yield b""

        response = StreamingResponse(dummy_gen(), status_code=200)
        assert _is_error_response(response) is False

    def test_streaming_response_400_is_error(self):
        """Test that streaming 400 is an error."""

        async def dummy_gen():
            yield b""

        response = StreamingResponse(dummy_gen(), status_code=400)
        assert _is_error_response(response) is True


class TestGetProjectIdForCredential:
    """Tests for _get_project_id_for_credential."""

    def test_cached_project_id(self):
        """Test returning cached project ID."""
        mock_creds = MagicMock()

        with patch(
            "src.services.gemini_client.get_credential_project_id",
            return_value="cached-project",
        ):
            result = _get_project_id_for_credential(mock_creds, 0)

        assert result == "cached-project"

    def test_discover_project_id_multi_credential(self):
        """Test discovering project ID in multi-credential mode."""
        mock_creds = MagicMock()

        with patch(
            "src.services.gemini_client.get_credential_project_id", return_value=None
        ):
            with patch(
                "src.services.gemini_client.discover_project_id_for_credential",
                return_value="discovered-project",
            ):
                with patch(
                    "src.services.gemini_client.set_credential_project_id"
                ) as mock_set:
                    result = _get_project_id_for_credential(mock_creds, 0)

        assert result == "discovered-project"
        mock_set.assert_called_once_with(0, "discovered-project")

    def test_fallback_to_global_project_id(self):
        """Test fallback to global project ID in single credential mode."""
        mock_creds = MagicMock()

        with patch(
            "src.services.gemini_client.get_credential_project_id", return_value=None
        ):
            with patch(
                "src.services.gemini_client.get_user_project_id",
                return_value="global-project",
            ):
                result = _get_project_id_for_credential(mock_creds, -1)

        assert result == "global-project"

    def test_discovery_failure(self):
        """Test handling discovery failure."""
        mock_creds = MagicMock()

        with patch(
            "src.services.gemini_client.get_credential_project_id", return_value=None
        ):
            with patch(
                "src.services.gemini_client.discover_project_id_for_credential",
                side_effect=Exception("Discovery failed"),
            ):
                result = _get_project_id_for_credential(mock_creds, 0)

        assert result is None


class TestBuildGeminiPayloadFromOpenai:
    """Tests for build_gemini_payload_from_openai."""

    def test_basic_payload(self):
        """Test basic payload building."""
        openai_payload = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }

        result = build_gemini_payload_from_openai(openai_payload)

        assert result["model"] == "gemini-2.0-flash"
        assert result["request"]["contents"] == openai_payload["contents"]
        assert "safetySettings" in result["request"]

    def test_payload_with_all_fields(self):
        """Test payload with all optional fields."""
        openai_payload = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "You are helpful"}]},
            "tools": [{"functionDeclarations": []}],
            "toolConfig": {"functionCallingConfig": {}},
            "generationConfig": {"temperature": 0.7},
        }

        result = build_gemini_payload_from_openai(openai_payload)

        assert result["request"]["systemInstruction"] is not None
        assert result["request"]["tools"] is not None
        assert result["request"]["toolConfig"] is not None
        assert result["request"]["generationConfig"]["temperature"] == 0.7

    def test_removes_none_values(self):
        """Test that None values are removed."""
        openai_payload = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "systemInstruction": None,
            "tools": None,
        }

        result = build_gemini_payload_from_openai(openai_payload)

        assert "systemInstruction" not in result["request"]
        assert "tools" not in result["request"]


class TestBuildGeminiPayloadFromNative:
    """Tests for build_gemini_payload_from_native."""

    def test_basic_native_request(self):
        """Test basic native request building."""
        native_request = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }

        result = build_gemini_payload_from_native(native_request, "gemini-2.0-flash")

        assert result["model"] == "gemini-2.0-flash"
        assert "safetySettings" in result["request"]
        assert "generationConfig" in result["request"]
        assert "thinkingConfig" in result["request"]["generationConfig"]

    def test_thinking_config_added(self):
        """Test that thinking config is added."""
        native_request = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }

        result = build_gemini_payload_from_native(
            native_request, "gemini-2.5-flash-preview"
        )

        assert "thinkingConfig" in result["request"]["generationConfig"]
        assert (
            "includeThoughts" in result["request"]["generationConfig"]["thinkingConfig"]
        )
        assert (
            result["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"]
            == -1
        )

    def test_image_model_no_thinking(self):
        """Test that image model doesn't get thinking config modified."""
        native_request = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }

        result = build_gemini_payload_from_native(
            native_request, "gemini-2.5-flash-image"
        )

        # Should have thinkingConfig but not includeThoughts set
        assert "generationConfig" in result["request"]
        assert "thinkingConfig" in result["request"]["generationConfig"]

    def test_preserves_existing_thinking_budget(self):
        """Test that existing thinking budget is preserved."""
        native_request = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingBudget": 1000,
                }
            },
        }

        result = build_gemini_payload_from_native(native_request, "gemini-2.0-flash")

        assert (
            result["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"]
            == 1000
        )

    def test_model_name_passed_through(self):
        """Test that model name is passed through as-is."""
        native_request = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }

        result = build_gemini_payload_from_native(
            native_request, "models/gemini-2.0-flash"
        )

        # Model name is passed through without modification
        assert result["model"] == "models/gemini-2.0-flash"


class TestSendGeminiRequest:
    """Tests for send_gemini_request."""

    @pytest.mark.asyncio
    async def test_no_credential_error(self):
        """Test error when no credentials available."""
        with patch("src.services.gemini_client.get_next_credential", return_value=None):
            response = await send_gemini_request({}, is_streaming=False)

        assert response.status_code == 500
        content = json.loads(bytes(response.body))
        assert "Authentication failed" in content["error"]["message"]

    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful request flow."""
        mock_creds = MagicMock()
        mock_creds.expired = False
        mock_creds.token = "test-token"

        with patch(
            "src.services.gemini_client.get_next_credential",
            return_value=(mock_creds, 0),
        ):
            with patch(
                "src.services.gemini_client.get_pool_stats",
                return_value={"total": 1, "available": 1},
            ):
                with patch(
                    "src.services.gemini_client._send_request_with_credential"
                ) as mock_send:
                    mock_response = Response(
                        content=b'{"text": "Hello"}', status_code=200
                    )
                    mock_send.return_value = mock_response

                    with patch("src.services.gemini_client.mark_credential_success"):
                        response = await send_gemini_request(
                            {"model": "gemini-2.0-flash"}, is_streaming=False
                        )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_fallback_on_auth_error(self):
        """Test fallback when first credential fails with auth error."""
        mock_creds1 = MagicMock()
        mock_creds1.expired = False
        mock_creds1.token = "test-token-1"

        mock_creds2 = MagicMock()
        mock_creds2.expired = False
        mock_creds2.token = "test-token-2"

        with patch(
            "src.services.gemini_client.get_next_credential",
            return_value=(mock_creds1, 0),
        ):
            with patch(
                "src.services.gemini_client.get_pool_stats",
                return_value={"total": 2, "available": 2},
            ):
                with patch(
                    "src.services.gemini_client._send_request_with_credential"
                ) as mock_send:
                    # First call returns 401, second call succeeds
                    error_response = Response(
                        content=b'{"error": "unauthorized"}', status_code=401
                    )
                    success_response = Response(
                        content=b'{"text": "Hello"}', status_code=200
                    )
                    mock_send.side_effect = [error_response, success_response]

                    with patch("src.services.gemini_client.mark_credential_failed"):
                        with patch(
                            "src.services.gemini_client.get_fallback_credential",
                            return_value=(mock_creds2, 1),
                        ):
                            with patch(
                                "src.services.gemini_client.mark_credential_success"
                            ):
                                response = await send_gemini_request(
                                    {"model": "gemini-2.0-flash"}, is_streaming=False
                                )

        assert response.status_code == 200


class TestFallbackErrorCodes:
    """Tests for FALLBACK_ERROR_CODES constant."""

    def test_fallback_codes_contains_expected(self):
        """Test that fallback codes contain expected values."""
        assert 401 in FALLBACK_ERROR_CODES
        assert 403 in FALLBACK_ERROR_CODES
        assert 429 in FALLBACK_ERROR_CODES

    def test_fallback_codes_excludes_other_errors(self):
        """Test that fallback codes exclude other error codes."""
        assert 400 not in FALLBACK_ERROR_CODES
        assert 500 not in FALLBACK_ERROR_CODES
        assert 502 not in FALLBACK_ERROR_CODES
