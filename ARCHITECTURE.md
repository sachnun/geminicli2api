# Architecture

## Overview

**geminicli2api** is a proxy server that provides OpenAI, Anthropic, and OpenAI Responses API compatible endpoints for Google's Gemini models. It leverages the Google Cloud Code Assist API to access Gemini models, transforming requests and responses between different API formats.

### Key Features
- OpenAI-compatible `/v1/chat/completions` endpoint
- Anthropic-compatible `/v1/messages` endpoint
- OpenAI Responses API `/v1/responses` endpoint
- Native Gemini API passthrough via `/v1beta/` endpoints
- Multi-credential support with round-robin selection and automatic failover
- Streaming responses via Server-Sent Events (SSE)
- Web search grounding via `tools` parameter
- Reasoning/thinking control for extended thinking models

---

## Directory Structure

```
geminicli2api/
├── run.py                    # Entry point - starts uvicorn server
├── src/
│   ├── main.py               # FastAPI app initialization and routers
│   ├── config.py             # Centralized configuration and constants
│   ├── utils.py              # Shared utility functions
│   ├── cli.py                # CLI interface
│   │
│   ├── routes/               # HTTP endpoint handlers
│   │   ├── __init__.py       # Router exports
│   │   ├── openai.py         # /v1/chat/completions, /v1/models
│   │   ├── anthropic.py      # /v1/messages
│   │   ├── responses.py      # /v1/responses (Responses API)
│   │   ├── gemini.py         # /v1beta/* (native Gemini passthrough)
│   │   ├── utils.py          # Shared route utilities
│   │   └── transformers/     # Request/response format converters
│   │       ├── __init__.py
│   │       ├── openai.py     # OpenAI <-> Gemini transformations
│   │       ├── anthropic.py  # Anthropic <-> Gemini transformations
│   │       └── responses.py  # Responses API <-> Gemini transformations
│   │
│   ├── schemas/              # Pydantic models for request/response validation
│   │   ├── __init__.py
│   │   ├── openai.py         # OpenAI request/response schemas
│   │   ├── anthropic.py      # Anthropic request/response schemas
│   │   └── responses.py      # Responses API schemas
│   │
│   ├── services/             # Business logic layer
│   │   ├── __init__.py
│   │   ├── auth.py           # OAuth, credential pool, authentication
│   │   └── gemini_client.py  # HTTP client for Gemini API
│   │
│   └── models/               # Model definitions and helpers
│       ├── __init__.py
│       ├── gemini.py         # SUPPORTED_MODELS list
│       └── helpers.py        # Model validation, name helpers
│
└── tests/                    # Test suite
```

---

## Request Flow

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT REQUEST                                  │
│           (OpenAI / Anthropic / Responses / Native Gemini format)           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI APPLICATION                                │
│                              (src/main.py)                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   /v1/chat   │  │ /v1/messages │  │ /v1/responses│  │  /v1beta/*   │    │
│  │ /completions │  │  (Anthropic) │  │ (Responses)  │  │   (Native)   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROUTE HANDLERS                                     │
│                         (src/routes/*.py)                                    │
│                                                                              │
│   1. Validate request using Pydantic schemas (schemas/*.py)                 │
│   2. Authenticate user (services/auth.py)                                   │
│   3. Transform request via transformers (routes/transformers/*.py)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GEMINI CLIENT                                      │
│                     (src/services/gemini_client.py)                         │
│                                                                              │
│   1. Get credential from pool (round-robin selection)                       │
│   2. Refresh OAuth token if expired                                         │
│   3. Get/discover project ID                                                │
│   4. Onboard user if needed                                                 │
│   5. Build final payload with model, project, request                       │
│   6. Send to Google Cloud Code Assist API                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GOOGLE CLOUD CODE ASSIST API                           │
│                  (cloudcode-pa.googleapis.com)                              │
│                                                                              │
│   Endpoints:                                                                 │
│   - /v1internal:generateContent     (non-streaming)                         │
│   - /v1internal:streamGenerateContent?alt=sse  (streaming)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESPONSE TRANSFORMATION                               │
│                     (src/routes/transformers/*.py)                          │
│                                                                              │
│   Transform Gemini response back to original format:                        │
│   - gemini_response_to_openai()                                             │
│   - gemini_response_to_anthropic()                                          │
│   - gemini_response_to_responses()                                          │
│   - (Native: passthrough with minimal processing)                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             CLIENT RESPONSE                                  │
│           (OpenAI / Anthropic / Responses / Native Gemini format)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Request Flow (OpenAI Example)

```
POST /v1/chat/completions
        │
        ▼
┌───────────────────────────────┐
│ 1. Pydantic Validation        │
│    ChatCompletionRequest      │
│    (schemas/openai.py)        │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 2. Authentication             │
│    authenticate_user()        │
│    (services/auth.py)         │
│    - Bearer token             │
│    - API key query param      │
│    - Basic auth               │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 3. Model Validation           │
│    validate_model()           │
│    (models/helpers.py)        │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 4. Request Transformation     │
│    openai_request_to_gemini() │
│    (transformers/openai.py)   │
│                               │
│    Converts:                  │
│    - messages → contents      │
│    - tools → functionDecl.    │
│    - temperature, top_p, etc. │
│    - reasoning_effort →       │
│      thinkingConfig           │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 5. Build Gemini Payload       │
│    build_gemini_payload_      │
│    from_openai()              │
│    (gemini_client.py)         │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 6. Send Request               │
│    send_gemini_request()      │
│    (gemini_client.py)         │
│                               │
│    - Select credential        │
│    - Refresh token if needed  │
│    - Get project ID           │
│    - POST to Google API       │
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ 7. Response Transformation    │
│    gemini_response_to_openai()│
│    (transformers/openai.py)   │
│                               │
│    Converts:                  │
│    - candidates → choices     │
│    - parts → content          │
│    - functionCall → tool_calls│
│    - thought → reasoning_     │
│      content                  │
└───────────────────────────────┘
        │
        ▼
    JSON Response
```

---

## Key Components

### Routes (`src/routes/`)

| File | Endpoints | Description |
|------|-----------|-------------|
| `openai.py` | `/v1/chat/completions`, `/v1/models` | OpenAI Chat Completions API compatible endpoint. Supports streaming, function calling, and reasoning effort. |
| `anthropic.py` | `/v1/messages` | Anthropic Messages API compatible endpoint. Supports extended thinking, tool use, and streaming with Anthropic-style events. |
| `responses.py` | `/v1/responses` | OpenAI Responses API compatible endpoint. Uses `input` instead of `messages`, simpler tool format. |
| `gemini.py` | `/v1beta/*` | Native Gemini API passthrough. Proxies requests directly to Google API with minimal transformation. |
| `utils.py` | - | Shared utilities: error response creation, chunk decoding, non-streaming handler. |

### Transformers (`src/routes/transformers/`)

Transformers handle bidirectional conversion between different API formats and Gemini's native format.

| File | Key Functions | Description |
|------|---------------|-------------|
| `openai.py` | `openai_request_to_gemini()`, `gemini_response_to_openai()`, `gemini_stream_chunk_to_openai()` | Handles OpenAI ↔ Gemini conversion. Maps messages to contents, tools to function declarations, and handles thinking config. |
| `anthropic.py` | `anthropic_request_to_gemini()`, `gemini_response_to_anthropic()`, `AnthropicStreamProcessor` | Handles Anthropic ↔ Gemini conversion. Includes stateful stream processor for Anthropic SSE events. |
| `responses.py` | `responses_request_to_gemini()`, `gemini_response_to_responses()`, `gemini_stream_chunk_to_responses_events()` | Handles Responses API ↔ Gemini conversion. |

#### Transformation Examples

**OpenAI to Gemini:**
```python
# OpenAI format
{
    "model": "gemini-2.5-pro",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
    "tools": [{"type": "function", "function": {...}}],
    "reasoning_effort": "high"
}

# Gemini format
{
    "model": "gemini-2.5-pro",
    "contents": [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ],
    "tools": [{"functionDeclarations": [...]}],
    "generationConfig": {
        "thinkingConfig": {"thinkingBudget": 32768, "includeThoughts": True}
    }
}
```

### Services (`src/services/`)

#### `gemini_client.py`

The core HTTP client for communicating with Google's Code Assist API.

**Key Functions:**
- `send_gemini_request()` - Main entry point. Handles credential selection, token refresh, and request dispatch.
- `_send_request_with_credential()` - Sends request with a specific credential.
- `_stream_generator()` - Async generator for streaming responses.
- `_make_non_streaming_request()` - Handles non-streaming requests.
- `build_gemini_payload_from_openai()` - Constructs final payload from transformed request.

**Request Flow:**
```
send_gemini_request()
    │
    ├── get_next_credential()      # Round-robin credential selection
    │
    ├── _send_request_with_credential()
    │       │
    │       ├── Refresh token if expired
    │       ├── Get project ID (cached or discovered)
    │       ├── Onboard user if needed
    │       └── POST to Google API
    │
    └── Handle fallback on error (401, 403, 429)
```

#### `auth.py`

Handles OAuth2 authentication and multi-credential management.

**Key Classes:**
- `CredentialPool` - Manages multiple credentials with round-robin selection and failure tracking.
- `CredentialEntry` - Represents a single credential with metadata.

**Key Functions:**
- `authenticate_user()` - FastAPI dependency for request authentication.
- `get_credentials()` - Load or obtain OAuth credentials (single credential mode).
- `get_next_credential()` - Get next available credential from pool.
- `initialize_credential_pool()` - Load credentials from multiple sources.
- `onboard_user()` - Ensure user is onboarded to Code Assist.
- `get_user_project_id()` - Discover user's GCP project ID.

**Credential Sources (in priority order):**
1. `GEMINI_CREDENTIALS_1`, `GEMINI_CREDENTIALS_2`, ... (indexed env vars)
2. `GEMINI_CREDENTIAL_FILES` (comma-separated file paths)
3. `GEMINI_CREDENTIALS` (single JSON env var)
4. `oauth_creds.json` (default credential file)
5. Interactive OAuth flow (if no credentials found)

### Schemas (`src/schemas/`)

Pydantic models for request/response validation.

| File | Key Models | Description |
|------|------------|-------------|
| `openai.py` | `ChatCompletionRequest`, `ChatMessage`, `Tool`, `ToolCall` | OpenAI API request/response schemas. |
| `anthropic.py` | `MessagesRequest`, `Message`, `Tool`, `ThinkingConfig` | Anthropic API schemas with content blocks. |
| `responses.py` | `ResponsesRequest`, `ResponsesResponse`, `ResponsesFunctionCall` | Responses API schemas. |

### Models (`src/models/`)

| File | Description |
|------|-------------|
| `gemini.py` | `SUPPORTED_MODELS` list with model specs (name, limits, capabilities). |
| `helpers.py` | Utility functions: `validate_model()`, `get_base_model_name()`, `should_include_thoughts()`. |

### Configuration (`src/config.py`)

Centralized configuration constants:

```python
# API Endpoints
CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# OAuth Configuration
CLIENT_ID = "..."
CLIENT_SECRET = "..."
SCOPES = [...]

# Timeouts
REQUEST_TIMEOUT = 300      # 5 minutes
STREAMING_TIMEOUT = 600    # 10 minutes

# Thinking Budgets
THINKING_MAX_BUDGETS = {"gemini-2.5-flash": 24576, "gemini-2.5-pro": 32768, ...}
THINKING_MINIMAL_BUDGETS = {"gemini-2.5-flash": 0, "gemini-2.5-pro": 128, ...}

# Safety Settings
DEFAULT_SAFETY_SETTINGS = [...]  # All categories set to BLOCK_NONE

# Authentication
GEMINI_AUTH_PASSWORD = os.getenv("GEMINI_AUTH_PASSWORD")  # API key for proxy auth
```

---

## Streaming Architecture

Streaming uses Server-Sent Events (SSE) with async generators.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STREAMING FLOW                                │
└─────────────────────────────────────────────────────────────────┘

Client                    Proxy                     Google API
   │                        │                           │
   │  POST (stream=true)    │                           │
   ├───────────────────────>│                           │
   │                        │  POST ?alt=sse            │
   │                        ├──────────────────────────>│
   │                        │                           │
   │                        │  SSE: data: {...}         │
   │                        │<──────────────────────────┤
   │  SSE: data: {...}      │                           │
   │<───────────────────────┤  (transform)              │
   │                        │                           │
   │                        │  SSE: data: {...}         │
   │                        │<──────────────────────────┤
   │  SSE: data: {...}      │                           │
   │<───────────────────────┤  (transform)              │
   │                        │                           │
   │                        │  (connection closed)      │
   │  SSE: data: [DONE]     │<──────────────────────────┤
   │<───────────────────────┤                           │
   │                        │                           │
```

### Implementation Details

**OpenAI Streaming:**
```python
async def _stream_openai_response(gemini_payload, model):
    response = await send_gemini_request(gemini_payload, is_streaming=True)
    
    async for chunk in response.body_iterator:
        gemini_chunk = json.loads(chunk[6:])  # Remove "data: " prefix
        openai_chunk = gemini_stream_chunk_to_openai(gemini_chunk, model, response_id)
        yield f"data: {json.dumps(openai_chunk)}\n\n"
    
    yield "data: [DONE]\n\n"
```

**Anthropic Streaming (Stateful):**
```python
class AnthropicStreamProcessor:
    """Maintains state across chunks for proper event emission."""
    
    def process_chunk(self, gemini_chunk):
        events = []
        # Emit message_start on first chunk
        if not self.has_started:
            events.append(create_anthropic_stream_message_start(...))
        
        # Track content blocks, emit start/delta/stop events
        for part in gemini_chunk["candidates"][0]["content"]["parts"]:
            # Handle text, thinking, function calls
            ...
        
        return events
    
    def finalize(self):
        # Emit message_delta and message_stop
        return [create_anthropic_message_delta(...), create_anthropic_message_stop()]
```

---

## Authentication Flow

### Proxy Authentication (Client → Proxy)

The proxy supports multiple authentication methods:

```
┌─────────────────────────────────────────────────────────────┐
│                  AUTHENTICATION METHODS                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Bearer Token                                             │
│     Authorization: Bearer <GEMINI_AUTH_PASSWORD>             │
│                                                              │
│  2. API Key Query Parameter                                  │
│     ?key=<GEMINI_AUTH_PASSWORD>                              │
│                                                              │
│  3. x-goog-api-key Header                                    │
│     x-goog-api-key: <GEMINI_AUTH_PASSWORD>                   │
│                                                              │
│  4. x-api-key Header (Anthropic style)                       │
│     x-api-key: <GEMINI_AUTH_PASSWORD>                        │
│                                                              │
│  5. Basic Auth                                               │
│     Authorization: Basic base64(user:GEMINI_AUTH_PASSWORD)   │
│                                                              │
│  If GEMINI_AUTH_PASSWORD is not set, auth is disabled.       │
└─────────────────────────────────────────────────────────────┘
```

### OAuth Authentication (Proxy → Google)

```
┌─────────────────────────────────────────────────────────────┐
│                    OAUTH FLOW                                │
└─────────────────────────────────────────────────────────────┘

First Run (No Credentials):
  1. Start local HTTP server on port 8080
  2. Display OAuth URL to user
  3. User authenticates in browser
  4. Receive callback with auth code
  5. Exchange code for tokens
  6. Save to oauth_creds.json

Subsequent Runs:
  1. Load credentials from file/env
  2. Refresh token if expired
  3. Use access token for API calls

Multi-Credential Mode:
  1. Load from GEMINI_CREDENTIALS_1, _2, ... env vars
  2. Or load from GEMINI_CREDENTIAL_FILES
  3. Round-robin selection across credentials
  4. Automatic fallback on 401/403/429 errors
  5. 5-minute recovery time for failed credentials
```

### Credential Pool Management

```python
class CredentialPool:
    """Round-robin credential selection with failure handling."""
    
    def get_next(self):
        """Get next available credential, skipping failed ones."""
        # Skip credentials that failed < 5 minutes ago
        # Return (CredentialEntry, index)
        
    def get_fallback(self, exclude_index):
        """Get fallback when primary credential fails."""
        
    def mark_failed(self, index):
        """Mark credential as temporarily failed."""
        
    def mark_success(self, index):
        """Clear failed status on success."""
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8888` |
| `GEMINI_AUTH_PASSWORD` | API key for proxy authentication | None (no auth) |
| `GEMINI_CREDENTIALS` | Single credential JSON | None |
| `GEMINI_CREDENTIALS_1`, `_2`, ... | Multiple credential JSONs | None |
| `GEMINI_CREDENTIAL_FILES` | Comma-separated credential file paths | None |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID override | None |
| `GOOGLE_APPLICATION_CREDENTIALS` | Custom credential file path | `oauth_creds.json` |

### Credential JSON Format

```json
{
    "client_id": "...",
    "client_secret": "...",
    "refresh_token": "...",
    "token": "...",
    "expiry": "2024-01-01T00:00:00Z",
    "project_id": "..."
}
```

---

## Adding New Features

### Adding a New API Format

1. **Create Schema** (`src/schemas/newformat.py`):
   ```python
   from pydantic import BaseModel, Field
   
   class NewFormatRequest(BaseModel):
       model: str
       # ... other fields
   ```

2. **Create Transformer** (`src/routes/transformers/newformat.py`):
   ```python
   def newformat_request_to_gemini(request: NewFormatRequest) -> Dict[str, Any]:
       """Transform NewFormat request to Gemini format."""
       contents = [...]  # Transform messages/input
       return {
           "contents": contents,
           "generationConfig": {...},
           "model": request.model,
       }
   
   def gemini_response_to_newformat(gemini_response: Dict, model: str) -> Dict:
       """Transform Gemini response to NewFormat."""
       return {...}
   ```

3. **Create Route Handler** (`src/routes/newformat.py`):
   ```python
   from fastapi import APIRouter, Depends
   from ..services.auth import authenticate_user
   from ..services.gemini_client import send_gemini_request, build_gemini_payload_from_openai
   from .transformers import newformat_request_to_gemini, gemini_response_to_newformat
   
   router = APIRouter()
   
   @router.post("/v1/newformat")
   async def newformat_endpoint(
       request: NewFormatRequest,
       username: str = Depends(authenticate_user),
   ):
       gemini_data = newformat_request_to_gemini(request)
       payload = build_gemini_payload_from_openai(gemini_data)
       response = await send_gemini_request(payload, is_streaming=request.stream)
       return gemini_response_to_newformat(response, request.model)
   ```

4. **Register Router** (`src/main.py`):
   ```python
   from .routes.newformat import router as newformat_router
   app.include_router(newformat_router)
   ```

### Adding a New Model

1. Add model definition to `src/models/gemini.py`:
   ```python
   SUPPORTED_MODELS.append({
       "name": "models/gemini-new-model",
       "displayName": "Gemini New Model",
       "inputTokenLimit": 1048576,
       "outputTokenLimit": 65535,
       # ...
   })
   ```

2. If model has special thinking budgets, update `src/config.py`:
   ```python
   THINKING_MAX_BUDGETS["gemini-new-model"] = 32768
   THINKING_MINIMAL_BUDGETS["gemini-new-model"] = 128
   ```

### Adding a New Tool Type

1. Update transformer to recognize the tool type:
   ```python
   # In transformers/openai.py
   def _transform_openai_tools_to_gemini(tools):
       for tool in tools:
           if tool.type == "new_tool_type":
               # Handle new tool type
               pass
   ```

2. Handle tool in response transformation if needed.

---

## Error Handling

All errors follow a standardized format:

```python
# From config.py
def create_error_response(message, error_type, code):
    return {
        "error": {
            "message": message,
            "type": error_type,  # "api_error", "invalid_request_error", etc.
            "code": code,        # HTTP status code
        }
    }
```

Error types:
- `api_error` - Server/upstream errors
- `invalid_request_error` - Validation/client errors
- `authentication_error` - Auth failures
- `not_found_error` - Resource not found
- `rate_limit_error` - Rate limiting (429)

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_auth.py -v

# Run with coverage
uv run pytest tests/ --cov=src
```

Test files:
- `tests/test_auth.py` - Authentication tests
- `tests/test_cli.py` - CLI tests
- `tests/test_function_calling.py` - Tool/function calling tests
- `tests/test_multi_credential.py` - Credential pool tests
