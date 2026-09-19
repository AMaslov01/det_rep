"""HTTP boundary for the shared Gemini gateway."""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .keys import KeyStore, SecretManagerKeySource
from .request import ClientInputError, parse_chat_request
from .settings import GatewaySettings
from .vertex import GoogleVertexClient, VertexClient, VertexGatewayError, VertexResult, VertexStreamChunk


logger = logging.getLogger("gemini_gateway")


def _error(message: str, status_code: int, *, parameter: str | None = None, retry_after: str | None = None, request_id: str | None = None) -> JSONResponse:
    payload: dict[str, Any] = {"error": {"message": message, "type": "invalid_request_error" if status_code == 400 else "server_error", "code": None}}
    if status_code == 401:
        payload["error"].update({"type": "authentication_error", "code": "invalid_api_key"})
    elif status_code == 429:
        payload["error"].update({"type": "rate_limit_error", "code": "rate_limit_exceeded"})
    elif status_code >= 500:
        payload["error"].update({"type": "api_error", "code": "upstream_error"})
    if parameter is not None:
        payload["error"]["param"] = parameter
    headers = {"X-Request-ID": request_id} if request_id else {}
    if status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    if retry_after and status_code == 429:
        headers["Retry-After"] = retry_after
    return JSONResponse(payload, status_code=status_code, headers=headers)


def _usage(prompt_tokens: int | None, completion_tokens: int | None) -> dict[str, int | None]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": (prompt_tokens or 0) + (completion_tokens or 0),
    }


def _completion_payload(chat_id: str, created: int, settings: GatewaySettings, result: VertexResult) -> dict[str, Any]:
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": settings.logical_model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": result.finish_reason}],
        "usage": _usage(result.prompt_tokens, result.completion_tokens),
        "system_fingerprint": settings.system_fingerprint,
    }


def _sse(payload: dict[str, Any] | str) -> str:
    value = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {value}\n\n"


def create_app(
    settings: GatewaySettings | None = None,
    *,
    key_store: KeyStore | None = None,
    vertex_client: VertexClient | None = None,
) -> FastAPI:
    settings = settings or GatewaySettings.from_env()
    key_store = key_store or KeyStore(SecretManagerKeySource(settings.project_id, settings.key_secret_id), settings.key_cache_ttl_seconds)
    vertex_client = vertex_client or GoogleVertexClient(settings)
    app = FastAPI(title="Gemini OpenAI Gateway", docs_url=None, redoc_url=None, openapi_url=None)

    def request_id() -> str:
        return f"req_{uuid.uuid4().hex}"

    def authenticate(request: Request, identifier: str) -> tuple[str | None, JSONResponse | None]:
        try:
            key_id = key_store.authenticate(request.headers.get("Authorization"))
        except Exception:
            logger.exception("gateway_key_source_failed request_id=%s", identifier)
            return None, _error("authentication service is unavailable", 503, request_id=identifier)
        if key_id is None:
            return None, _error("invalid or missing API key", 401, request_id=identifier)
        return key_id, None

    def log_request(identifier: str, key_id: str, status: int, started: float, prompt_tokens: int | None = None, completion_tokens: int | None = None) -> None:
        logger.info(
            "gateway_request request_id=%s key_id=%s status=%d elapsed_ms=%d prompt_tokens=%s completion_tokens=%s",
            identifier, key_id, status, round((time.monotonic() - started) * 1000), prompt_tokens, completion_tokens,
        )

    def health_payload() -> dict[str, str]:
        return {"status": "ok", "model": settings.logical_model}

    # Cloud Run's public Google Front End reserves /healthz, so expose a
    # non-reserved liveness route while retaining /healthz for local runners.
    @app.get("/health")
    def health() -> dict[str, str]:
        return health_payload()

    @app.get("/healthz")
    def local_healthz() -> dict[str, str]:
        return health_payload()

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        identifier = request_id()
        key_id, failure = authenticate(request, identifier)
        if failure:
            return failure
        assert key_id is not None
        log_request(identifier, key_id, 200, time.monotonic())
        return JSONResponse({"object": "list", "data": [{"id": settings.logical_model, "object": "model", "created": 0, "owned_by": "google"}]}, headers={"X-Request-ID": identifier})

    @app.get("/v1/hallu/manifest")
    async def manifest(request: Request) -> JSONResponse:
        identifier = request_id()
        key_id, failure = authenticate(request, identifier)
        if failure:
            return failure
        assert key_id is not None
        log_request(identifier, key_id, 200, time.monotonic())
        return JSONResponse(settings.manifest(), headers={"X-Request-ID": identifier})

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Any:
        identifier = request_id()
        started = time.monotonic()
        key_id, failure = authenticate(request, identifier)
        if failure:
            return failure
        assert key_id is not None
        try:
            body = await request.json()
        except Exception:
            log_request(identifier, key_id, 400, started)
            return _error("request body must be valid JSON", 400, request_id=identifier)
        try:
            chat = parse_chat_request(body, settings)
        except ClientInputError as exc:
            log_request(identifier, key_id, 400, started)
            return _error(str(exc), 400, parameter=exc.parameter, request_id=identifier)
        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if not chat.stream:
            try:
                result = vertex_client.generate(chat.vertex_request)
            except VertexGatewayError as exc:
                log_request(identifier, key_id, exc.status_code, started)
                return _error(str(exc), exc.status_code, retry_after=exc.retry_after, request_id=identifier)
            log_request(identifier, key_id, 200, started, result.prompt_tokens, result.completion_tokens)
            return JSONResponse(_completion_payload(chat_id, created, settings, result), headers={"X-Request-ID": identifier})

        def events() -> Iterator[str]:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            finish_reason = "stop"
            status = 200
            try:
                for chunk in vertex_client.generate_stream(chat.vertex_request):
                    if chunk.text:
                        yield _sse({
                            "id": chat_id, "object": "chat.completion.chunk", "created": created,
                            "model": settings.logical_model,
                            "choices": [{"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}],
                            "system_fingerprint": settings.system_fingerprint,
                        })
                    if chunk.finish_reason:
                        finish_reason = chunk.finish_reason
                    prompt_tokens = chunk.prompt_tokens if chunk.prompt_tokens is not None else prompt_tokens
                    completion_tokens = chunk.completion_tokens if chunk.completion_tokens is not None else completion_tokens
                yield _sse({
                    "id": chat_id, "object": "chat.completion.chunk", "created": created,
                    "model": settings.logical_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    "usage": _usage(prompt_tokens, completion_tokens), "system_fingerprint": settings.system_fingerprint,
                })
                yield _sse("[DONE]")
            except VertexGatewayError as exc:
                status = exc.status_code
                yield _sse({"error": {"message": str(exc), "type": "api_error", "code": "upstream_error"}})
                yield _sse("[DONE]")
            finally:
                log_request(identifier, key_id, status, started, prompt_tokens, completion_tokens)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Request-ID": identifier, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()
