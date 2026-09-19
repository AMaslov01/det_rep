"""Vertex AI adapter kept separate from the HTTP/OpenAI translation layer."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .settings import GatewaySettings


@dataclass(frozen=True)
class VertexRequest:
    messages: tuple[dict[str, str], ...]
    max_output_tokens: int | None
    temperature: float | None
    response_schema: dict[str, Any] | None
    json_mode: bool


@dataclass(frozen=True)
class VertexResult:
    text: str
    finish_reason: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class VertexStreamChunk:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class VertexGatewayError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class VertexClient(Protocol):
    def generate(self, request: VertexRequest) -> VertexResult: ...

    def generate_stream(self, request: VertexRequest) -> Iterator[VertexStreamChunk]: ...


def _usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None)


def _finish_reason(response: Any, default: str | None = "stop") -> str | None:
    candidates = getattr(response, "candidates", None) or []
    reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    if reason is None:
        return default
    value = str(getattr(reason, "name", reason)).upper()
    if "MAX" in value or "LENGTH" in value:
        return "length"
    if "STOP" in value:
        return "stop"
    return default


class GoogleVertexClient:
    """Lazy Google Gen AI SDK client using Cloud Run's service identity (ADC)."""

    def __init__(self, settings: GatewaySettings) -> None:
        self.settings = settings
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.settings.project_id:
                raise VertexGatewayError("GOOGLE_CLOUD_PROJECT is required", 500)
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise VertexGatewayError("google-genai is required", 500) from exc
            self._client = genai.Client(vertexai=True, project=self.settings.project_id, location=self.settings.vertex_location)
        return self._client

    @staticmethod
    def _contents(request: VertexRequest) -> tuple[list[dict[str, Any]], str | None]:
        system = [item["content"] for item in request.messages if item["role"] == "system"]
        contents = [
            {"role": "model" if item["role"] == "assistant" else "user", "parts": [{"text": item["content"]}]}
            for item in request.messages if item["role"] != "system"
        ]
        return contents, "\n\n".join(system) or None

    @staticmethod
    def _config(request: VertexRequest, system_instruction: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if request.max_output_tokens is not None:
            config["max_output_tokens"] = request.max_output_tokens
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.json_mode:
            config["response_mime_type"] = "application/json"
        if request.response_schema is not None:
            config["response_json_schema"] = request.response_schema
        return config

    @staticmethod
    def _translate_error(exc: Exception) -> VertexGatewayError:
        raw_status = getattr(exc, "code", None) or getattr(exc, "status_code", None) or 502
        if callable(raw_status):
            raw_status = raw_status()
        try:
            status = int(raw_status)
        except (TypeError, ValueError):
            status = 502
        if status not in {400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504}:
            status = 502
        retry_after = getattr(exc, "retry_after", None)
        return VertexGatewayError("Vertex AI request failed", status, str(retry_after) if retry_after else None)

    def generate(self, request: VertexRequest) -> VertexResult:
        try:
            contents, system = self._contents(request)
            response = self._get_client().models.generate_content(
                model=self.settings.vertex_model, contents=contents, config=self._config(request, system)
            )
            prompt, completion = _usage(response)
            return VertexResult(str(getattr(response, "text", "") or ""), _finish_reason(response), prompt, completion)
        except VertexGatewayError:
            raise
        except Exception as exc:  # pragma: no cover - exact SDK errors vary by release
            raise self._translate_error(exc) from exc

    def generate_stream(self, request: VertexRequest) -> Iterator[VertexStreamChunk]:
        try:
            contents, system = self._contents(request)
            for response in self._get_client().models.generate_content_stream(
                model=self.settings.vertex_model, contents=contents, config=self._config(request, system)
            ):
                prompt, completion = _usage(response)
                yield VertexStreamChunk(str(getattr(response, "text", "") or ""), _finish_reason(response, None), prompt, completion)
        except VertexGatewayError:
            raise
        except Exception as exc:  # pragma: no cover - exact SDK errors vary by release
            raise self._translate_error(exc) from exc
