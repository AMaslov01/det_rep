"""Strict, deliberately small OpenAI Chat Completions request surface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .settings import GatewaySettings
from .vertex import VertexRequest


class ClientInputError(ValueError):
    def __init__(self, message: str, parameter: str | None = None) -> None:
        super().__init__(message)
        self.parameter = parameter


@dataclass(frozen=True)
class ChatRequest:
    stream: bool
    vertex_request: VertexRequest


SUPPORTED_FIELDS = frozenset({
    "model", "messages", "stream", "max_tokens", "max_completion_tokens", "temperature", "response_format",
})


def _require_text_message(value: Any, index: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ClientInputError("each message must be an object", f"messages[{index}]")
    role = value.get("role")
    if role not in {"system", "user", "assistant"}:
        raise ClientInputError("message role must be system, user, or assistant", f"messages[{index}].role")
    content = value.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ClientInputError("only non-empty text message content is supported", f"messages[{index}].content")
    unexpected = set(value) - {"role", "content"}
    if unexpected:
        raise ClientInputError(f"unsupported message field: {sorted(unexpected)[0]}", f"messages[{index}]")
    return {"role": role, "content": content}


def _parse_positive_int(value: Any, parameter: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65_536:
        raise ClientInputError("must be an integer from 1 through 65536", parameter)
    return value


def _parse_response_format(value: Any) -> tuple[bool, dict[str, Any] | None]:
    if value is None:
        return False, None
    if not isinstance(value, dict):
        raise ClientInputError("response_format must be an object", "response_format")
    kind = value.get("type")
    if kind == "json_object" and set(value) == {"type"}:
        return True, None
    if kind != "json_schema" or set(value) - {"type", "json_schema"}:
        raise ClientInputError("only json_object and json_schema response formats are supported", "response_format")
    json_schema = value.get("json_schema")
    if not isinstance(json_schema, dict) or set(json_schema) - {"name", "schema", "strict"}:
        raise ClientInputError("json_schema must contain only name, schema, and strict", "response_format.json_schema")
    if not isinstance(json_schema.get("name"), str) or not json_schema["name"].strip():
        raise ClientInputError("json_schema.name must be a non-empty string", "response_format.json_schema.name")
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        raise ClientInputError("json_schema.schema must be an object", "response_format.json_schema.schema")
    if "strict" in json_schema and not isinstance(json_schema["strict"], bool):
        raise ClientInputError("json_schema.strict must be a boolean", "response_format.json_schema.strict")
    return True, schema


def parse_chat_request(value: Any, settings: GatewaySettings) -> ChatRequest:
    if not isinstance(value, dict):
        raise ClientInputError("request body must be a JSON object")
    unsupported = set(value) - SUPPORTED_FIELDS
    if unsupported:
        raise ClientInputError(f"unsupported request field: {sorted(unsupported)[0]}", sorted(unsupported)[0])
    if value.get("model") != settings.logical_model:
        raise ClientInputError(f"only model {settings.logical_model} is available", "model")
    raw_messages = value.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ClientInputError("messages must be a non-empty list", "messages")
    messages = tuple(_require_text_message(item, index) for index, item in enumerate(raw_messages))
    non_system = [item for item in messages if item["role"] != "system"]
    if not non_system or non_system[-1]["role"] != "user":
        raise ClientInputError("the last non-system message must have role user", "messages")
    stream = value.get("stream", False)
    if not isinstance(stream, bool):
        raise ClientInputError("stream must be a boolean", "stream")
    max_tokens = value.get("max_completion_tokens", value.get("max_tokens"))
    if "max_tokens" in value and "max_completion_tokens" in value and value["max_tokens"] != value["max_completion_tokens"]:
        raise ClientInputError("max_tokens and max_completion_tokens disagree", "max_completion_tokens")
    if max_tokens is not None:
        max_tokens = _parse_positive_int(max_tokens, "max_completion_tokens" if "max_completion_tokens" in value else "max_tokens")
    temperature = value.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= float(temperature) <= 2:
            raise ClientInputError("temperature must be a number from 0 through 2", "temperature")
        temperature = float(temperature)
    json_mode, response_schema = _parse_response_format(value.get("response_format"))
    return ChatRequest(stream, VertexRequest(messages, max_tokens, temperature, response_schema, json_mode))
