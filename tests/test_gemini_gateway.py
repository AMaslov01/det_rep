import json
import logging

from fastapi.testclient import TestClient

from gemini_gateway.app import create_app
from gemini_gateway.keys import KeyStore, key_digest
from gemini_gateway.settings import GatewaySettings
from gemini_gateway.vertex import VertexGatewayError, VertexResult, VertexStreamChunk


RAW_KEY = "gk_alice_a-very-secret-test-token"


class FakeVertex:
    def __init__(self) -> None:
        self.requests = []
        self.error = None

    def generate(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return VertexResult('{"answer":"Paris"}', "stop", 11, 5)

    def generate_stream(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        yield VertexStreamChunk("Par")
        yield VertexStreamChunk("is", "stop", 11, 2)


def settings():
    return GatewaySettings(
        project_id="test-project",
        key_secret_id="test-secret",
        gateway_release="abc123",
        cloud_run_revision="gateway-00001-test",
    )


def client(vertex=None, document=None):
    document = document or {"version": 1, "keys": [{"id": "alice", "sha256": key_digest(RAW_KEY)}]}
    return TestClient(create_app(settings(), key_store=KeyStore(lambda: document), vertex_client=vertex or FakeVertex()))


def headers():
    return {"Authorization": f"Bearer {RAW_KEY}"}


def base_body(**overrides):
    value = {
        "model": "openai/gemini-3.5-flash",
        "messages": [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Where is Paris?"}],
        "max_completion_tokens": 20,
    }
    value.update(overrides)
    return value


def test_chat_completion_translates_text_schema_and_returns_openai_shape():
    vertex = FakeVertex()
    response = client(vertex).post(
        "/v1/chat/completions",
        headers=headers(),
        json=base_body(response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True}}),
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "openai/gemini-3.5-flash"
    assert payload["choices"][0]["message"]["content"] == '{"answer":"Paris"}'
    assert payload["usage"] == {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
    assert payload["system_fingerprint"].startswith(settings().manifest_sha256[:16] + ":")
    request = vertex.requests[0]
    assert request.messages[0] == {"role": "system", "content": "Be concise."}
    assert request.response_schema == {"type": "object"}
    assert request.json_mode


def test_streaming_uses_sse_and_does_not_log_prompt(caplog):
    caplog.set_level(logging.INFO, logger="gemini_gateway")
    response = client().post("/v1/chat/completions", headers=headers(), json=base_body(stream=True))
    assert response.status_code == 200
    events = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    assert json.loads(events[0])["choices"][0]["delta"] == {"content": "Par"}
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "stop"
    assert events[-1] == "[DONE]"
    assert "Where is Paris?" not in caplog.text
    assert RAW_KEY not in caplog.text


def test_authentication_and_unsupported_content_are_rejected_before_vertex():
    vertex = FakeVertex()
    app_client = client(vertex)
    unauthenticated = app_client.post("/v1/chat/completions", json=base_body())
    assert unauthenticated.status_code == 401
    assert unauthenticated.headers["www-authenticate"] == "Bearer"
    image_message = app_client.post(
        "/v1/chat/completions", headers=headers(),
        json=base_body(messages=[{"role": "user", "content": [{"type": "image_url"}]}]),
    )
    assert image_message.status_code == 400
    assert image_message.json()["error"]["param"] == "messages[0].content"
    assert not vertex.requests


def test_upstream_rate_limit_keeps_retry_hint():
    vertex = FakeVertex()
    vertex.error = VertexGatewayError("Vertex AI request failed", 429, "7")
    response = client(vertex).post("/v1/chat/completions", headers=headers(), json=base_body())
    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_manifest_and_model_list_are_authenticated():
    app_client = client()
    assert app_client.get("/health").json()["status"] == "ok"
    assert app_client.get("/v1/hallu/manifest").status_code == 401
    manifest = app_client.get("/v1/hallu/manifest", headers=headers())
    assert manifest.status_code == 200
    assert manifest.json()["vertex_location"] == "eu"
    models = app_client.get("/v1/models", headers=headers()).json()
    assert models["data"][0]["id"] == "openai/gemini-3.5-flash"


def test_key_store_refresh_makes_revocation_effective_without_restart():
    now = [0.0]
    document = {"version": 1, "keys": [{"id": "alice", "sha256": key_digest(RAW_KEY)}]}
    store = KeyStore(lambda: document, ttl_seconds=60, now=lambda: now[0])
    assert store.authenticate(f"Bearer {RAW_KEY}") == "alice"
    document = {"version": 1, "keys": [{"id": "alice", "sha256": key_digest(RAW_KEY), "disabled": True}]}
    now[0] = 59
    assert store.authenticate(f"Bearer {RAW_KEY}") == "alice"
    now[0] = 60
    assert store.authenticate(f"Bearer {RAW_KEY}") is None
