import pytest

from det_rep.gemini import validate_gateway_manifest
from det_rep.llm import OpenAICompatibleCorrector
from det_rep.core.dspy_adapter import GatewayIdentityDriftError, validate_completion_envelope
from det_rep.core.config import Config, litellm_transport_model
from gemini_gateway.settings import GatewaySettings


def test_gateway_manifest_must_match_exact_model_and_revision():
    value = {
        "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
        "logical_model": "openai/gemini-3.5-flash", "vertex_model": "gemini-3.5-flash",
        "vertex_location": "eu", "gateway_release": "r1", "cloud_run_revision": "abc",
    }
    assert len(validate_gateway_manifest(value, "openai/gemini-3.5-flash")) == 64
    with pytest.raises(ValueError, match="cloud_run_revision"):
        validate_gateway_manifest({**value, "cloud_run_revision": ""}, "openai/gemini-3.5-flash")


def test_litellm_preserves_gateway_model_after_provider_prefix_is_removed():
    gateway = Config({"llm": {"model": "openai/gemini-3.5-flash", "structured_output_backend": "vertex"}})
    local = Config({"llm": {"model": "openai/local-model", "structured_output_backend": "xgrammar"}})
    assert gateway.llm.model == "openai/gemini-3.5-flash"
    assert litellm_transport_model(gateway) == "openai/openai/gemini-3.5-flash"
    assert litellm_transport_model(local) == "openai/local-model"


def test_gemini_completion_guard_accepts_only_pinned_gateway_revision(monkeypatch):
    settings = GatewaySettings("project", "secret", gateway_release="release", cloud_run_revision="revision")
    monkeypatch.setenv("EXPECTED_GATEWAY_MANIFEST_SHA256", settings.manifest_sha256)
    response = {
        "system_fingerprint": settings.system_fingerprint,
        "choices": [{"finish_reason": "stop"}],
    }
    validate_completion_envelope(response)
    changed = GatewaySettings("project", "secret", gateway_release="release", cloud_run_revision="other")
    with pytest.raises(GatewayIdentityDriftError):
        validate_completion_envelope({**response, "system_fingerprint": changed.system_fingerprint})


def test_generation_cache_fingerprint_changes_with_checkpoint_and_never_calls_cache_only(tmp_path):
    base = dict(base_url="http://localhost:8000/v1", model="meta-llama/Llama-3.1-8B-Instruct", cache_dir=tmp_path)
    one = OpenAICompatibleCorrector(**base, checkpoint="weights-a", cache_only=True)
    two = OpenAICompatibleCorrector(**base, checkpoint="weights-b", cache_only=True)
    assert one.fingerprint != two.fingerprint
    with pytest.raises(Exception, match="cache-only"):
        one.generate("prompt", arm="B", iteration=1)


def test_stateless_vllm_request_is_cached_by_prompt_and_arm(tmp_path):
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "Paris."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 2}}

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, json, headers):
            self.calls.append((url, json, headers))
            return Response()

    client = Client()
    settings = dict(base_url="http://localhost:8000/v1", model="meta-llama/Llama-3.1-8B-Instruct", checkpoint="weights-a", cache_dir=tmp_path, client=client)
    corrector = OpenAICompatibleCorrector(**settings)
    assert corrector.generate("same", arm="B", iteration=1).answer == "Paris."
    assert corrector.generate("same", arm="B", iteration=1).cache_hit
    corrector.generate("same", arm="E", iteration=1)
    corrector.generate("changed", arm="B", iteration=1)
    assert len(client.calls) == 3
    assert all(len(call[1]["messages"]) == 2 for call in client.calls)
    assert all("Authorization" not in call[2] for call in client.calls)


def test_vllm_preflight_pins_the_served_model_metadata(tmp_path):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "wrong"}, {"id": "llama", "root": "weights-a", "max_model_len": 8192}]}

    class Client:
        def get(self, url, headers):
            assert url == "http://localhost:8000/v1/models"
            return Response()

    corrector = OpenAICompatibleCorrector(
        base_url="http://localhost:8000/v1", model="llama", checkpoint="weights-a",
        cache_dir=tmp_path, client=Client(),
    )
    before = corrector.fingerprint
    assert corrector.preflight()["max_model_len"] == 8192
    assert corrector.fingerprint != before
