import pytest

from det_rep.gemini import validate_gateway_manifest
from det_rep.llm import OpenAICompatibleCorrector


def test_gateway_manifest_must_match_exact_model_and_revision():
    value = {
        "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
        "logical_model": "openai/gemini-2.5-flash", "vertex_model": "gemini-2.5-flash",
        "vertex_location": "europe-west4", "gateway_release": "r1", "cloud_run_revision": "abc",
    }
    assert len(validate_gateway_manifest(value, "openai/gemini-2.5-flash")) == 64
    with pytest.raises(ValueError, match="cloud_run_revision"):
        validate_gateway_manifest({**value, "cloud_run_revision": ""}, "openai/gemini-2.5-flash")


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
