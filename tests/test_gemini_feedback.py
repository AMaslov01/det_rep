import json
from pathlib import Path

import pytest
import yaml

from det_rep.contracts import Example
from det_rep.core.dspy_adapter import StructuredOutputParseError
from det_rep.core.extract import Graph
from det_rep.core.matching import DictEmbedder
from det_rep.evidence import build_evidence
from det_rep.gemini import (
    ClaimFeedbackError,
    EntityFeedbackError,
    GeminiFeedbackProducer,
    runtime_config,
)


MANIFEST = {
    "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
    "logical_model": "openai/gemini-3.5-flash", "vertex_model": "gemini-3.5-flash",
    "vertex_location": "eu", "gateway_release": "fixture", "cloud_run_revision": "fixture",
}
EXAMPLE = Example(
    "1", "llama31_8b_1", "Alice lives in Paris.", "Where does Alice live?",
    "She lives in London.", 1, "fixture", "train",
)


def config(root: Path):
    snapshot = root / "sbert"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}")
    base = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    return runtime_config(base, MANIFEST, "https://example.invalid", snapshot, root / "cache")


class FakeExtractor:
    calls = 0

    def extract_reference(self, context, query):
        self.calls += 1
        return Graph({"Alice", "Paris"}, set()), Graph.empty()

    def extract(self, answer, kind="response"):
        self.calls += 1
        return Graph({"Alice", "London"}, set())


class FakeClaimPipeline:
    calls = 0

    def assess(self, answer, context, query):
        self.calls += 1
        return [{
            "claim": {
                "text": "The answer states a move to London.",
                "sentence_id": "A0", "sentence_start": 0, "sentence_end": len(answer),
                "sources": ["veriscore"],
            },
            "verdict": "contradicted", "verifier_protocol_fallback": False,
            "verifier_fallback_reason": None,
            "evidence": [{"source": "context", "index": 0, "text": context}],
        }]


def test_components_are_separate_and_claim_anchor_is_the_focus_sentence(tmp_path):
    producer = GeminiFeedbackProducer(config(tmp_path), MANIFEST, tmp_path / "cache" / "feedback")
    extractor, claims = FakeExtractor(), FakeClaimPipeline()
    producer.extractor = extractor
    producer.embedder = DictEmbedder(dim=16)
    producer.claim_pipeline = claims
    evidence = build_evidence(EXAMPLE)

    c = producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, evidence)
    assert extractor.calls == 0 and claims.calls == 1
    assert c == ({
        "id": "c0", "text": "The answer states a move to London.",
        "sentence_id": "A0", "sentence_start": 0,
        "sentence_end": len(EXAMPLE.original_answer),
        "verdict": "contradicted", "candidate_sources": ["veriscore"],
        "protocol_fallback": False, "fallback_reason": None,
    },)
    assert EXAMPLE.original_answer[c[0]["sentence_start"]:c[0]["sentence_end"]] != c[0]["text"]
    assert "evidence" not in c[0]
    cache_file = next((tmp_path / "cache" / "feedback" / "claims").glob("*.json"))
    assert json.loads(cache_file.read_text())["audits"][0]["evidence"][0]["text"] == EXAMPLE.context

    e = producer.produce_entities(EXAMPLE, EXAMPLE.original_answer, evidence)
    assert extractor.calls == 2 and claims.calls == 1
    assert {item["name"] for item in e} == {"Alice", "London"}
    assert next(item for item in e if item["name"] == "London")["grounded"] is False

    producer.extractor = None
    producer.claim_pipeline = None
    assert producer.produce_entities(EXAMPLE, EXAMPLE.original_answer, evidence) == e
    assert producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, evidence) == c


def test_component_failures_are_typed_and_do_not_block_the_other_component(tmp_path):
    producer = GeminiFeedbackProducer(config(tmp_path), MANIFEST, tmp_path / "cache" / "feedback")
    producer.embedder = DictEmbedder(dim=16)
    evidence = build_evidence(EXAMPLE)

    class Broken:
        def extract_reference(self, context, query):
            raise RuntimeError("private response")

        def assess(self, answer, context, query):
            raise RuntimeError("private response")

    producer.extractor = Broken()
    producer.claim_pipeline = FakeClaimPipeline()
    assert producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, evidence)
    with pytest.raises(EntityFeedbackError) as caught:
        producer.produce_entities(EXAMPLE, EXAMPLE.original_answer, evidence)
    assert str(caught.value) == "entities feedback failed"

    producer.extractor = FakeExtractor()
    producer.claim_pipeline = Broken()
    other = Example("2", "llama31_8b_2", EXAMPLE.context, EXAMPLE.query, EXAMPLE.original_answer, 1, "fixture", "train")
    assert producer.produce_entities(other, other.original_answer, build_evidence(other))
    with pytest.raises(ClaimFeedbackError) as caught:
        producer.produce_claims(other, other.original_answer, build_evidence(other))
    assert str(caught.value) == "claims feedback failed"


def test_embedding_snapshot_change_invalidates_feedback_identity(tmp_path):
    snapshot = tmp_path / "sbert"
    snapshot.mkdir()
    config_file = snapshot / "config.json"
    config_file.write_text('{"revision": 1}')
    base = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    first = runtime_config(base, MANIFEST, "https://example.invalid", snapshot, tmp_path / "cache")
    config_file.write_text('{"revision": 2}')
    second = runtime_config(base, MANIFEST, "https://example.invalid", snapshot, tmp_path / "cache")
    assert first.matching.embedding_model_revision != second.matching.embedding_model_revision
    one = GeminiFeedbackProducer(first, MANIFEST, tmp_path / "cache" / "feedback", cache_only=True)
    two = GeminiFeedbackProducer(second, MANIFEST, tmp_path / "cache" / "feedback", cache_only=True)
    assert one.fingerprint != two.fingerprint


@pytest.mark.parametrize("failure", [TimeoutError("gateway unavailable"), StructuredOutputParseError("bad schema")])
@pytest.mark.parametrize("stage", ["extraction", "verification"])
def test_gemini_failure_cannot_become_successful_claim_feedback(tmp_path, failure, stage):
    producer = GeminiFeedbackProducer(config(tmp_path), MANIFEST, tmp_path / "cache" / "feedback")
    pipeline = producer.claim_pipeline

    def fail(*args, **kwargs):
        raise failure

    if stage == "extraction":
        pipeline.extractor._retry_validated_json = fail
    else:
        pipeline.extractor._call_json = lambda messages, schema, name, *, max_tokens: {"claims": ["She lives in London."]}
        pipeline.verifier._retry_validated_json = fail

    with pytest.raises(ClaimFeedbackError) as caught:
        producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, build_evidence(EXAMPLE))
    assert caught.value.component == "claims"
    assert not list((tmp_path / "cache" / "feedback" / "claims").glob("*.json"))


def test_cached_claim_fallback_is_rejected(tmp_path):
    producer = GeminiFeedbackProducer(config(tmp_path), MANIFEST, tmp_path / "cache" / "feedback")
    producer.claim_pipeline = FakeClaimPipeline()
    evidence = build_evidence(EXAMPLE)
    producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, evidence)
    path = next((tmp_path / "cache" / "feedback" / "claims").glob("*.json"))
    payload = json.loads(path.read_text())
    payload["items"][0]["protocol_fallback"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ClaimFeedbackError):
        producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, evidence)


def test_supplied_claim_embedder_failure_cannot_fall_back_to_lexical_success(tmp_path):
    producer = GeminiFeedbackProducer(config(tmp_path), MANIFEST, tmp_path / "cache" / "feedback")
    pipeline = producer.claim_pipeline
    pipeline.extractor._call_json = lambda messages, schema, name, *, max_tokens: {"claims": ["She lives in London."]}

    class BrokenEmbedder:
        def encode(self, texts):
            raise RuntimeError("embedding snapshot cannot load")

    pipeline.verifier.embedder = BrokenEmbedder()
    with pytest.raises(ClaimFeedbackError) as caught:
        producer.produce_claims(EXAMPLE, EXAMPLE.original_answer, build_evidence(EXAMPLE))
    assert caught.value.component == "claims"
    assert not list((tmp_path / "cache" / "feedback" / "claims").glob("*.json"))
