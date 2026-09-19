from pathlib import Path
from types import SimpleNamespace

import yaml

from det_rep.contracts import Example
from det_rep.core.extract import Graph
from det_rep.core.matching import DictEmbedder
from det_rep.core.verifier import EvidenceSpan
from det_rep.evidence import build_evidence
from det_rep.gemini import GeminiFeedbackProducer, runtime_config


MANIFEST = {
    "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
    "logical_model": "openai/gemini-2.5-flash", "vertex_model": "gemini-2.5-flash",
    "vertex_location": "europe-west4", "gateway_release": "fixture", "cloud_run_revision": "fixture",
}


class FakeExtractor:
    def extract_reference(self, context, query):
        return Graph({"Alice", "Paris"}, {("Alice", "lives in", "Paris")}), Graph.empty()

    def extract(self, answer, kind="response"):
        return Graph({"Alice", "Paris"}, {("Alice", "lives in", "Paris")})


class FakeRelationVerifier:
    def verify(self, triple, context, query, matching_params=None):
        return SimpleNamespace(verdict="entailed", evidence=(EvidenceSpan("context", 0, 0, 21, "Alice lives in Paris.", 5),), cache_hit=False, protocol_fallback=False, fallback_reason=None)


class FakeClaimPipeline:
    def assess(self, answer, context, query, progress_hook=None):
        return [{
            "claim": {"text": answer, "start": 0, "end": len(answer), "sources": ["atomic"]},
            "verdict": "entailed", "evidence": [{"source": "context", "index": 0, "text": "Alice lives in Paris."}],
        }]


def test_existing_scoring_is_rendered_as_four_separate_feedback_blocks(tmp_path):
    snapshot = tmp_path / "sbert"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    base = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    cfg = runtime_config(base, MANIFEST, "https://example.invalid", snapshot, tmp_path / "cache")
    producer = GeminiFeedbackProducer(cfg, MANIFEST, tmp_path / "cache" / "feedback")
    producer.extractor = FakeExtractor()
    producer.embedder = DictEmbedder(dim=16)
    producer.relation_verifier = FakeRelationVerifier()
    producer.claim_pipeline = FakeClaimPipeline()
    example = Example("1", "llama31_8b_1", "Alice lives in Paris.", "Where does Alice live?", "Alice lives in Paris.", 0, "fixture", "train")
    evidence = build_evidence(example)
    record = producer.produce(example, example.original_answer, evidence)
    assert record.entities and all(item["grounded"] for item in record.entities)
    assert record.relations[0]["verdict"] == "entailed"
    assert record.claims[0]["verdict"] == "entailed"
    assert record.links[0]["sentence_ids"] == ["C0"]
    assert "evidence" not in record.claims[0]
    # The second call reads the outer feedback cache and cannot call a model.
    producer.extractor = None
    assert producer.produce(example, example.original_answer, evidence) == record


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
