import json
import pytest

from det_rep.arms import render_prompt, visible_feedback
from det_rep.contracts import ARM_CODES, Example, FeedbackRecord, Generation
from det_rep.evidence import build_evidence
from det_rep.runner import replay, run_arms
from det_rep.util import digest, text_digest


EXAMPLE = Example("12448", "llama31_8b_12448", "Alice lives in Paris. Bob lives in Rome.", "Where does Alice live?", "Alice lives in London.", 1, "fixture", "train")


class FakeProducer:
    fingerprint = "producer-fixture-v1"

    def produce(self, example, answer, evidence):
        return FeedbackRecord(
            example.source_id, text_digest(answer), digest(evidence.to_dict()), self.fingerprint,
            ({"name": "ENTITY_MARKER"},), ({"status": "RELATION_MARKER"},),
            ({"text": "CLAIM_MARKER", "verdict": "contradicted"},),
            ({"claim_id": "c0", "sentence_ids": ["C0"]},), "scorable",
        )


class FakeCorrector:
    fingerprint = "corrector-fixture-v1"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, *, arm, iteration):
        self.prompts.append((arm, iteration, prompt))
        return Generation(f"Revised {iteration}.", 10, 3, 0.1, "stop")


def test_twelve_arms_start_fresh_and_hide_other_blocks(tmp_path):
    producer, corrector = FakeProducer(), FakeCorrector()
    root = tmp_path / "run"
    summary = run_arms([EXAMPLE], producer, corrector, run_dir=root, cache_root=tmp_path / "cache", input_manifest_sha256="fixture", iterations=2)
    assert summary["completed"] == len(ARM_CODES) == 12
    assert len(corrector.prompts) == 24
    first = {arm: prompt for arm, iteration, prompt in corrector.prompts if iteration == 1}
    assert set(first) == set(ARM_CODES)
    assert all("Alice lives in London." in prompt for prompt in first.values())
    assert all("[C0] Alice lives in Paris." in prompt for prompt in first.values())
    assert "ENTITY_MARKER" not in first["B"]
    assert "RELATION_MARKER" not in first["E"]
    assert "CLAIM_MARKER" not in first["R"]
    assert "sentence_ids" not in first["C"]
    assert "sentence_ids" in first["CX"]
    assert "Revised 1." not in " ".join(first.values())
    assert all("Revised 1." in prompt for arm, iteration, prompt in corrector.prompts if iteration == 2)
    report = replay(root, tmp_path / "cache", [EXAMPLE])
    assert report["completed"] == 12 and report["missing"] == 0
    assignments = json.loads((root / "evaluation_assignment.json").read_text())
    assert len(assignments) == 12
    request_files = list((root / "evaluation_requests").glob("*.json"))
    assert request_files
    assert all(
        "arm" not in json.loads(path.read_text())
        and "source_id" not in json.loads(path.read_text())
        for path in request_files
    )
    assert replay(root, tmp_path / "cache", [EXAMPLE]) == report
    resume_corrector = FakeCorrector()
    resumed = run_arms([EXAMPLE], producer, resume_corrector, run_dir=root, cache_root=tmp_path / "cache", input_manifest_sha256="fixture", iterations=2)
    assert resumed["run_fingerprint"] == summary["run_fingerprint"]
    assert resumed["completed"] == 12
    assert resume_corrector.prompts == []


def test_replay_rejects_cache_mutation_without_calling_models(tmp_path, monkeypatch):
    root = tmp_path / "run"
    cache = tmp_path / "cache"
    run_arms([EXAMPLE], FakeProducer(), FakeCorrector(), run_dir=root, cache_root=cache, input_manifest_sha256="fixture")
    cache.mkdir()
    (cache / "new.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        replay(root, cache, [EXAMPLE])


def test_renderer_rejects_invalid_arm():
    evidence = build_evidence(EXAMPLE)
    feedback = FakeProducer().produce(EXAMPLE, EXAMPLE.original_answer, evidence)
    with pytest.raises(ValueError, match="unknown"):
        visible_feedback(feedback, "X")


def test_extraction_failure_for_source_12448_is_explicit_in_all_arms(tmp_path):
    class BrokenProducer(FakeProducer):
        calls = 0

        def produce(self, example, answer, evidence):
            self.calls += 1
            raise RuntimeError("private raw completion must not be logged")

    root = tmp_path / "run"
    producer = BrokenProducer()
    summary = run_arms([EXAMPLE], producer, FakeCorrector(), run_dir=root, cache_root=tmp_path / "cache", input_manifest_sha256="fixture")
    assert summary["completed"] == 0 and summary["failed"] == 12
    assert producer.calls == 1
    failure = json.loads((root / "failures" / "12448" / "B.json").read_text())
    assert failure["source_id"] == "12448" and failure["stage"] == "feedback"
    assert failure["error_type"] == "RuntimeError"
    assert "private raw completion" not in json.dumps(failure)
    assert replay(root, tmp_path / "cache", [EXAMPLE])["missing"] == 12
