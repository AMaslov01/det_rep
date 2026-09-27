import json

import pytest

import det_rep.runner as runner_module
from det_rep.arms import visible_feedback
from det_rep.contracts import ARM_CODES, Example, Generation
from det_rep.runner import export_blind_audit, replay, run_arms


EXAMPLE = Example("12448", "llama31_8b_12448", "Alice lives in Paris. Bob lives in Rome.", "Where does Alice live?", "Alice lives in London.", 1, "fixture", "train")
SECOND_EXAMPLE = Example("12449", "llama31_8b_12449", "Eve lives in Lyon. Dan lives in Oslo.", "Where does Eve live?", "Eve lives in Madrid.", 0, "fixture", "train")


class FakeProducer:
    fingerprint = "producer-ec-fixture-v1"

    def __init__(self, *, fail_entities=False, fail_claims=False):
        self.calls = []
        self.fail_entities = fail_entities
        self.fail_claims = fail_claims

    def produce_entities(self, example, answer, evidence):
        self.calls.append((example.source_id, "E"))
        if self.fail_entities:
            raise RuntimeError("private entity completion")
        return ({"name": "ENTITY_MARKER", "grounded": False},)

    def produce_claims(self, example, answer, evidence):
        self.calls.append((example.source_id, "C"))
        if self.fail_claims:
            raise LookupError("private claim completion")
        return ({"text": "CLAIM_MARKER", "verdict": "contradicted"},)


class FakeCorrector:
    fingerprint = "corrector-ec-fixture-v1"

    def __init__(self, *, distinct=False):
        self.prompts = []
        self.distinct = distinct

    def generate(self, prompt, *, arm, iteration):
        self.prompts.append((arm, iteration, prompt))
        answer = f"Revised for {arm}." if self.distinct else "Revised answer."
        return Generation(answer, 10, 3, 0.1, "stop")


def _run(tmp_path, examples=(EXAMPLE,), producer=None, corrector=None, **kwargs):
    producer = producer or FakeProducer()
    corrector = corrector or FakeCorrector()
    root, cache = tmp_path / "run", tmp_path / "cache"
    summary = run_arms(examples, producer, corrector, run_dir=root, cache_root=cache,
                       input_manifest_sha256="fixture-input-manifest", **kwargs)
    return root, cache, summary, producer, corrector


def test_four_prompts_are_isolated_and_resume_without_model_calls(tmp_path):
    root, cache, summary, producer, corrector = _run(tmp_path)
    assert ARM_CODES == ("B", "E", "C", "EC")
    assert summary["completed"] == summary["expected_trajectories"] == 4
    assert summary["failed"] == 0
    assert producer.calls == [(EXAMPLE.source_id, "E"), (EXAMPLE.source_id, "C")]
    prompts = {arm: prompt for arm, iteration, prompt in corrector.prompts}
    assert set(prompts) == set(ARM_CODES)
    assert all("Alice lives in London." in prompt and "[C0] Alice lives in Paris." in prompt for prompt in prompts.values())
    assert "ENTITY_MARKER" not in prompts["B"] and "CLAIM_MARKER" not in prompts["B"]
    assert "ENTITY_MARKER" in prompts["E"] and "CLAIM_MARKER" not in prompts["E"]
    assert "CLAIM_MARKER" in prompts["C"] and "ENTITY_MARKER" not in prompts["C"]
    assert "ENTITY_MARKER" in prompts["EC"] and "CLAIM_MARKER" in prompts["EC"]
    assert all(iteration == 1 for _, iteration, _ in corrector.prompts)
    baseline = json.loads((root / "trajectories" / EXAMPLE.source_id / "B.json").read_text())
    assert baseline["steps"][0]["feedback_sha256"] is None
    report = replay(root, cache, [EXAMPLE])
    assert report["expected"] == report["completed"] == 4 and report["missing"] == 0
    assert len(json.loads((root / "evaluation_assignment.json").read_text())) == 4
    assert len(list((root / "evaluation_requests").glob("*.json"))) == 1
    resumed_producer, resumed_corrector = FakeProducer(), FakeCorrector()
    resumed = run_arms([EXAMPLE], resumed_producer, resumed_corrector, run_dir=root,
                       cache_root=cache, input_manifest_sha256="fixture-input-manifest")
    assert resumed["run_fingerprint"] == summary["run_fingerprint"]
    assert resumed["completed"] == 4
    assert resumed_producer.calls == [] and resumed_corrector.prompts == []
    assert replay(root, cache, [EXAMPLE]) == report


@pytest.mark.parametrize(("fail_entities", "fail_claims", "completed_arms"), [
    (True, False, {"B", "C"}),
    (False, True, {"B", "E"}),
    (True, True, {"B"}),
])
def test_component_failures_are_typed_and_independent(tmp_path, fail_entities, fail_claims, completed_arms):
    producer = FakeProducer(fail_entities=fail_entities, fail_claims=fail_claims)
    root, cache, summary, _, corrector = _run(tmp_path, producer=producer)
    actual = {path.stem for path in (root / "trajectories" / EXAMPLE.source_id).glob("*.json")}
    assert actual == completed_arms
    assert {arm for arm, _, _ in corrector.prompts} == completed_arms
    assert summary["completed"] == len(completed_arms)
    assert summary["failed"] == 4 - len(completed_arms)
    assert producer.calls == [(EXAMPLE.source_id, "E"), (EXAMPLE.source_id, "C")]
    failures = {path.stem: json.loads(path.read_text()) for path in (root / "failures" / EXAMPLE.source_id).glob("*.json")}
    assert set(failures) == set(ARM_CODES) - completed_arms
    assert all(value["stage"] == "feedback" for value in failures.values())
    assert all("private" not in json.dumps(value) for value in failures.values())
    assert len(list((root / "failure_history" / EXAMPLE.source_id).rglob("*.json"))) == len(failures)
    if fail_entities and fail_claims:
        assert [item["component"] for item in failures["EC"]["component_failures"]] == ["E", "C"]
    assert replay(root, cache, [EXAMPLE])["missing"] == len(failures)


def test_failed_component_resumes_without_redoing_successful_work(tmp_path):
    root, cache, first, _, _ = _run(tmp_path, producer=FakeProducer(fail_claims=True))
    assert first["completed"] == 2 and first["failed"] == 2
    producer, corrector = FakeProducer(), FakeCorrector()
    second = run_arms([EXAMPLE], producer, corrector, run_dir=root, cache_root=cache,
                      input_manifest_sha256="fixture-input-manifest")
    assert second["completed"] == 4 and second["failed"] == 0
    assert producer.calls == [(EXAMPLE.source_id, "C")]
    assert {arm for arm, _, _ in corrector.prompts} == {"C", "EC"}
    assert not list((root / "failures" / EXAMPLE.source_id).glob("*.json"))
    assert len(list((root / "failure_history" / EXAMPLE.source_id).rglob("*.json"))) == 2
    assert replay(root, cache, [EXAMPLE])["missing"] == 0


def test_blind_export_requires_complete_replay(tmp_path):
    root, cache, _, _, _ = _run(tmp_path, producer=FakeProducer(fail_claims=True))
    assert replay(root, cache, [EXAMPLE])["missing"] == 2
    with pytest.raises(ValueError, match="complete"):
        export_blind_audit(root, tmp_path / "blind")
    assert not (tmp_path / "blind" / "requests.jsonl").exists()


def test_failure_history_keeps_sanitized_cause_chain(tmp_path):
    class RateLimit(Exception):
        status_code = 429

    class WrappedProducer(FakeProducer):
        def produce_claims(self, example, answer, evidence):
            try:
                raise RateLimit("secret token and raw prompt")
            except RateLimit as cause:
                raise RuntimeError("private claim response") from cause

    root, _, _, _, _ = _run(tmp_path, producer=WrappedProducer())
    failure = json.loads((root / "failures" / EXAMPLE.source_id / "C.json").read_text())
    assert failure["component_failures"] == [{
        "component": "C", "error_type": "RuntimeError", "status_code": None,
        "causes": [{"error_type": "RateLimit", "status_code": 429}],
    }]
    assert "secret" not in json.dumps(failure) and "prompt" not in json.dumps(failure)


def test_interruption_keeps_each_completed_arm(tmp_path):
    class InterruptedCorrector(FakeCorrector):
        def generate(self, prompt, *, arm, iteration):
            if len(self.prompts) == 2:
                raise KeyboardInterrupt
            return super().generate(prompt, arm=arm, iteration=iteration)

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, corrector=InterruptedCorrector())
    root = tmp_path / "run"
    assert {path.stem for path in (root / "trajectories" / EXAMPLE.source_id).glob("*.json")} == {"B", "E"}
    producer, corrector = FakeProducer(), FakeCorrector()
    summary = run_arms([EXAMPLE], producer, corrector, run_dir=root,
                       cache_root=tmp_path / "cache", input_manifest_sha256="fixture-input-manifest")
    assert summary["completed"] == 4
    assert producer.calls == []
    assert {arm for arm, _, _ in corrector.prompts} == {"C", "EC"}


def test_replay_cache_check_and_blind_export_all_unique_requests(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "original.json").write_text("{}")
    root, _, _, _, _ = _run(tmp_path, examples=(EXAMPLE, SECOND_EXAMPLE), corrector=FakeCorrector(distinct=True))
    (cache / "new.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        replay(root, cache, [EXAMPLE, SECOND_EXAMPLE])
    (cache / "new.json").unlink()
    assert replay(root, cache, [EXAMPLE, SECOND_EXAMPLE])["missing"] == 0
    packet = tmp_path / "packet"
    report = export_blind_audit(root, packet)
    assert report == {"assignments": 8, "unique_requests": 8, "available_requests": 8, "missing": 0}
    rows = [json.loads(line) for line in (packet / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["request_id"] for row in rows}) == 8
    assert all(set(row) == {"request_id", "context", "query", "original_answer", "revised_answer", "schema_version"} for row in rows)
    assert not (packet / "evaluation_assignment.json").exists()
    assert export_blind_audit(root, packet) == report
    with pytest.raises(ValueError, match="separate"):
        export_blind_audit(root, root / "packet")
    (cache / "original.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="inventory"):
        replay(root, cache, [EXAMPLE, SECOND_EXAMPLE])


def test_renderer_and_identity_reject_protocol_changes(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        visible_feedback(None, "X")
    with pytest.raises(ValueError, match="exactly one"):
        _run(tmp_path, iterations=2)
    root, cache, _, _, _ = _run(tmp_path)
    with pytest.raises(ValueError, match="different protocol"):
        run_arms([SECOND_EXAMPLE], FakeProducer(), FakeCorrector(), run_dir=root,
                 cache_root=cache, input_manifest_sha256="fixture-input-manifest")


def test_resume_and_replay_reject_scientific_code_drift(tmp_path, monkeypatch):
    root, cache, _, _, _ = _run(tmp_path)
    monkeypatch.setattr(runner_module, "science_code_inventory", lambda package_root: {"changed.py": "different"})
    with pytest.raises(ValueError, match="different protocol"):
        run_arms([EXAMPLE], FakeProducer(), FakeCorrector(), run_dir=root,
                 cache_root=cache, input_manifest_sha256="fixture-input-manifest")
    with pytest.raises(ValueError, match="scientific code differs"):
        replay(root, cache, [EXAMPLE])


def test_runtime_config_is_pinned_in_identity_and_replay(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "runtime_config.json").write_text('{"config_sha256":"first","model":"frozen"}\n')
    _, cache, _, _, _ = _run(tmp_path)
    (root / "runtime_config.json").write_text('{"config_sha256":"first","model":"changed"}\n')
    with pytest.raises(ValueError, match="different protocol"):
        run_arms([EXAMPLE], FakeProducer(), FakeCorrector(), run_dir=root,
                 cache_root=cache, input_manifest_sha256="fixture-input-manifest")
    with pytest.raises(ValueError, match="runtime config differs"):
        replay(root, cache, [EXAMPLE])
