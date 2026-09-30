import json

import pytest

from det_rep.contracts import Example
from det_rep.core.extract import Graph
from det_rep.relations import run_relations, verify_relations
from det_rep.util import atomic_json


SOURCES = {
    "1": {"passages": "Paris is in France.", "question": "Where is Paris?"},
    "2": {"passages": "Rome is in Italy.", "question": "Where is Rome?"},
    "3": {"passages": "Oslo is in Norway.", "question": "Where is Oslo?"},
}
ANSWERS = {
    sid: Example(sid, f"llama31_8b_{sid}", SOURCES[sid]["passages"],
                 SOURCES[sid]["question"], answer, 0, "fixture", "train")
    for sid, answer in (("1", "Paris is in Germany."), ("2", "Rome is in Spain."),
                        ("3", "Oslo is in Sweden."))
}


class FakeKGExtractor:
    def __init__(self):
        self.calls = []

    def extract_reference(self, context, query):
        self.calls.extend(("context", "query"))
        place = context.split()[0]
        return Graph({place}, {(place, "located in", context.split()[-1].rstrip("."))}), Graph.empty()

    def extract(self, answer, kind="response"):
        self.calls.append(kind)
        place = answer.split()[0]
        return Graph({place}, {(place, "located in", answer.split()[-1].rstrip("."))})


def _run(tmp_path, extractor, *, max_sources=None):
    run_dir = tmp_path / "relations"
    atomic_json(run_dir / "runtime_config.json", {
        "source_sha256": "source-fixture", "answer_sha256": "answer-fixture",
        "config_sha256": "config-fixture", "gateway_manifest_sha256": "gateway-fixture",
        "extractor_fingerprint": "extractor-fixture", "cache_namespace": "ec-veriscore-r-v1",
    })
    return run_relations(
        ANSWERS.values(), extractor, run_dir=run_dir, kg_cache_root=tmp_path / "cache" / "kg",
        input_manifest_sha256="manifest-fixture", max_sources=max_sources,
    )


def test_relation_sweep_covers_every_answer_and_resumes_without_extraction(tmp_path):
    extractor = FakeKGExtractor()
    first = _run(tmp_path, extractor)
    assert first["expected_sources"] == first["completed_sources"] == 3
    assert first["answer_graphs"] == 3
    assert first["failed"] == 0 and len(extractor.calls) == 9
    artifact = json.loads((tmp_path / "relations" / "sources" / "1.json").read_text())
    assert artifact["relations"] == {
        "context": [["Paris", "located in", "France"]],
        "query": [], "answer": [["Paris", "located in", "Germany"]],
    }
    assert verify_relations(tmp_path / "relations", tmp_path / "cache" / "kg", ANSWERS.values(),
                            input_manifest_sha256="manifest-fixture")["completed_sources"] == 3

    _run(tmp_path, extractor)
    assert len(extractor.calls) == 9
    assert len(list((tmp_path / "relations" / "sources").glob("*.json"))) == 3
    artifact["relations"]["answer"] = []
    (tmp_path / "relations" / "sources" / "1.json").write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="relation"):
        verify_relations(tmp_path / "relations", tmp_path / "cache" / "kg", ANSWERS.values(),
                         input_manifest_sha256="manifest-fixture")


def test_incomplete_sweep_cannot_gate_correction(tmp_path):
    _run(tmp_path, FakeKGExtractor(), max_sources=1)
    with pytest.raises(ValueError, match="incomplete"):
        verify_relations(tmp_path / "relations", tmp_path / "cache" / "kg", ANSWERS.values(),
                         input_manifest_sha256="manifest-fixture")


def test_relation_sweep_rejects_duplicate_source_ids(tmp_path):
    with pytest.raises(ValueError, match="one answer per source ID"):
        run_relations((ANSWERS["1"], ANSWERS["1"]), FakeKGExtractor(),
                      run_dir=tmp_path / "relations", kg_cache_root=tmp_path / "cache",
                      input_manifest_sha256="fixture")


def test_failed_relation_source_is_typed_and_resumable(tmp_path):
    class FailsOnce(FakeKGExtractor):
        def __init__(self):
            super().__init__()
            self.failed = False

        def extract(self, answer, kind="response"):
            if answer.startswith("Rome") and not self.failed:
                self.failed = True
                raise RuntimeError("private answer must not enter failure log")
            return super().extract(answer, kind)

    extractor = FailsOnce()
    first = _run(tmp_path, extractor)
    assert first["completed_sources"] == 2 and first["failed"] == 1
    failure = json.loads((tmp_path / "relations" / "failures" / "2.json").read_text())
    assert failure["error_type"] == "RuntimeError"
    assert "private answer" not in json.dumps(failure)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        verify_relations(tmp_path / "relations", tmp_path / "cache" / "kg", ANSWERS.values(),
                         input_manifest_sha256="manifest-fixture")
    second = _run(tmp_path, extractor)
    assert second["completed_sources"] == 3 and second["failed"] == 0
    assert not (tmp_path / "relations" / "failures" / "2.json").exists()
    assert verify_relations(tmp_path / "relations", tmp_path / "cache" / "kg", ANSWERS.values(),
                            input_manifest_sha256="manifest-fixture")["completed_sources"] == 3


def test_resume_adopts_valid_artifact_written_before_checkpoint(tmp_path):
    extractor = FakeKGExtractor()
    _run(tmp_path, extractor, max_sources=1)
    root = tmp_path / "relations"
    old_inventory = (root / "source_inventory.json").read_bytes()
    old_summary = (root / "run_summary.json").read_bytes()
    _run(tmp_path, extractor, max_sources=2)
    assert len(extractor.calls) == 6
    (root / "source_inventory.json").write_bytes(old_inventory)
    (root / "run_summary.json").write_bytes(old_summary)
    resumed = _run(tmp_path, extractor)
    assert resumed["completed_sources"] == 3
    assert len(extractor.calls) == 9
    assert verify_relations(root, tmp_path / "cache" / "kg", ANSWERS.values(),
                            input_manifest_sha256="manifest-fixture")["failed"] == 0
