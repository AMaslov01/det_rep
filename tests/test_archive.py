import csv
import json
from pathlib import Path

import pytest

from det_rep import __main__ as cli
from det_rep.__main__ import smoke_selection
from det_rep.archive import export_variant_provenance, package_export
from det_rep.contracts import Example, Generation
from det_rep.dataset import load_prepared, prepare
from det_rep.runner import replay, run_arms
from det_rep.util import atomic_json, digest, file_digest


REPO = Path(__file__).resolve().parents[1]


class Producer:
    fingerprint = "archive-fixture-producer"

    def produce_entities(self, example, answer, evidence):
        return ({"name": "entity", "grounded": True},)

    def produce_claims(self, example, answer, evidence):
        return ({"text": "claim", "verdict": "entailed"},)


class Corrector:
    fingerprint = "archive-fixture-corrector"

    def generate(self, prompt, *, arm, iteration):
        return Generation("Same correction.", 10, 3, 0.1, "stop")


def _prepared_run(tmp_path, *, max_sources=None):
    source_path = tmp_path / "source_info.jsonl"
    answer_path = tmp_path / "answers.csv"
    with source_path.open("w") as stream:
        for number in range(10_000, 10_989):
            stream.write(json.dumps({
                "task_type": "QA", "source_id": number,
                "source_info": {"passages": f"Context for {number}.", "question": f"Question for {number}?"},
            }) + "\n")
    with answer_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "id", "generated_response", "prompt", "hallucination",
            "annotation_reason", "annotation_raw", "annotation_model",
        ])
        writer.writeheader()
        for number in range(10_000, 10_989):
            writer.writerow({
                "id": f"llama31_8b_{number}", "generated_response": f"Wrong answer {number}.",
                "prompt": f"Context for {number}. Question for {number}?",
                "hallucination": number % 2, "annotation_reason": "fixture",
                "annotation_raw": "fixture", "annotation_model": "fixture",
            })
    work = tmp_path / "work"
    manifest, _ = prepare(source_path, answer_path, work)
    selected = smoke_selection(load_prepared(source_path, answer_path, manifest), 100)
    cache = work / "cache" / "ec-veriscore-v1"
    cache.mkdir(parents=True)
    (cache / "scientific-cache.json").write_text("{}")
    run = work / "runs" / "ec100"
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n")
    gateway = {"protocol": "fixture", "revision": "gateway-r1"}
    runtime = {
        "config_sha256": file_digest(config), "gateway_manifest": gateway,
        "embedding_revision": "sbert-fixture-revision",
        "corrector": {"checkpoint": "llama-fixture-commit"},
    }
    atomic_json(run / "runtime_config.json", runtime)
    atomic_json(run / "gateway_manifest.json", gateway)
    summary = run_arms(
        selected, Producer(), Corrector(), run_dir=run, cache_root=cache,
        input_manifest_sha256=file_digest(manifest), max_sources=max_sources,
    )
    environment = tmp_path / "environment"
    environment.mkdir()
    atomic_json(environment / "container-image.json", {
        "experiment_image_id": "sha256:experiment-fixture", "vllm_image_digest": "sha256:vllm-fixture",
    })
    atomic_json(environment / "model-manifest.json", {
        "llama_checkpoint": runtime["corrector"]["checkpoint"],
        "embedding_revision": runtime["embedding_revision"],
        "gateway_manifest_sha256": digest(gateway),
    })
    (environment / "python-packages.txt").write_text("pytest==9.1.1\n")
    (environment / "os-packages.txt").write_text("python3.12\n")
    return {
        "run_dir": run, "work_dir": work, "sources": source_path, "answers": answer_path,
        "manifest": manifest, "repo_dir": REPO, "config_path": config,
        "environment_dir": environment, "out_dir": tmp_path / "archive",
    }, summary, selected


def _pin_fixture_inputs(monkeypatch, args, selected):
    monkeypatch.setattr(cli, "SOURCE_SHA256", file_digest(args["sources"]))
    monkeypatch.setattr(cli, "ANSWER_SHA256", file_digest(args["answers"]))
    monkeypatch.setattr(cli, "SELECTED_SOURCE_IDS_SHA256", digest([example.source_id for example in selected]))


def test_complete_archive_has_private_replay_and_all_unique_blind_requests(tmp_path, monkeypatch):
    args, summary, selected = _prepared_run(tmp_path)
    _pin_fixture_inputs(monkeypatch, args, selected)
    assert summary["completed"] == 400 and summary["failed"] == 0
    result = package_export(**args)
    assert result["sources"] == 100 and result["trajectories"] == 400
    assert result["private_variants"] == 400 and result["blind_requests"] == 100
    owner = args["out_dir"] / "owner-private"
    blind = args["out_dir"] / "evaluator-team"
    assert len(list((owner / "run" / "trajectories").rglob("*.json"))) == 400
    assert len(list((owner / "run" / "feedback_components").rglob("*.json"))) == 200
    assert len((owner / "run" / "variant_provenance.jsonl").read_text().splitlines()) == 400
    assert len(json.loads((owner / "run" / "evaluation_assignment.json").read_text())) == 400
    assert (owner / "inputs" / "annotated_answers.csv").read_bytes() == args["answers"].read_bytes()
    assert (owner / "cache" / "ec-veriscore-v1" / "scientific-cache.json").is_file()
    assert (owner / "code" / "det_rep" / "runner.py").is_file()
    assert not (owner / "code" / "tmp").exists()
    assert {path.name for path in blind.iterdir()} == {"requests.jsonl", "README.md", "request.schema.json", "SHA256SUMS"}
    rows = [json.loads(line) for line in (blind / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["request_id"] for row in rows}) == 100
    assert all(set(row) == {"request_id", "context", "query", "original_answer", "revised_answer", "schema_version"} for row in rows)
    assert not (blind / "evaluation_assignment.json").exists()
    for line in (blind / "SHA256SUMS").read_text().splitlines():
        sha256, relative = line.split("  ", 1)
        assert file_digest(blind / relative) == sha256
    checksums = (args["out_dir"] / "SHA256SUMS").read_text().splitlines()
    assert len(checksums) > 400
    for line in checksums:
        sha256, relative = line.split("  ", 1)
        assert file_digest(args["out_dir"] / relative) == sha256
    with pytest.raises(ValueError, match="already exists"):
        package_export(**args)


def test_archive_rejects_incomplete_run_and_missing_environment_snapshot(tmp_path, monkeypatch):
    args, summary, selected = _prepared_run(tmp_path, max_sources=1)
    _pin_fixture_inputs(monkeypatch, args, selected)
    assert summary["completed"] == 4
    with pytest.raises(ValueError, match="400/400"):
        package_export(**args)
    run_arms(selected, Producer(), Corrector(), run_dir=args["run_dir"],
             cache_root=args["work_dir"] / "cache" / "ec-veriscore-v1",
             input_manifest_sha256=file_digest(args["manifest"]))
    args["environment_dir"].rename(tmp_path / "environment-moved")
    with pytest.raises(ValueError, match="environment snapshot"):
        package_export(**args)


def test_archive_rejects_changed_inputs_and_code_identity(tmp_path, monkeypatch):
    args, _, selected = _prepared_run(tmp_path)
    _pin_fixture_inputs(monkeypatch, args, selected)
    original_answers = args["answers"].read_bytes()
    args["answers"].write_bytes(original_answers + b"\n")
    with pytest.raises(ValueError, match="exact QA100"):
        package_export(**args)
    args["answers"].write_bytes(original_answers)
    # A changed identity digest is rejected against the actual code before packaging.
    identity_path = args["run_dir"] / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["science_code_sha256"] = "wrong-code"
    atomic_json(identity_path, identity)
    with pytest.raises(ValueError, match="repository code differs"):
        package_export(**args)


def test_blind_request_identity_deduplicates_identical_public_payload_across_sources(tmp_path):
    first = Example("101", "llama31_8b_101", "Shared context.", "Shared query?", "Same wrong answer.", 0, "fixture", "train")
    second = Example("102", "llama31_8b_102", "Shared context.", "Shared query?", "Same wrong answer.", 1, "fixture", "train")
    root, cache = tmp_path / "run", tmp_path / "cache"
    run_arms([first, second], Producer(), Corrector(), run_dir=root, cache_root=cache, input_manifest_sha256="fixture")
    assert replay(root, cache, [first, second])["missing"] == 0
    assignments = json.loads((root / "evaluation_assignment.json").read_text())
    assert len(assignments) == 8 and len({item["request_id"] for item in assignments}) == 1
    private = export_variant_provenance(root, root / "variant_provenance.jsonl")
    assert private == {"variants": 8, "unique_requests": 1, "duplicate_variants": 7}
    assert len(list((root / "evaluation_requests").glob("*.json"))) == 1
