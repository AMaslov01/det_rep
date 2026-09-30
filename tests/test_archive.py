import csv
import json
import sys
from pathlib import Path

import pytest

from det_rep import dataset
from det_rep.archive import export_variant_provenance, package_export
from det_rep.contracts import Example, Generation
from det_rep.core.extract import Graph
from det_rep.dataset import load_prepared, prepare
from det_rep.relations import run_relations
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


class Extractor:
    def extract_reference(self, context, query):
        return Graph(set(), {("source", "has", "context")}), Graph.empty()

    def extract(self, answer, kind="response"):
        return Graph.empty()


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
        for number in range(10_000, 10_002):
            writer.writerow({
                "id": f"llama31_8b_{number}", "generated_response": f"Wrong answer {number}.",
                "prompt": f"Context for {number}. Question for {number}?",
                "hallucination": number % 2, "annotation_reason": "fixture",
                "annotation_raw": "fixture", "annotation_model": "fixture",
            })
    work = tmp_path / "work"
    manifest, _ = prepare(source_path, answer_path, work)
    selected = load_prepared(source_path, answer_path, manifest)
    cache = work / "cache" / dataset.CACHE_NAMESPACE
    cache.mkdir(parents=True)
    (cache / "scientific-cache.json").write_text("{}")
    run = work / "runs" / "ec750"
    relation_dir = work / "runs" / "r750"
    config = tmp_path / "config.yaml"
    config.write_text("fixture: true\n")
    gateway = {"protocol": "fixture", "revision": "gateway-r1"}
    atomic_json(relation_dir / "gateway_manifest.json", gateway)
    atomic_json(relation_dir / "runtime_config.json", {
        "source_sha256": file_digest(source_path), "answer_sha256": file_digest(answer_path),
        "config_sha256": file_digest(config), "gateway_manifest_sha256": digest(gateway),
        "extractor_fingerprint": "fixture", "cache_namespace": dataset.CACHE_NAMESPACE,
        "embedding_revision": "sbert-fixture-revision", "gemini_runtime": {"fixture": True},
    })
    relation_summary = run_relations(
        selected, Extractor(),
        run_dir=relation_dir, kg_cache_root=cache / "kg",
        input_manifest_sha256=file_digest(manifest),
    )
    runtime = {
        "config_sha256": file_digest(config), "gateway_manifest": gateway,
        "embedding_revision": "sbert-fixture-revision",
        "gemini_runtime": {"fixture": True},
        "relation_run_fingerprint": relation_summary["run_fingerprint"],
        "relation_identity_sha256": file_digest(relation_dir / "run_identity.json"),
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
        "run_dir": run, "relation_dir": relation_dir, "work_dir": work,
        "sources": source_path, "answers": answer_path,
        "manifest": manifest, "repo_dir": REPO, "config_path": config,
        "environment_dir": environment, "out_dir": tmp_path / "archive",
    }, summary, selected


def _pin_fixture_inputs(monkeypatch, args, selected):
    monkeypatch.setattr(dataset, "COHORT_SIZE", len(selected))
    monkeypatch.setattr(dataset, "SOURCE_SHA256", file_digest(args["sources"]))
    monkeypatch.setattr(dataset, "ANSWER_SHA256", file_digest(args["answers"]))
    monkeypatch.setattr(dataset, "SELECTED_SOURCE_IDS_SHA256", digest([example.source_id for example in selected]))


def test_complete_archive_has_private_replay_and_all_unique_blind_requests(tmp_path, monkeypatch):
    args, summary, selected = _prepared_run(tmp_path)
    _pin_fixture_inputs(monkeypatch, args, selected)
    assert summary["completed"] == 8 and summary["failed"] == 0
    result = package_export(**args)
    assert result["sources"] == 2 and result["trajectories"] == 8
    assert result["relation_artifacts"] == 2
    assert result["private_variants"] == 8 and result["blind_requests"] == 2
    owner = args["out_dir"] / "owner-private"
    blind = args["out_dir"] / "evaluator-team"
    assert len(list((owner / "run" / "trajectories").rglob("*.json"))) == 8
    assert len(list((owner / "relation-extraction" / "sources").glob("*.json"))) == 2
    assert len(list((owner / "run" / "feedback_components").rglob("*.json"))) == 4
    assert len((owner / "run" / "variant_provenance.jsonl").read_text().splitlines()) == 8
    assert len(json.loads((owner / "run" / "evaluation_assignment.json").read_text())) == 8
    assert (owner / "inputs" / "annotated_answers.csv").read_bytes() == args["answers"].read_bytes()
    assert (owner / "cache" / dataset.CACHE_NAMESPACE / "scientific-cache.json").is_file()
    assert (owner / "code" / "det_rep" / "runner.py").is_file()
    assert not (owner / "code" / "tmp").exists()
    assert {path.name for path in blind.iterdir()} == {"requests.jsonl", "README.md", "request.schema.json", "SHA256SUMS"}
    rows = [json.loads(line) for line in (blind / "requests.jsonl").read_text().splitlines()]
    assert len(rows) == len({row["request_id"] for row in rows}) == 2
    assert all(set(row) == {"request_id", "context", "query", "original_answer", "revised_answer", "schema_version"} for row in rows)
    assert not (blind / "evaluation_assignment.json").exists()
    for line in (blind / "SHA256SUMS").read_text().splitlines():
        sha256, relative = line.split("  ", 1)
        assert file_digest(blind / relative) == sha256
    checksums = (args["out_dir"] / "SHA256SUMS").read_text().splitlines()
    assert len(checksums) > 8
    for line in checksums:
        sha256, relative = line.split("  ", 1)
        assert file_digest(args["out_dir"] / relative) == sha256
    with pytest.raises(ValueError, match="already exists"):
        package_export(**args)


def test_archive_rejects_incomplete_run_and_missing_environment_snapshot(tmp_path, monkeypatch):
    args, summary, selected = _prepared_run(tmp_path, max_sources=1)
    _pin_fixture_inputs(monkeypatch, args, selected)
    assert summary["completed"] == 4
    with pytest.raises(ValueError, match="8/8"):
        package_export(**args)
    run_arms(selected, Producer(), Corrector(), run_dir=args["run_dir"],
             cache_root=args["work_dir"] / "cache" / dataset.CACHE_NAMESPACE,
             input_manifest_sha256=file_digest(args["manifest"]))
    args["environment_dir"].rename(tmp_path / "environment-moved")
    with pytest.raises(ValueError, match="environment snapshot"):
        package_export(**args)


def test_archive_rejects_changed_inputs_and_code_identity(tmp_path, monkeypatch):
    args, _, selected = _prepared_run(tmp_path)
    _pin_fixture_inputs(monkeypatch, args, selected)
    original_answers = args["answers"].read_bytes()
    args["answers"].write_bytes(original_answers + b"\n")
    with pytest.raises(ValueError, match="exact source and answer snapshots"):
        package_export(**args)
    args["answers"].write_bytes(original_answers)
    # A changed identity digest is rejected against the actual code before packaging.
    identity_path = args["run_dir"] / "run_identity.json"
    identity = json.loads(identity_path.read_text())
    identity["science_code_sha256"] = "wrong-code"
    atomic_json(identity_path, identity)
    with pytest.raises(ValueError, match="repository code differs"):
        package_export(**args)


def test_archive_rejects_changed_private_r_triples(tmp_path, monkeypatch):
    args, _, selected = _prepared_run(tmp_path)
    _pin_fixture_inputs(monkeypatch, args, selected)
    path = args["relation_dir"] / "sources" / f"{selected[0].source_id}.json"
    artifact = json.loads(path.read_text())
    artifact["relations"]["context"] = []
    atomic_json(path, artifact)
    with pytest.raises(ValueError, match="relation source inventory"):
        package_export(**args)


def test_verify_r_cli_is_inference_free_and_checks_the_pinned_cohort(tmp_path, monkeypatch, capsys):
    from det_rep.__main__ import main

    args, _, selected = _prepared_run(tmp_path)
    _pin_fixture_inputs(monkeypatch, args, selected)
    monkeypatch.setattr(sys, "argv", [
        "det_rep", "verify-r", "--sources", str(args["sources"]),
        "--answers", str(args["answers"]), "--manifest", str(args["manifest"]),
        "--work-dir", str(args["work_dir"]), "--relation-dir", str(args["relation_dir"]),
    ])
    main()
    assert json.loads(capsys.readouterr().out)["completed_sources"] == 2


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
