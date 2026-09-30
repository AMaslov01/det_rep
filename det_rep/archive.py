"""Verify and package a completed E/C correction run for a blind handoff."""
from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dataset
from .contracts import ARM_CODES, RUN_PROTOCOL
from .dataset import load_prepared
from .relations import verify_relations
from .runner import export_blind_audit, replay, science_code_inventory
from .util import atomic_json, atomic_text, digest, file_digest, read_json


REQUIRED_ENVIRONMENT_FILES = (
    "container-image.json", "model-manifest.json", "python-packages.txt", "os-packages.txt",
)
CODE_FILES = (
    ".gcloudignore", ".gitignore", ".dockerignore", "AGENTS.md", "Dockerfile",
    "Dockerfile.experiment", "README.md", "config.example.yaml", "pyproject.toml",
    "requirements-gateway.txt", "gemini_recipient.py",
)
CODE_DIRS = ("det_rep", "gemini_gateway", "scripts", "docs", "tests")
SKIP_NAMES = {".DS_Store", "__pycache__", ".pytest_cache", ".ruff_cache"}
PRIVATE_NAMES = {".env", "config.local.yaml", "gateway-key", "hf-token", "credentials.json"}


def _copy_tree(source: Path, target: Path, *, code: bool = False) -> None:
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"archive source is not a regular directory: {source}")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in SKIP_NAMES or part.startswith(".env") for part in relative.parts):
            continue
        if any(part in PRIVATE_NAMES or part.endswith(".key") for part in relative.parts):
            raise ValueError(f"private credential-like path in archive source: {relative}")
        if path.is_symlink():
            raise ValueError(f"archive source contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"archive source contains non-file: {path}")
        if code and path.suffix in {".pyc", ".csv", ".jsonl"}:
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _require_nonempty(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ValueError(f"required archive snapshot is absent or empty: {path}")


def _read_environment(environment_dir: Path, runtime: dict[str, Any], gateway_manifest: dict[str, Any]) -> None:
    if not environment_dir.is_dir():
        raise ValueError("environment snapshot directory is required")
    for name in REQUIRED_ENVIRONMENT_FILES:
        _require_nonempty(environment_dir / name)
    image = read_json(environment_dir / "container-image.json")
    models = read_json(environment_dir / "model-manifest.json")
    if not isinstance(image, dict) or any(not isinstance(image.get(key), str) or not image[key]
                                           for key in ("experiment_image_id", "vllm_image_digest")):
        raise ValueError("container-image.json must pin experiment_image_id and vllm_image_digest")
    expected = {
        "llama_checkpoint": runtime["corrector"]["checkpoint"],
        "embedding_revision": runtime["embedding_revision"],
        "gateway_manifest_sha256": digest(gateway_manifest),
    }
    if not isinstance(models, dict) or any(models.get(key) != value for key, value in expected.items()):
        raise ValueError("model-manifest.json differs from pinned runtime models")


def export_variant_provenance(run_dir: str | Path, output: str | Path,
                              *, require_complete: bool = True) -> dict[str, int]:
    """One private row per source and arm, including duplicate revisions."""
    root, destination = Path(run_dir), Path(output)
    identity = read_json(root / "run_identity.json")
    report = read_json(root / "replay_summary.json")
    assignments = read_json(root / "evaluation_assignment.json")
    if identity.get("protocol") != RUN_PROTOCOL or identity.get("arms") != list(ARM_CODES) or identity.get("iterations") != 1:
        raise ValueError("private provenance requires the fixed E/C protocol")
    if not isinstance(assignments, list) or report.get("assignments_sha256") != digest(assignments):
        raise ValueError("replay assignment hash differs")
    if report.get("completed") != len(assignments):
        raise ValueError("replay assignment count differs from completed trajectories")
    if require_complete and (report.get("missing") != 0 or report.get("expected") != len(assignments)):
        raise ValueError("run is incomplete")

    pairs: set[tuple[str, str]] = set()
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in assignments:
        pair = (item["source_id"], item["arm"])
        if item["source_id"] not in identity["source_ids"] or item["arm"] not in ARM_CODES or pair in pairs:
            raise ValueError("duplicate or unpinned source/arm assignment")
        pairs.add(pair)
        groups[(item["source_id"], item["request_id"])].append(item["arm"])
    records: list[dict[str, Any]] = []
    for item in sorted(assignments, key=lambda row: (int(row["source_id"]), ARM_CODES.index(row["arm"]))):
        source_id, arm, request_id = item["source_id"], item["arm"], item["request_id"]
        trajectory = read_json(root / "trajectories" / source_id / f"{arm}.json")
        request = read_json(root / "evaluation_requests" / f"{request_id}.json")
        if trajectory.get("source_id") != source_id or trajectory.get("arm") != arm or request.get("request_id") != request_id:
            raise ValueError("trajectory or request differs from assignment")
        steps = trajectory.get("steps")
        if not isinstance(steps, list) or len(steps) != 1 or steps[0]["generation"]["answer"] != request["revised_answer"]:
            raise ValueError("revision differs from blinded request")
        final = steps[0]
        records.append({
            "schema_version": "det-rep-ec-variant-provenance-v1",
            "source_id": source_id, "response_id": trajectory["response_id"],
            "arm": arm,
            "feedback_blocks": [name for letter, name in (("E", "entities"), ("C", "claims")) if letter in arm],
            "request_id": request_id,
            "same_revision_arms": sorted(groups[(source_id, request_id)], key=ARM_CODES.index),
            "prompt_sha256": final["prompt_sha256"],
            "feedback_sha256": final["feedback_sha256"],
            "revised_answer": final["generation"]["answer"],
        })
    lines = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    if destination.exists() and destination.read_text(encoding="utf-8") != lines:
        raise ValueError("existing provenance differs from replay")
    atomic_text(destination, lines)
    unique_requests = len({record["request_id"] for record in records})
    return {
        "variants": len(records), "unique_requests": unique_requests,
        "duplicate_variants": len(records) - unique_requests,
    }


def package_export(
    *, run_dir: str | Path, relation_dir: str | Path, work_dir: str | Path, sources: str | Path,
    answers: str | Path, manifest: str | Path, repo_dir: str | Path,
    config_path: str | Path, environment_dir: str | Path, out_dir: str | Path,
) -> dict[str, Any]:
    """Create a complete private archive and a separate all-unique blind packet."""
    root, relations, work, source_path, answer_path = (Path(value).resolve() for value in (run_dir, relation_dir, work_dir, sources, answers))
    manifest_path, repository, config = (Path(value).resolve() for value in (manifest, repo_dir, config_path))
    environment, destination = (Path(value).resolve() for value in (environment_dir, out_dir))
    cache_root = work / "cache" / dataset.CACHE_NAMESPACE
    expected_trajectories = dataset.COHORT_SIZE * len(ARM_CODES)
    if destination.exists():
        raise ValueError("archive output already exists")
    if any(destination == item or destination.is_relative_to(item) or item.is_relative_to(destination)
           for item in (root, relations, cache_root, repository, environment)):
        raise ValueError("archive output must be separate from run, cache, code and environment sources")
    if not manifest_path.is_relative_to(work):
        raise ValueError("input manifest must belong to the work directory")
    for path in (source_path, answer_path, manifest_path, work / "source_split.json", config,
                 root / "run_identity.json", root / "runtime_config.json", root / "gateway_manifest.json",
                 root / "run_summary.json", relations / "gateway_manifest.json",
                 relations / "run_identity.json",
                 relations / "runtime_config.json", relations / "run_summary.json"):
        _require_nonempty(path)
    if not cache_root.is_dir():
        raise ValueError("the isolated scientific cache is missing")
    for name in CODE_FILES:
        _require_nonempty(repository / name)
    for name in CODE_DIRS:
        path = repository / name
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"code snapshot directory is missing or unsafe: {path}")
    dataset.validate_pinned_inputs(source_path, answer_path)

    identity, runtime = read_json(root / "run_identity.json"), read_json(root / "runtime_config.json")
    gateway_manifest = read_json(root / "gateway_manifest.json")
    manifest_data, summary = read_json(manifest_path), read_json(root / "run_summary.json")
    if identity.get("protocol") != RUN_PROTOCOL or identity.get("arms") != list(ARM_CODES) or identity.get("iterations") != 1:
        raise ValueError("archive requires the fixed one-iteration E/C protocol")
    if len(identity.get("source_ids", [])) != dataset.COHORT_SIZE or len(set(identity["source_ids"])) != dataset.COHORT_SIZE:
        raise ValueError("archive requires exactly 750 distinct source IDs")
    if identity.get("input_manifest_sha256") != file_digest(manifest_path):
        raise ValueError("run input manifest hash differs")
    if manifest_data.get("source_jsonl_sha256") != file_digest(source_path) or manifest_data.get("answer_csv_sha256") != file_digest(answer_path):
        raise ValueError("input files differ from manifest")
    if manifest_data.get("split_sha256") != file_digest(work / "source_split.json"):
        raise ValueError("source split differs from manifest")
    if (identity.get("runner_code_sha256") != file_digest(repository / "det_rep" / "runner.py")
        or identity.get("renderer_code_sha256") != file_digest(repository / "det_rep" / "arms.py")
        or identity.get("science_code_sha256") != digest(science_code_inventory(repository / "det_rep"))):
        raise ValueError("repository code differs from run identity")
    if (runtime.get("config_sha256") != file_digest(config)
        or identity.get("config_sha256") != file_digest(config)
        or identity.get("runtime_config_sha256") != file_digest(root / "runtime_config.json")
        or runtime.get("gateway_manifest") != gateway_manifest):
        raise ValueError("config or gateway differs from pinned runtime")
    if (summary.get("sources") != dataset.COHORT_SIZE or summary.get("expected_trajectories") != expected_trajectories
        or summary.get("completed") != expected_trajectories or summary.get("failed") != 0):
        raise ValueError(f"archive requires {expected_trajectories}/{expected_trajectories} completed trajectories and zero final failures")
    if summary.get("run_fingerprint") != digest(identity):
        raise ValueError("run summary differs from identity")
    if (root / "failures").exists() and any((root / "failures").rglob("*.json")):
        raise ValueError("archive contains unresolved failure files")
    _read_environment(environment, runtime, gateway_manifest)
    examples = load_prepared(source_path, answer_path, manifest_path)
    selected = dataset.selected_answers(examples)
    if [example.source_id for example in selected] != identity["source_ids"]:
        raise ValueError("selected source IDs differ from fixed 750-answer cohort")
    if digest(identity["source_ids"]) != dataset.SELECTED_SOURCE_IDS_SHA256:
        raise ValueError("selected source IDs differ from frozen 750-answer cohort")
    relation_summary = verify_relations(
        relations, cache_root / "kg", selected,
        input_manifest_sha256=file_digest(manifest_path),
    )
    relation_runtime = read_json(relations / "runtime_config.json")
    if (runtime.get("relation_run_fingerprint") != relation_summary["run_fingerprint"]
        or runtime.get("relation_identity_sha256") != file_digest(relations / "run_identity.json")
        or read_json(relations / "gateway_manifest.json") != gateway_manifest
        or relation_runtime.get("cache_namespace") != dataset.CACHE_NAMESPACE
        or relation_runtime.get("source_sha256") != file_digest(source_path)
        or relation_runtime.get("answer_sha256") != file_digest(answer_path)
        or relation_runtime.get("config_sha256") != file_digest(config)
        or relation_runtime.get("gateway_manifest_sha256") != digest(gateway_manifest)
        or relation_runtime.get("embedding_revision") != runtime.get("embedding_revision")
        or relation_runtime.get("gemini_runtime") != runtime.get("gemini_runtime")):
        raise ValueError("R sweep differs from correction runtime")
    replay_report = replay(root, cache_root, selected)
    if (replay_report["missing"] != 0 or replay_report["completed"] != expected_trajectories
        or replay_report["expected"] != expected_trajectories):
        raise ValueError(f"archive requires replay.missing=0 for {expected_trajectories} assignments")
    provenance = export_variant_provenance(root, root / "variant_provenance.jsonl", require_complete=True)
    if provenance["variants"] != expected_trajectories:
        raise ValueError(f"private provenance must contain {expected_trajectories} rows")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        owner = staging / "owner-private"
        blind = staging / "evaluator-team"
        (owner / "inputs").mkdir(parents=True)
        shutil.copy2(source_path, owner / "inputs" / "source_info.jsonl")
        shutil.copy2(answer_path, owner / "inputs" / "annotated_answers.csv")
        (owner / "work").mkdir()
        shutil.copy2(manifest_path, owner / "work" / manifest_path.name)
        shutil.copy2(work / "source_split.json", owner / "work" / "source_split.json")
        (owner / "config").mkdir()
        shutil.copy2(config, owner / "config" / "run_config.yaml")
        _copy_tree(root, owner / "run")
        _copy_tree(relations, owner / "relation-extraction")
        _copy_tree(cache_root, owner / "cache" / dataset.CACHE_NAMESPACE)
        _copy_tree(environment, owner / "environment")
        code_root = owner / "code"
        code_root.mkdir()
        for name in CODE_FILES:
            original = repository / name
            shutil.copy2(original, code_root / name)
        for name in CODE_DIRS:
            original = repository / name
            _copy_tree(original, code_root / name, code=True)
        if digest(science_code_inventory(code_root / "det_rep")) != identity["science_code_sha256"]:
            raise ValueError("archived code differs from run identity")
        code_inventory = {
            str(path.relative_to(code_root)): file_digest(path)
            for path in sorted(code_root.rglob("*")) if path.is_file()
        }
        atomic_json(owner / "code_inventory.json", code_inventory)
        atomic_json(owner / "archive_manifest.json", {
            "protocol": "det-rep-ec-r-archive-v1", "source_count": dataset.COHORT_SIZE,
            "relation_artifacts": dataset.COHORT_SIZE,
            "trajectories": expected_trajectories, "unique_requests": provenance["unique_requests"],
            "run_fingerprint": digest(identity), "input_manifest_sha256": file_digest(manifest_path),
            "cache_namespace": dataset.CACHE_NAMESPACE,
        })
        atomic_text(owner / "README.md", (
            "# Private correction archive\n\n"
            "Keep this directory with the experiment owner. It contains exact inputs, code, runtime and model snapshots, "
            "scientific caches, evidence, feedback, trajectories, failure history, replay, and the private treatment map. "
            "Only the sibling evaluator-team directory is suitable for a blind handoff.\n"
        ))
        blind_report = export_blind_audit(root, blind)
        if blind_report["assignments"] != expected_trajectories or blind_report["unique_requests"] != provenance["unique_requests"]:
            raise ValueError("blind packet differs from private provenance")
        atomic_text(blind / "README.md", (
            "# Blind correction requests\n\n"
            "Each JSONL row is a unique correction request. Use `request_id` as the opaque key. "
            "`context` and `query` are the supplied evidence and question; `original_answer` and "
            "`revised_answer` are the answer pair. The packet contains every unique pair from the completed run. "
            "Return one result per `request_id`. Treatment assignments and feedback are held by the owner.\n"
        ))
        atomic_json(blind / "request.schema.json", {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object", "additionalProperties": False,
            "required": ["request_id", "context", "query", "original_answer", "revised_answer", "schema_version"],
            "properties": {name: {"type": "string"} for name in
                           ("request_id", "context", "query", "original_answer", "revised_answer", "schema_version")},
        })
        blind_checksums = "".join(
            f"{file_digest(path)}  {path.relative_to(blind)}\n"
            for path in sorted(blind.rglob("*")) if path.is_file()
        )
        atomic_text(blind / "SHA256SUMS", blind_checksums)
        checksums = "".join(
            f"{file_digest(path)}  {path.relative_to(staging)}\n"
            for path in sorted(staging.rglob("*")) if path.is_file()
        )
        atomic_text(staging / "SHA256SUMS", checksums)
        for path in staging.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)
        staging.rename(destination)
    return {
        "sources": dataset.COHORT_SIZE, "relation_artifacts": dataset.COHORT_SIZE,
        "trajectories": expected_trajectories, "final_failures": 0,
        "blind_requests": blind_report["unique_requests"],
        "private_variants": provenance["variants"], "archive": str(destination),
    }
