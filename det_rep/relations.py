"""One resumable KGGen relation sweep over the correction cohort."""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Iterable

from .contracts import Example
from .runner import cache_inventory, science_code_inventory
from .util import atomic_json, digest, file_digest, read_json, text_digest


RELATION_PROTOCOL = "det-rep-r-qa750-v1"
RELATION_SOURCE_SCHEMA = "det-rep-r-source-v1"
RUNTIME_KEYS = (
    "source_sha256", "answer_sha256", "config_sha256", "gateway_manifest_sha256",
    "extractor_fingerprint", "cache_namespace",
)


def _indexed(examples: Iterable[Example]) -> dict[str, Example]:
    rows = tuple(examples)
    if not rows or any(not isinstance(row, Example) or not row.source_id.isdigit() for row in rows):
        raise ValueError("R sweep requires validated answers with numeric source IDs")
    indexed = {row.source_id: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("R sweep requires one answer per source ID")
    return indexed


def _identity(
    root: Path, indexed: dict[str, Example], input_manifest_sha256: str,
) -> dict[str, Any]:
    runtime_path = root / "runtime_config.json"
    runtime = read_json(runtime_path)
    if not isinstance(runtime, dict) or any(not isinstance(runtime.get(key), str) or not runtime[key]
                                            for key in RUNTIME_KEYS):
        raise ValueError("R sweep runtime config is incomplete")
    return {
        "protocol": RELATION_PROTOCOL,
        "source_ids": sorted(indexed, key=int),
        "input_manifest_sha256": input_manifest_sha256,
        "runtime_config_sha256": file_digest(runtime_path),
        "science_code_sha256": digest(science_code_inventory(Path(__file__).resolve().parent)),
        **{key: runtime[key] for key in RUNTIME_KEYS},
    }


def _triples(graph: Any) -> list[list[str]]:
    relations = getattr(graph, "relations", None)
    if not isinstance(relations, set):
        raise ValueError("KG extractor returned no relation set")
    if any(not isinstance(row, tuple) or len(row) != 3
           or any(not isinstance(value, str) for value in row) for row in relations):
        raise ValueError("KG extractor returned malformed relation triples")
    return [list(row) for row in sorted(relations)]


def _artifact(
    example: Example,
    g_context: Any, g_query: Any, g_answer: Any, fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": RELATION_SOURCE_SCHEMA, "run_fingerprint": fingerprint,
        "source_id": example.source_id, "response_id": example.response_id,
        "context_sha256": text_digest(example.context),
        "query_sha256": text_digest(example.query),
        "answer_sha256": text_digest(example.original_answer),
        "relations": {
            "context": _triples(g_context), "query": _triples(g_query),
            "answer": _triples(g_answer),
        },
    }


def _validate_artifact(
    value: Any, example: Example, fingerprint: str,
) -> None:
    source_id = example.source_id
    if not isinstance(value, dict) or value.get("schema_version") != RELATION_SOURCE_SCHEMA:
        raise ValueError(f"relation artifact {source_id} has the wrong schema")
    expected = {
        "run_fingerprint": fingerprint, "source_id": source_id,
        "response_id": example.response_id,
        "context_sha256": text_digest(example.context),
        "query_sha256": text_digest(example.query),
        "answer_sha256": text_digest(example.original_answer),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError(f"relation artifact {source_id} differs from pinned input")
    relations = value.get("relations")
    if not isinstance(relations, dict) or set(relations) != {"context", "query", "answer"}:
        raise ValueError(f"relation artifact {source_id} has incomplete triples")
    for kind in ("context", "query", "answer"):
        rows = relations[kind]
        if not isinstance(rows, list) or any(not isinstance(row, list) or len(row) != 3
                                             or any(not isinstance(value, str) for value in row) for row in rows):
            raise ValueError(f"relation artifact {source_id} has malformed {kind} triples")
        if rows != [list(row) for row in sorted({tuple(row) for row in rows})]:
            raise ValueError(f"relation artifact {source_id} has noncanonical {kind} triples")


def _source_inventory(root: Path) -> dict[str, str]:
    directory = root / "sources"
    return {path.name: file_digest(path) for path in sorted(directory.glob("*.json"))} if directory.exists() else {}


def _checkpoint(root: Path, cache_root: Path, identity: dict[str, Any], source_id: str) -> dict[str, Any]:
    inventory = _source_inventory(root)
    cache = cache_inventory(cache_root)
    complete = len(inventory)
    failed = sum(
        (root / "failures" / f"{source_id}.json").is_file()
        and f"{source_id}.json" not in inventory
        for source_id in identity["source_ids"]
    )
    summary = {
        "run_fingerprint": digest(identity), "expected_sources": len(identity["source_ids"]),
        "expected_answer_graphs": len(identity["source_ids"]),
        "completed_sources": complete, "answer_graphs": complete, "failed": failed,
        "last_checkpoint_source_id": source_id,
        "source_inventory_sha256": digest(inventory), "kg_cache_inventory_sha256": digest(cache),
    }
    atomic_json(root / "source_inventory.json", inventory)
    atomic_json(root / "kg_cache_inventory.json", cache)
    atomic_json(root / "run_summary.json", summary)
    return summary


def run_relations(
    examples: Iterable[Example], extractor: Any,
    *, run_dir: str | Path, kg_cache_root: str | Path, input_manifest_sha256: str,
    max_sources: int | None = None,
) -> dict[str, Any]:
    """Extract triples once for every selected context, question, and answer."""
    indexed = _indexed(examples)
    root, cache_root = Path(run_dir).resolve(), Path(kg_cache_root).resolve()
    expected = _identity(root, indexed, input_manifest_sha256)
    identity_path = root / "run_identity.json"
    if identity_path.exists() and read_json(identity_path) != expected:
        raise ValueError("R sweep directory belongs to different inputs or code")
    current_sources = _source_inventory(root)
    recorded_sources = read_json(root / "source_inventory.json") if (root / "source_inventory.json").exists() else {}
    if not isinstance(recorded_sources, dict) or any(current_sources.get(name) != sha for name, sha in recorded_sources.items()):
        raise ValueError("relation source inventory differs from checkpoint")
    if set(current_sources) - {f"{sid}.json" for sid in expected["source_ids"]}:
        raise ValueError("R sweep contains unpinned source artifacts")
    current_cache = cache_inventory(cache_root)
    recorded_cache = read_json(root / "kg_cache_inventory.json") if (root / "kg_cache_inventory.json").exists() else {}
    if not isinstance(recorded_cache, dict) or any(current_cache.get(name) != sha for name, sha in recorded_cache.items()):
        raise ValueError("relation KG cache inventory differs from checkpoint")
    atomic_json(identity_path, expected)
    ordered = expected["source_ids"]
    if max_sources is not None and not 1 <= max_sources <= len(ordered):
        raise ValueError("max_sources must fit the R sweep source count")
    fingerprint = digest(expected)
    summary: dict[str, Any] = read_json(root / "run_summary.json") if (root / "run_summary.json").exists() else {}
    needs_checkpoint = current_sources != recorded_sources or current_cache != recorded_cache
    for source_id in ordered[:max_sources]:
        path = root / "sources" / f"{source_id}.json"
        example = indexed[source_id]
        if path.exists():
            _validate_artifact(read_json(path), example, fingerprint)
            failure_path = root / "failures" / f"{source_id}.json"
            if failure_path.exists():
                failure_path.unlink()
                needs_checkpoint = True
            continue
        try:
            g_context, g_query = extractor.extract_reference(example.context, example.query)
            g_answer = extractor.extract(example.original_answer, kind="response")
            artifact = _artifact(example, g_context, g_query, g_answer, fingerprint)
            atomic_json(path, artifact)
            (root / "failures" / f"{source_id}.json").unlink(missing_ok=True)
        except Exception as exc:
            diagnostic = {
                "protocol": "det-rep-r-failure-v1", "run_fingerprint": fingerprint,
                "source_id": source_id, "error_type": type(exc).__name__,
            }
            atomic_json(root / "failures" / f"{source_id}.json", diagnostic)
            atomic_json(root / "failure_history" / source_id / f"{secrets.token_hex(16)}.json", diagnostic)
        summary = _checkpoint(root, cache_root, expected, source_id)
        needs_checkpoint = False
    if needs_checkpoint or not summary:
        summary = _checkpoint(root, cache_root, expected, ordered[:max_sources][-1])
    return summary


def verify_relations(
    run_dir: str | Path, kg_cache_root: str | Path,
    examples: Iterable[Example],
    *, input_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate completed private R artifacts without calling a model."""
    indexed = _indexed(examples)
    root, cache_root = Path(run_dir).resolve(), Path(kg_cache_root).resolve()
    identity = read_json(root / "run_identity.json")
    if identity != _identity(root, indexed, input_manifest_sha256):
        raise ValueError("R sweep identity differs from pinned inputs or code")
    inventory = _source_inventory(root)
    if inventory != read_json(root / "source_inventory.json"):
        raise ValueError("relation source inventory differs from checkpoint")
    if set(inventory) != {f"{sid}.json" for sid in identity["source_ids"]}:
        raise ValueError("R sweep source coverage is incomplete")
    current_cache = cache_inventory(cache_root)
    if current_cache != read_json(root / "kg_cache_inventory.json"):
        raise ValueError("relation KG cache inventory differs from checkpoint")
    summary = read_json(root / "run_summary.json")
    if (summary.get("run_fingerprint") != digest(identity)
        or summary.get("expected_sources") != len(indexed)
        or summary.get("expected_answer_graphs") != len(indexed)
        or summary.get("source_inventory_sha256") != digest(inventory)
        or summary.get("kg_cache_inventory_sha256") != digest(current_cache)):
        raise ValueError("relation summary differs from checkpoint")
    if (summary.get("completed_sources") != len(indexed) or len(inventory) != len(indexed)
        or summary.get("answer_graphs") != len(indexed)
        or summary.get("failed") != 0
        or (root / "failures").exists() and any((root / "failures").glob("*.json"))):
        raise ValueError("R sweep is incomplete")
    fingerprint = digest(identity)
    for source_id in identity["source_ids"]:
        _validate_artifact(read_json(root / "sources" / f"{source_id}.json"), indexed[source_id], fingerprint)
    return summary
