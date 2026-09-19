"""Complete RAGTruth QA split and incrementally available Llama answers."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .contracts import Example, SCHEMA_VERSION
from .util import atomic_json, digest, file_digest, read_json, text_digest


EXPECTED_QA_SOURCES = 989
SPLIT_SEED = 42
TEST_SOURCES = 198
REQUIRED_COLUMNS = frozenset({"id", "generated_response", "prompt", "hallucination", "annotation_reason", "annotation_raw", "annotation_model"})
ID_PREFIX = "llama31_8b_"


def qa_sources(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_type") != "QA":
                continue
            source_id = str(row.get("source_id", ""))
            info = row.get("source_info")
            if not source_id.isdigit() or not isinstance(info, dict):
                raise ValueError(f"invalid QA source row {number}")
            if source_id in result:
                raise ValueError(f"duplicate QA source {source_id}")
            result[source_id] = info
    if len(result) != EXPECTED_QA_SOURCES:
        raise ValueError(f"expected {EXPECTED_QA_SOURCES} QA sources, found {len(result)}")
    return result


def source_split(source_ids: set[str]) -> dict[str, str]:
    if len(source_ids) != EXPECTED_QA_SOURCES:
        raise ValueError("source-level split requires all 989 QA sources")
    ordered = sorted(source_ids, key=lambda value: (hashlib.sha256(f"{SPLIT_SEED}\0{value}".encode()).hexdigest(), value))
    held_out = set(ordered[-TEST_SOURCES:])
    return {source_id: ("test" if source_id in held_out else "train") for source_id in source_ids}


def _contained(prompt: str, value: str) -> bool:
    return re.sub(r"\s+", " ", value).strip() in re.sub(r"\s+", " ", prompt).strip()


def llama_answers(path: str | Path, sources: dict[str, dict[str, Any]], splits: dict[str, str]) -> dict[str, Example]:
    answers: dict[str, Example] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"answer CSV missing columns: {sorted(missing)}")
        for number, row in enumerate(reader, 2):
            response_id = str(row.get("id") or "").strip()
            if not response_id.startswith(ID_PREFIX):
                raise ValueError(f"row {number}: invalid response id")
            source_id = response_id.removeprefix(ID_PREFIX)
            if source_id not in sources or source_id in answers:
                raise ValueError(f"row {number}: unknown or duplicate source {source_id}")
            info = sources[source_id]
            context = str(info.get("passages") or "").strip()
            query = str(info.get("question") or "").strip()
            prompt = str(row.get("prompt") or "")
            if not prompt or not _contained(prompt, context) or (query and not _contained(prompt, query)):
                raise ValueError(f"row {number}: prompt does not contain source context/query")
            raw_label = str(row.get("hallucination") or "").strip()
            if raw_label not in {"0", "1"}:
                raise ValueError(f"row {number}: invalid hallucination label")
            annotation_model = str(row.get("annotation_model") or "").strip()
            if not annotation_model:
                raise ValueError(f"row {number}: missing annotation provenance")
            answers[source_id] = Example(
                source_id=source_id, response_id=response_id, context=context,
                query=query, original_answer=str(row.get("generated_response") or "").strip(),
                label=int(raw_label), annotation_model=annotation_model, split=splits[source_id],
            )
    return answers


def prepare(source_path: str | Path, answer_path: str | Path, work_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    sources = qa_sources(source_path)
    splits = source_split(set(sources))
    root = Path(work_dir).resolve()
    split_path = root / "source_split.json"
    split_payload = {
        "schema_version": SCHEMA_VERSION, "seed": SPLIT_SEED,
        "source_jsonl_sha256": file_digest(source_path), "splits": dict(sorted(splits.items())),
    }
    if split_path.exists():
        if read_json(split_path) != split_payload:
            raise ValueError("frozen source split or source snapshot changed")
    else:
        atomic_json(split_path, split_payload)
    answers = llama_answers(answer_path, sources, splits)
    records = [
        {
            "source_id": sid, "split": splits[sid],
            "status": "available" if sid in answers else "pending",
            "response_id": answers[sid].response_id if sid in answers else None,
            "answer_sha256": text_digest(answers[sid].original_answer) if sid in answers else None,
            "label": answers[sid].label if sid in answers else None,
        }
        for sid in sorted(sources, key=int)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION, "split_sha256": file_digest(split_path),
        "answer_csv_sha256": file_digest(answer_path), "source_jsonl_sha256": file_digest(source_path),
        "sources": len(sources), "available": len(answers), "pending": len(sources) - len(answers),
        "train_available": sum(x.split == "train" for x in answers.values()),
        "test_available": sum(x.split == "test" for x in answers.values()),
        "records": records,
    }
    manifest_path = root / f"input_manifest-{digest(payload)[:16]}.json"
    if manifest_path.exists():
        if read_json(manifest_path) != payload:
            raise ValueError("input manifest hash collision or mutation")
    else:
        atomic_json(manifest_path, payload)
    return manifest_path, payload


def load_prepared(source_path: str | Path, answer_path: str | Path, manifest_path: str | Path) -> tuple[Example, ...]:
    payload = read_json(manifest_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("input manifest schema mismatch")
    if payload.get("source_jsonl_sha256") != file_digest(source_path) or payload.get("answer_csv_sha256") != file_digest(answer_path):
        raise ValueError("input files differ from immutable manifest")
    sources = qa_sources(source_path)
    split_path = Path(manifest_path).parent / "source_split.json"
    if payload.get("split_sha256") != file_digest(split_path):
        raise ValueError("source split hash changed")
    splits = read_json(split_path)["splits"]
    if splits != source_split(set(sources)):
        raise ValueError("source split contents changed")
    answers = llama_answers(answer_path, sources, splits)
    if set(answers) != {row["source_id"] for row in payload["records"] if row["status"] == "available"}:
        raise ValueError("response coverage differs from manifest")
    return tuple(sorted(answers.values(), key=lambda x: int(x.source_id)))
