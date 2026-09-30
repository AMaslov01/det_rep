"""Paired arm execution, durable trajectories, and inference-free replay."""
from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Iterable

from .arms import PROMPT_VERSION, render_prompt
from .contracts import ARM_CODES, RUN_PROTOCOL, Corrector, EvaluationRequest, Example, FeedbackProducer, FeedbackRecord, Trajectory, TrajectoryStep
from .evidence import build_evidence
from .util import atomic_json, atomic_text, digest, file_digest, read_json, text_digest


def cache_inventory(cache_root: str | Path) -> dict[str, str]:
    root = Path(cache_root)
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    } if root.exists() else {}


def science_code_inventory(package_root: str | Path) -> dict[str, str]:
    root = Path(package_root)
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("*.py")) if path.is_file() and "__pycache__" not in path.parts
    }


def validate_feedback(record: FeedbackRecord, example: Example, answer: str, evidence_sha256: str, producer_fingerprint: str) -> None:
    if (record.source_id != example.source_id or record.answer_sha256 != text_digest(answer)
        or record.evidence_sha256 != evidence_sha256 or record.producer_fingerprint != producer_fingerprint):
        raise ValueError("producer returned cross-source or stale feedback")


def _component_path(root: Path, source_id: str, component: str) -> Path:
    return root / "feedback_components" / source_id / f"{component}.json"


def _component_metadata(example: Example, answer: str, evidence_sha256: str, fingerprint: str, component: str) -> dict[str, str]:
    return {
        "schema_version": "det-rep-feedback-component-v1", "component": component,
        "source_id": example.source_id, "answer_sha256": text_digest(answer),
        "evidence_sha256": evidence_sha256, "producer_fingerprint": fingerprint,
    }


def _prepare_component(
    root: Path, producer: FeedbackProducer, example: Example, answer: str,
    evidence: Any, evidence_sha256: str, component: str,
) -> tuple[dict[str, Any], ...]:
    """Keep successful E and C work separately so a failed channel can be retried."""
    metadata = _component_metadata(example, answer, evidence_sha256, producer.fingerprint, component)
    path = _component_path(root, example.source_id, component)
    if path.exists():
        saved = read_json(path)
        if not isinstance(saved, dict) or {key: saved.get(key) for key in metadata} != metadata:
            raise ValueError(f"{component} feedback component identity mismatch")
        value = saved.get("value")
    else:
        method = producer.produce_entities if component == "E" else producer.produce_claims
        value = method(example, answer, evidence)
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{component} producer returned invalid feedback")
        value = list(value)
        atomic_json(path, {**metadata, "value": value})
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{component} feedback component is invalid")
    return tuple(value)


def _feedback_record(
    example: Example, evidence_sha256: str, producer_fingerprint: str,
    components: dict[str, tuple[dict[str, Any], ...]],
) -> FeedbackRecord:
    return FeedbackRecord(
        source_id=example.source_id, answer_sha256=text_digest(example.original_answer),
        evidence_sha256=evidence_sha256, producer_fingerprint=producer_fingerprint,
        entities=components.get("E", ()), claims=components.get("C", ()),
        entity_status="ok" if "E" in components else "failed",
        claim_status="ok" if "C" in components else "failed",
    )


def _exception_causes(exc: Exception) -> list[dict[str, Any]]:
    """Retain retry diagnostics without recording exception messages or model text."""
    causes: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = exc.__cause__
    while current is not None and len(causes) < 4 and id(current) not in seen:
        seen.add(id(current))
        causes.append({
            "error_type": type(current).__name__,
            "status_code": _status_code(current),
        })
        current = current.__cause__
    return causes


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    return value if type(value) is int else None


def _write_failure(root: Path, fingerprint: str, example: Example, arm: str, stage: str,
                   failures: list[tuple[str, Exception]]) -> None:
    diagnostic = {
        "protocol": "det-rep-failure-v2", "run_fingerprint": fingerprint,
        "source_id": example.source_id, "response_id": example.response_id,
        "arm": arm, "iteration": 1, "stage": stage,
        "component_failures": [
            {"component": component, "error_type": type(exc).__name__,
             "status_code": _status_code(exc),
             "causes": _exception_causes(exc)}
            for component, exc in failures
        ],
        "error_type": type(failures[0][1]).__name__,
        "status_code": _status_code(failures[0][1]),
    }
    atomic_json(root / "failures" / example.source_id / f"{arm}.json", diagnostic)
    atomic_json(root / "failure_history" / example.source_id / arm / f"{secrets.token_hex(16)}.json", diagnostic)


def run_arms(
    examples: Iterable[Example], producer: FeedbackProducer, corrector: Corrector,
    *, run_dir: str | Path, cache_root: str | Path, input_manifest_sha256: str,
    iterations: int = 1, max_sources: int | None = None,
) -> dict[str, Any]:
    if iterations != 1:
        raise ValueError("the E/C protocol has exactly one revision iteration")
    selected = tuple(examples)
    if not selected:
        raise ValueError("correction runner requires available answers")
    if len({example.source_id for example in selected}) != len(selected):
        raise ValueError("one original answer per source is required")
    if max_sources is not None and not 1 <= max_sources <= len(selected):
        raise ValueError("max_sources must be between 1 and the selected source count")
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime_path = root / "runtime_config.json"
    runtime = read_json(runtime_path) if runtime_path.exists() else None
    runtime_sha = file_digest(runtime_path) if runtime is not None else "unversioned-test-runtime"
    config_sha = runtime.get("config_sha256") if isinstance(runtime, dict) else "unversioned-test-config"
    if not isinstance(config_sha, str) or not config_sha:
        raise ValueError("runtime config must pin config_sha256")
    identity = {
        "protocol": RUN_PROTOCOL, "input_manifest_sha256": input_manifest_sha256,
        "source_ids": [x.source_id for x in selected], "iterations": iterations,
        "arms": list(ARM_CODES), "prompt_version": PROMPT_VERSION,
        "runner_code_sha256": file_digest(Path(__file__)),
        "renderer_code_sha256": file_digest(Path(__file__).with_name("arms.py")),
        "science_code_sha256": digest(science_code_inventory(Path(__file__).resolve().parent)),
        "runtime_config_sha256": runtime_sha, "config_sha256": config_sha,
        "producer_fingerprint": getattr(producer, "fingerprint", "unversioned-test-producer"),
        "corrector_fingerprint": getattr(corrector, "fingerprint", "unversioned-test-corrector"),
    }
    identity_path = root / "run_identity.json"
    if identity_path.exists():
        recorded_identity = read_json(identity_path)
        salt = recorded_identity.get("blind_salt")
        if not isinstance(salt, str) or len(salt) != 64:
            raise ValueError("run directory has no valid blinding salt")
        identity["blind_salt"] = salt
        if recorded_identity != identity:
            raise ValueError("run directory belongs to a different protocol/input/model")
    else:
        identity["blind_salt"] = secrets.token_hex(32)
    fingerprint = digest(identity)
    atomic_json(identity_path, identity)

    def checkpoint(source_id: str) -> dict[str, Any]:
        """Commit progress after each source; all arm files are saved earlier."""
        inventory = cache_inventory(cache_root)
        trajectory_inventory = {
            str(path.relative_to(root / "trajectories")): file_digest(path)
            for path in sorted((root / "trajectories").rglob("*.json"))
        }
        completed = sum(
            (root / "trajectories" / example.source_id / f"{arm}.json").is_file()
            for example in selected for arm in ARM_CODES
        )
        failed = sum(
            (root / "failures" / example.source_id / f"{arm}.json").is_file()
            and not (root / "trajectories" / example.source_id / f"{arm}.json").is_file()
            for example in selected for arm in ARM_CODES
        )
        summary = {
            "run_fingerprint": fingerprint, "sources": len(selected),
            "expected_trajectories": len(selected) * len(ARM_CODES),
            "completed": completed, "failed": failed,
            "last_checkpoint_source_id": source_id,
            "cache_inventory_sha256": digest(inventory),
            "trajectory_inventory_sha256": digest(trajectory_inventory),
        }
        atomic_json(root / "cache_inventory.json", inventory)
        atomic_json(root / "run_summary.json", summary)
        return summary

    summary: dict[str, Any] = {}
    for example in selected[:max_sources]:
        evidence = build_evidence(example)
        evidence_sha256 = digest(evidence.to_dict())
        evidence_path = root / "evidence" / f"{example.source_id}.json"
        if evidence_path.exists() and digest(read_json(evidence_path)) != evidence_sha256:
            raise ValueError("existing evidence differs from pinned source")
        atomic_json(evidence_path, evidence.to_dict())
        remaining: list[str] = []
        for arm in ARM_CODES:
            path = root / "trajectories" / example.source_id / f"{arm}.json"
            if path.exists():
                existing = read_json(path)
                if existing.get("run_fingerprint") != fingerprint or len(existing.get("steps", [])) != iterations:
                    raise ValueError("trajectory identity mismatch")
            else:
                remaining.append(arm)
        if not remaining:
            summary = checkpoint(example.source_id)
            continue

        # B has no Gemini dependency. Persist it before preparing either channel.
        if "B" in remaining:
            prompt = render_prompt(example, example.original_answer, evidence, None, "B")
            try:
                generation = corrector.generate(prompt, arm="B", iteration=1)
                if not generation.answer.strip():
                    raise ValueError("empty corrected answer")
                trajectory = Trajectory(
                    example.source_id, example.response_id, "B", example.split, example.original_answer,
                    (TrajectoryStep(1, example.original_answer, None, text_digest(prompt), generation),),
                    fingerprint,
                )
                atomic_json(root / "trajectories" / example.source_id / "B.json", trajectory.to_dict())
                (root / "failures" / example.source_id / "B.json").unlink(missing_ok=True)
            except Exception as exc:
                _write_failure(root, fingerprint, example, "B", "generation", [("generation", exc)])

        needed = [arm for arm in remaining if arm != "B"]
        if not needed:
            summary = checkpoint(example.source_id)
            continue
        components: dict[str, tuple[dict[str, Any], ...]] = {}
        component_errors: dict[str, Exception] = {}
        for component in ("E", "C"):
            try:
                components[component] = _prepare_component(
                    root, producer, example, example.original_answer, evidence,
                    evidence_sha256, component,
                )
            except Exception as exc:
                component_errors[component] = exc
        feedback = _feedback_record(example, evidence_sha256, identity["producer_fingerprint"], components)
        validate_feedback(feedback, example, example.original_answer, evidence_sha256, identity["producer_fingerprint"])
        feedback_sha = digest(feedback.to_dict())
        atomic_json(root / "feedback" / example.source_id / f"{feedback_sha}.json", feedback.to_dict())
        for arm in remaining:
            if arm == "B":
                continue
            unavailable = [(component, component_errors[component]) for component in ("E", "C")
                           if component in arm and component in component_errors]
            if unavailable:
                _write_failure(root, fingerprint, example, arm, "feedback", unavailable)
                continue
            path = root / "trajectories" / example.source_id / f"{arm}.json"
            try:
                prompt = render_prompt(example, example.original_answer, evidence, feedback, arm)
                generation = corrector.generate(prompt, arm=arm, iteration=1)
                if not generation.answer.strip():
                    raise ValueError("empty corrected answer")
                trajectory = Trajectory(
                    example.source_id, example.response_id, arm, example.split,
                    example.original_answer,
                    (TrajectoryStep(1, example.original_answer, feedback_sha, text_digest(prompt), generation),),
                    fingerprint,
                )
                atomic_json(path, trajectory.to_dict())
                (root / "failures" / example.source_id / f"{arm}.json").unlink(missing_ok=True)
            except Exception as exc:
                # No prompts, source text, completions, or credentials in diagnostics.
                _write_failure(root, fingerprint, example, arm, "generation", [("generation", exc)])
        summary = checkpoint(example.source_id)
    return summary


def replay(run_dir: str | Path, cache_root: str | Path, examples: Iterable[Example]) -> dict[str, Any]:
    """Verify saved artifacts and write audit inputs without model calls."""
    root = Path(run_dir)
    identity = read_json(root / "run_identity.json")
    if identity.get("protocol") != RUN_PROTOCOL or identity.get("arms") != list(ARM_CODES) or identity.get("iterations") != 1:
        raise ValueError("replay requires the fixed one-iteration E/C protocol")
    if identity.get("science_code_sha256") != digest(science_code_inventory(Path(__file__).resolve().parent)):
        raise ValueError("scientific code differs from pinned run identity")
    runtime_path = root / "runtime_config.json"
    if identity.get("runtime_config_sha256") != "unversioned-test-runtime":
        if identity["runtime_config_sha256"] != file_digest(runtime_path):
            raise ValueError("runtime config differs from pinned run identity")
        if identity.get("config_sha256") != read_json(runtime_path).get("config_sha256"):
            raise ValueError("config hash differs from pinned run identity")
    fingerprint = digest(identity)
    expected = {(source_id, arm) for source_id in identity["source_ids"] for arm in identity["arms"]}
    found: set[tuple[str, str]] = set()
    by_source = {example.source_id: example for example in examples}
    if set(identity["source_ids"]) - set(by_source):
        raise ValueError("replay requires every source in the pinned run")
    assignment: list[dict[str, str]] = []
    requests: dict[str, dict[str, Any]] = {}
    for source_id, arm in sorted(expected):
        path = root / "trajectories" / source_id / f"{arm}.json"
        if not path.exists():
            continue
        value = read_json(path)
        if value.get("run_fingerprint") != fingerprint or value.get("arm") != arm or value.get("source_id") != source_id:
            raise ValueError("trajectory identity mismatch in replay")
        if len(value.get("steps", [])) != identity["iterations"]:
            raise ValueError("incomplete trajectory in replay")
        evidence = read_json(root / "evidence" / f"{source_id}.json")
        example = by_source[source_id]
        rebuilt = build_evidence(example)
        if digest(evidence) != digest(rebuilt.to_dict()):
            raise ValueError("evidence pack differs from pinned source")
        if value["original_answer"] != example.original_answer:
            raise ValueError("original answer differs from pinned input")
        current = example.original_answer
        for number, step in enumerate(value["steps"], 1):
            if step["iteration"] != number or step["input_answer"] != current:
                raise ValueError("iteration state differs in replay")
            if arm == "B":
                if step["feedback_sha256"] is not None:
                    raise ValueError("baseline trajectory unexpectedly depends on feedback")
                record = None
            else:
                if not isinstance(step["feedback_sha256"], str):
                    raise ValueError("feedback hash missing in replay")
                feedback = read_json(root / "feedback" / source_id / f"{step['feedback_sha256']}.json")
                if digest(feedback) != step["feedback_sha256"]:
                    raise ValueError("feedback hash differs in replay")
                record = FeedbackRecord.from_dict(feedback)
                validate_feedback(record, example, current, digest(rebuilt.to_dict()), identity["producer_fingerprint"])
                for component, status, items in (("E", record.entity_status, record.entities),
                                                  ("C", record.claim_status, record.claims)):
                    if status == "ok":
                        saved = read_json(_component_path(root, source_id, component))
                        metadata = _component_metadata(example, current, digest(rebuilt.to_dict()), identity["producer_fingerprint"], component)
                        if {key: saved.get(key) for key in metadata} != metadata or saved.get("value") != list(items):
                            raise ValueError("feedback component differs in replay")
            prompt = render_prompt(example, current, rebuilt, record, arm)
            if text_digest(prompt) != step["prompt_sha256"]:
                raise ValueError("prompt hash differs in replay")
            current = step["generation"]["answer"]
        public_payload = {
            "context": example.context, "query": example.query,
            "original_answer": example.original_answer,
            "revised_answer": value["steps"][-1]["generation"]["answer"],
        }
        request_id = digest({"blind_salt": identity["blind_salt"], "request": public_payload})
        request = EvaluationRequest(
            request_id, **public_payload,
        )
        if request_id in requests and requests[request_id] != request.__dict__:
            raise ValueError("evaluation request identity collision")
        requests[request_id] = request.__dict__
        assignment.append({"request_id": request_id, "source_id": source_id, "arm": arm})
        found.add((source_id, arm))
    recorded = read_json(root / "cache_inventory.json")
    current_inventory = cache_inventory(cache_root)
    if current_inventory != recorded:
        raise ValueError("recorded cache inventory changed after live run")
    summary = read_json(root / "run_summary.json")
    trajectory_inventory = {
        str(path.relative_to(root / "trajectories")): file_digest(path)
        for path in sorted((root / "trajectories").rglob("*.json"))
    }
    if (summary["run_fingerprint"] != fingerprint or summary["completed"] != len(found)
        or summary["cache_inventory_sha256"] != digest(recorded)
        or summary["trajectory_inventory_sha256"] != digest(trajectory_inventory)):
        raise ValueError("run summary differs from replayed trajectories")
    report = {
        "run_fingerprint": fingerprint, "expected": len(expected), "completed": len(found),
        "missing": len(expected - found), "cache_inventory_sha256": digest(recorded),
        "assignments_sha256": digest(assignment), "requests_sha256": digest(requests),
    }
    for request_id, request in requests.items():
        request_path = root / "evaluation_requests" / f"{request_id}.json"
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError("evaluation request changed after replay")
        atomic_json(request_path, request)
    atomic_json(root / "replay_summary.json", report)
    atomic_json(root / "evaluation_assignment.json", assignment)
    return report


def export_blind_audit(run_dir: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Export every unique request; the assignment stays in the run directory."""
    root = Path(run_dir).resolve()
    target = Path(out_dir).resolve()
    if target == root or target in root.parents or root in target.parents:
        raise ValueError("blind audit output must be separate from the private run directory")
    identity = read_json(root / "run_identity.json")
    report = read_json(root / "replay_summary.json")
    assignment = read_json(root / "evaluation_assignment.json")
    if report.get("missing") != 0 or report.get("completed") != report.get("expected"):
        raise ValueError("blind export requires a complete replay")
    if (root / "failures").exists() and any((root / "failures").rglob("*.json")):
        raise ValueError("blind export requires zero unresolved failures")
    if report["assignments_sha256"] != digest(assignment) or report["completed"] != len(assignment):
        raise ValueError("replay assignment differs from the audit report")
    request_ids = {item["request_id"] for item in assignment}
    public_fields = {"request_id", "context", "query", "original_answer", "revised_answer", "schema_version"}
    requests: list[dict[str, Any]] = []
    for request_id in request_ids:
        request = read_json(root / "evaluation_requests" / f"{request_id}.json")
        if set(request) != public_fields or request["request_id"] != request_id:
            raise ValueError("evaluation request contains unexpected fields")
        requests.append(request)
    if digest({request["request_id"]: request for request in requests}) != report["requests_sha256"]:
        raise ValueError("evaluation requests differ from the replay report")
    requests.sort(key=lambda item: digest({"salt": identity["blind_salt"], "request_id": item["request_id"]}))
    available = len(requests)
    lines = "".join(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n" for request in requests)
    packet_path = target / "requests.jsonl"
    if target.exists() and any(path.name != packet_path.name for path in target.iterdir()):
        raise ValueError("blind audit output directory contains unexpected files")
    if packet_path.exists() and packet_path.read_text(encoding="utf-8") != lines:
        raise ValueError("blind audit packet already exists with different contents")
    atomic_text(packet_path, lines)
    exported_ids = {request["request_id"] for request in requests}
    return {
        "assignments": sum(item["request_id"] in exported_ids for item in assignment),
        "unique_requests": len(requests), "available_requests": available,
        "missing": report["missing"],
    }
