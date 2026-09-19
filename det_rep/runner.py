"""Paired arm execution, durable trajectories, and inference-free replay."""
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Iterable

from .arms import PROMPT_VERSION, render_prompt
from .contracts import ARM_CODES, Corrector, EvaluationRequest, Example, FeedbackProducer, FeedbackRecord, Trajectory, TrajectoryStep
from .evidence import build_evidence
from .util import atomic_json, digest, file_digest, read_json, text_digest


def cache_inventory(cache_root: str | Path) -> dict[str, str]:
    root = Path(cache_root)
    return {
        str(path.relative_to(root)): file_digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    } if root.exists() else {}


def validate_feedback(record: FeedbackRecord, example: Example, answer: str, evidence_sha256: str, producer_fingerprint: str) -> None:
    if (record.source_id != example.source_id or record.answer_sha256 != text_digest(answer)
        or record.evidence_sha256 != evidence_sha256 or record.producer_fingerprint != producer_fingerprint):
        raise ValueError("producer returned cross-source or stale feedback")


def run_arms(
    examples: Iterable[Example], producer: FeedbackProducer, corrector: Corrector,
    *, run_dir: str | Path, cache_root: str | Path, input_manifest_sha256: str,
    iterations: int = 1,
) -> dict[str, Any]:
    if iterations not in {1, 2, 3}:
        raise ValueError("revision iterations must be 1, 2, or 3")
    selected = tuple(examples)
    if not selected or any(example.split != "train" for example in selected):
        raise ValueError("v1 smoke runner accepts available training examples only")
    if len({example.source_id for example in selected}) != len(selected):
        raise ValueError("one original answer per source is required")
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "protocol": "det-rep-trajectories-v1", "input_manifest_sha256": input_manifest_sha256,
        "source_ids": [x.source_id for x in selected], "iterations": iterations,
        "arms": list(ARM_CODES), "prompt_version": PROMPT_VERSION,
        "runner_code_sha256": file_digest(Path(__file__)),
        "renderer_code_sha256": file_digest(Path(__file__).with_name("arms.py")),
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
    completed = 0
    failed = 0
    for example in selected:
        evidence = build_evidence(example)
        evidence_sha256 = digest(evidence.to_dict())
        atomic_json(root / "evidence" / f"{example.source_id}.json", evidence.to_dict())
        remaining: list[str] = []
        for arm in ARM_CODES:
            path = root / "trajectories" / example.source_id / f"{arm}.json"
            if path.exists():
                existing = read_json(path)
                if existing.get("run_fingerprint") != fingerprint or len(existing.get("steps", [])) != iterations:
                    raise ValueError("trajectory identity mismatch")
                completed += 1
            else:
                remaining.append(arm)
        if not remaining:
            continue
        try:
            initial_feedback = producer.produce(example, example.original_answer, evidence)
            validate_feedback(initial_feedback, example, example.original_answer, evidence_sha256, identity["producer_fingerprint"])
        except Exception as exc:
            for arm in remaining:
                atomic_json(root / "failures" / example.source_id / f"{arm}.json", {
                    "protocol": "det-rep-failure-v1", "run_fingerprint": fingerprint,
                    "source_id": example.source_id, "response_id": example.response_id,
                    "arm": arm, "iteration": 1, "stage": "feedback",
                    "error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
                })
            failed += len(remaining)
            continue
        for arm in remaining:
            path = root / "trajectories" / example.source_id / f"{arm}.json"
            current_answer = example.original_answer
            steps: list[TrajectoryStep] = []
            stage = "feedback"
            iteration = 0
            try:
                for iteration in range(1, iterations + 1):
                    stage = "feedback"
                    feedback = initial_feedback if iteration == 1 else producer.produce(example, current_answer, evidence)
                    validate_feedback(feedback, example, current_answer, evidence_sha256, identity["producer_fingerprint"])
                    feedback_sha = digest(feedback.to_dict())
                    atomic_json(root / "feedback" / example.source_id / f"{feedback_sha}.json", feedback.to_dict())
                    prompt = render_prompt(example, current_answer, evidence, feedback, arm)
                    stage = "generation"
                    generation = corrector.generate(prompt, arm=arm, iteration=iteration)
                    if not generation.answer.strip():
                        raise ValueError("empty corrected answer")
                    steps.append(TrajectoryStep(iteration, current_answer, feedback_sha, text_digest(prompt), generation))
                    current_answer = generation.answer
                trajectory = Trajectory(
                    example.source_id, example.response_id, arm, example.split,
                    example.original_answer, tuple(steps), fingerprint,
                )
                atomic_json(path, trajectory.to_dict())
                failure = root / "failures" / example.source_id / f"{arm}.json"
                if failure.exists():
                    failure.unlink()
                completed += 1
            except Exception as exc:
                # No prompts, source text, completions, or credentials in diagnostics.
                atomic_json(root / "failures" / example.source_id / f"{arm}.json", {
                    "protocol": "det-rep-failure-v1", "run_fingerprint": fingerprint,
                    "source_id": example.source_id, "response_id": example.response_id,
                    "arm": arm, "iteration": iteration, "stage": stage,
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                })
                failed += 1
    inventory = cache_inventory(cache_root)
    trajectory_inventory = {
        str(path.relative_to(root / "trajectories")): file_digest(path)
        for path in sorted((root / "trajectories").rglob("*.json"))
    }
    summary = {
        "run_fingerprint": fingerprint, "sources": len(selected), "expected_trajectories": len(selected) * len(ARM_CODES),
        "completed": completed, "failed": failed, "cache_inventory_sha256": digest(inventory),
        "trajectory_inventory_sha256": digest(trajectory_inventory),
    }
    atomic_json(root / "run_summary.json", summary)
    atomic_json(root / "cache_inventory.json", inventory)
    return summary


def replay(run_dir: str | Path, cache_root: str | Path, examples: Iterable[Example]) -> dict[str, Any]:
    """Read artifacts only. Never creates a feedback or generation client."""
    root = Path(run_dir)
    identity = read_json(root / "run_identity.json")
    fingerprint = digest(identity)
    expected = {(source_id, arm) for source_id in identity["source_ids"] for arm in ARM_CODES}
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
            feedback = read_json(root / "feedback" / source_id / f"{step['feedback_sha256']}.json")
            if digest(feedback) != step["feedback_sha256"]:
                raise ValueError("feedback hash differs in replay")
            record = FeedbackRecord.from_dict(feedback)
            validate_feedback(record, example, current, digest(rebuilt.to_dict()), identity["producer_fingerprint"])
            prompt = render_prompt(example, current, rebuilt, record, arm)
            if text_digest(prompt) != step["prompt_sha256"]:
                raise ValueError("prompt hash differs in replay")
            current = step["generation"]["answer"]
        request_id = digest({
            "blind_salt": identity["blind_salt"], "source_id": source_id,
            "original": example.original_answer,
            "revised": value["steps"][-1]["generation"]["answer"],
            "evidence": evidence["source_sha256"],
        })
        request = EvaluationRequest(
            request_id, example.context, example.query, example.original_answer,
            value["steps"][-1]["generation"]["answer"],
        )
        if request_id in requests and requests[request_id] != request.__dict__:
            raise ValueError("evaluation request identity collision")
        requests[request_id] = request.__dict__
        assignment.append({"request_id": request_id, "source_id": source_id, "arm": arm})
        found.add((source_id, arm))
    recorded = read_json(root / "cache_inventory.json")
    if recorded != cache_inventory(cache_root):
        raise ValueError("cache inventory changed after live run")
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
        "assignments_sha256": digest(assignment),
    }
    for request_id, request in requests.items():
        request_path = root / "evaluation_requests" / f"{request_id}.json"
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError("evaluation request changed after replay")
        atomic_json(request_path, request)
    atomic_json(root / "replay_summary.json", report)
    atomic_json(root / "evaluation_assignment.json", assignment)
    return report
