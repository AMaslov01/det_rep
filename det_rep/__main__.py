"""Prepare and package the fixed 100-source E/C correction run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .contracts import Example
from .dataset import load_prepared, prepare
from .util import atomic_json, digest, file_digest, read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_NAMESPACE = "ec-veriscore-v1"
SOURCE_SHA256 = "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b"
ANSWER_SHA256 = "46cc5679aabc03a101354d1cc5ab7cf8014a7b92f69e758680247b2cb7578ebc"
SELECTED_SOURCE_IDS_SHA256 = "c0c893de71d2b887984ae44190328599d7e7fc8c21f1f899da390de90ca300a8"


def external(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value == PROJECT_ROOT or PROJECT_ROOT in value.parents:
        raise ValueError("data, work, and run paths must be outside det_rep")
    return value


def validate_pinned_inputs(source_path: Path, answer_path: Path) -> None:
    if file_digest(source_path) != SOURCE_SHA256 or file_digest(answer_path) != ANSWER_SHA256:
        raise ValueError("the E/C run requires the exact QA100 source and answer snapshots")


def pinned_gateway_manifest(
    run_dir: Path, gateway_url: str, model: str, supplied_path: Path | None = None,
) -> dict[str, object]:
    """Keep the frozen Gemini identity available when a gateway outage only blocks E/C."""
    from .gemini import fetch_gateway_manifest, validate_gateway_manifest

    pinned_path = run_dir / "gateway_manifest.json"
    supplied = read_json(supplied_path) if supplied_path is not None else None
    if pinned_path.exists():
        manifest = read_json(pinned_path)
        if supplied is not None and supplied != manifest:
            raise ValueError("run directory has a different gateway revision")
    elif supplied is not None:
        manifest = supplied
    else:
        manifest = fetch_gateway_manifest(gateway_url, model)
    if not isinstance(manifest, dict):
        raise ValueError("gateway manifest must be an object")
    validate_gateway_manifest(manifest, model)
    return manifest


def smoke_selection(examples: tuple[Example, ...], count: int) -> tuple[Example, ...]:
    if count <= 0 or count % 2:
        raise ValueError("balanced smoke size must be a positive even number")
    selected: list[Example] = []
    for label in (0, 1):
        pool = sorted(
            (row for row in examples if row.split == "train" and row.label == label),
            key=lambda row: hashlib.sha256(f"42\0smoke\0{row.source_id}".encode()).hexdigest(),
        )
        if len(pool) < count // 2:
            raise ValueError("not enough annotated training examples for balanced smoke")
        selected.extend(pool[:count // 2])
    return tuple(sorted(selected, key=lambda row: int(row.source_id)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepared = sub.add_parser("prepare")
    prepared.add_argument("--sources", required=True, help="RAGTruth source_info.jsonl")
    prepared.add_argument("--answers", required=True, help="Llama annotation CSV")
    prepared.add_argument("--work-dir", required=True)

    smoke = sub.add_parser("smoke")
    smoke.add_argument("--sources", required=True)
    smoke.add_argument("--answers", required=True)
    smoke.add_argument("--manifest", required=True)
    smoke.add_argument("--work-dir", required=True)
    smoke.add_argument("--run-dir", required=True)
    smoke.add_argument("--config", default=str(PROJECT_ROOT / "config.example.yaml"))
    smoke.add_argument("--gateway-url", default=os.environ.get("HALLU_GATEWAY_URL"))
    smoke.add_argument("--gateway-manifest", help="Optional frozen gateway manifest; permits B when Gemini is unavailable")
    smoke.add_argument("--embedding-path", required=True)
    smoke.add_argument("--vllm-url", default=os.environ.get("DET_REP_VLLM_BASE_URL"))
    smoke.add_argument("--checkpoint", default=os.environ.get("DET_REP_VLLM_CHECKPOINT"))
    smoke.add_argument("--max-sources", type=int, help="Stop after this many selected source IDs; resume in the same run directory")

    replay_cmd = sub.add_parser("replay")
    replay_cmd.add_argument("--sources", required=True)
    replay_cmd.add_argument("--answers", required=True)
    replay_cmd.add_argument("--manifest", required=True)
    replay_cmd.add_argument("--work-dir", required=True)
    replay_cmd.add_argument("--run-dir", required=True)
    audit_cmd = sub.add_parser("audit-export", help="Write a blind audit packet after replay")
    audit_cmd.add_argument("--run-dir", required=True)
    audit_cmd.add_argument("--out-dir", required=True)
    package_cmd = sub.add_parser("package-export", help="Build private provenance and a separate blind handoff")
    package_cmd.add_argument("--run-dir", required=True)
    package_cmd.add_argument("--work-dir", required=True)
    package_cmd.add_argument("--sources", required=True)
    package_cmd.add_argument("--answers", required=True)
    package_cmd.add_argument("--manifest", required=True)
    package_cmd.add_argument("--repo-dir", required=True)
    package_cmd.add_argument("--config", required=True)
    package_cmd.add_argument("--environment-dir", required=True)
    package_cmd.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.command == "audit-export":
        from .runner import export_blind_audit

        report = export_blind_audit(external(args.run_dir), external(args.out_dir))
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "package-export":
        from .archive import package_export

        report = package_export(
            run_dir=external(args.run_dir), work_dir=external(args.work_dir),
            sources=external(args.sources), answers=external(args.answers),
            manifest=external(args.manifest), repo_dir=Path(args.repo_dir).resolve(),
            config_path=Path(args.config).resolve(), environment_dir=external(args.environment_dir),
            out_dir=external(args.out_dir),
        )
        print(json.dumps(report, sort_keys=True))
        return
    source_path = external(args.sources)
    answer_path = external(args.answers)
    validate_pinned_inputs(source_path, answer_path)

    if args.command == "prepare":
        manifest_path, summary = prepare(source_path, answer_path, external(args.work_dir))
        print(json.dumps({"manifest": str(manifest_path), **{key: summary[key] for key in ("sources", "available", "pending", "train_available", "test_available")}}, sort_keys=True))
        return

    work_dir = external(args.work_dir)
    cache_root = work_dir / "cache" / CACHE_NAMESPACE
    run_dir = external(args.run_dir)
    manifest_path = Path(args.manifest).resolve()
    if work_dir not in manifest_path.parents:
        raise ValueError("manifest must belong to the external work directory")
    examples = load_prepared(source_path, answer_path, manifest_path)
    if args.command == "replay":
        from .runner import replay

        report = replay(run_dir, cache_root, examples)
        print(json.dumps(report, sort_keys=True))
        return

    if not args.vllm_url or not args.checkpoint:
        raise ValueError("local vLLM URL and exact checkpoint are required")
    if not args.gateway_url:
        raise ValueError("--gateway-url or HALLU_GATEWAY_URL is required")
    import yaml

    from .gemini import GeminiFeedbackProducer, runtime_config
    from .llm import OpenAICompatibleCorrector
    from .runner import run_arms

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("config must be a YAML mapping")
    selected = smoke_selection(examples, 100)
    if digest([row.source_id for row in selected]) != SELECTED_SOURCE_IDS_SHA256:
        raise ValueError("the selected source IDs differ from the frozen QA100 cohort")
    corrector = OpenAICompatibleCorrector(
        base_url=args.vllm_url, model=base["corrector"]["model"], checkpoint=args.checkpoint,
        cache_dir=cache_root / "generation", seed=int(base["corrector"]["seed"]),
        temperature=float(base["corrector"]["temperature"]),
        max_tokens=int(base["corrector"]["max_tokens"]),
    )
    served_model = corrector.preflight()
    gateway_manifest = pinned_gateway_manifest(
        run_dir, args.gateway_url, base["llm"]["model"],
        Path(args.gateway_manifest).resolve() if args.gateway_manifest else None,
    )
    os.environ["EXPECTED_GATEWAY_MANIFEST_SHA256"] = digest(gateway_manifest)
    cfg = runtime_config(base, gateway_manifest, args.gateway_url, args.embedding_path, cache_root)
    from .core.cache import evaluation_runtime_metadata
    gateway_path = run_dir / "gateway_manifest.json"
    if gateway_path.exists() and json.loads(gateway_path.read_text(encoding="utf-8")) != gateway_manifest:
        raise ValueError("run directory has a different gateway revision")
    atomic_json(gateway_path, gateway_manifest)
    runtime_record = {
        "config_sha256": file_digest(args.config), "gateway_manifest": gateway_manifest,
        "cache_namespace": CACHE_NAMESPACE,
        "embedding_path": str(Path(args.embedding_path).resolve()),
        "embedding_revision": cfg.matching.embedding_model_revision,
        "gemini_runtime": evaluation_runtime_metadata(cfg),
        "corrector": {
            "base_url": args.vllm_url, "model": base["corrector"]["model"],
            "checkpoint": args.checkpoint, "seed": base["corrector"]["seed"],
            "temperature": base["corrector"]["temperature"],
            "max_tokens": base["corrector"]["max_tokens"],
            "served_model": served_model,
        },
    }
    runtime_path = run_dir / "runtime_config.json"
    if runtime_path.exists() and json.loads(runtime_path.read_text(encoding="utf-8")) != runtime_record:
        raise ValueError("run directory has different runtime settings")
    atomic_json(runtime_path, runtime_record)
    producer = GeminiFeedbackProducer(cfg, gateway_manifest, cache_root / "feedback")
    summary = run_arms(
        selected, producer, corrector, run_dir=run_dir, cache_root=cache_root,
        input_manifest_sha256=file_digest(manifest_path), iterations=1,
        max_sources=args.max_sources,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
