"""Prepare R extraction and the fixed 750-answer E/C correction run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import dataset
from .dataset import load_prepared, prepare
from .util import atomic_json, digest, file_digest, read_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def external(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value == PROJECT_ROOT or PROJECT_ROOT in value.parents:
        raise ValueError("data, work, and run paths must be outside det_rep")
    return value


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepared = sub.add_parser("prepare")
    prepared.add_argument("--sources", required=True, help="RAGTruth source_info.jsonl")
    prepared.add_argument("--answers", required=True, help="Llama annotation CSV")
    prepared.add_argument("--work-dir", required=True)

    for command in ("extract-r", "run"):
        stage = sub.add_parser(command)
        stage.add_argument("--sources", required=True)
        stage.add_argument("--answers", required=True)
        stage.add_argument("--manifest", required=True)
        stage.add_argument("--work-dir", required=True)
        stage.add_argument("--relation-dir", required=True)
        stage.add_argument("--config", default=str(PROJECT_ROOT / "config.example.yaml"))
        stage.add_argument("--gateway-url", default=os.environ.get("HALLU_GATEWAY_URL"))
        stage.add_argument("--gateway-manifest", help="Frozen authenticated Gemini gateway manifest")
        stage.add_argument("--embedding-path", required=True)
        stage.add_argument("--max-sources", type=int, help="Checkpoint after this many cohort IDs; resume in the same directory")
        if command == "run":
            stage.add_argument("--run-dir", required=True)
            stage.add_argument("--vllm-url", default=os.environ.get("DET_REP_VLLM_BASE_URL"))
            stage.add_argument("--checkpoint", default=os.environ.get("DET_REP_VLLM_CHECKPOINT"))

    verify_r = sub.add_parser("verify-r", help="Verify complete R artifacts without model calls")
    for option in ("sources", "answers", "manifest", "work-dir", "relation-dir"):
        verify_r.add_argument(f"--{option}", required=True)

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
    package_cmd.add_argument("--relation-dir", required=True)
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
            run_dir=external(args.run_dir), relation_dir=external(args.relation_dir), work_dir=external(args.work_dir),
            sources=external(args.sources), answers=external(args.answers),
            manifest=external(args.manifest), repo_dir=Path(args.repo_dir).resolve(),
            config_path=Path(args.config).resolve(), environment_dir=external(args.environment_dir),
            out_dir=external(args.out_dir),
        )
        print(json.dumps(report, sort_keys=True))
        return
    source_path = external(args.sources)
    answer_path = external(args.answers)
    dataset.validate_pinned_inputs(source_path, answer_path)

    if args.command == "prepare":
        manifest_path, summary = prepare(source_path, answer_path, external(args.work_dir))
        print(json.dumps({"manifest": str(manifest_path), **{key: summary[key] for key in ("sources", "available", "pending", "train_available", "test_available")}}, sort_keys=True))
        return

    work_dir = external(args.work_dir)
    cache_root = work_dir / "cache" / dataset.CACHE_NAMESPACE
    run_dir = external(args.run_dir) if args.command in {"run", "replay"} else None
    manifest_path = Path(args.manifest).resolve()
    if work_dir not in manifest_path.parents:
        raise ValueError("manifest must belong to the external work directory")
    examples = load_prepared(source_path, answer_path, manifest_path)
    selected = dataset.selected_answers(examples)
    if args.command == "replay":
        from .runner import replay

        report = replay(run_dir, cache_root, selected)
        print(json.dumps(report, sort_keys=True))
        return
    if args.command == "verify-r":
        from .relations import verify_relations

        report = verify_relations(
            external(args.relation_dir), cache_root / "kg", selected,
            input_manifest_sha256=file_digest(manifest_path),
        )
        print(json.dumps(report, sort_keys=True))
        return

    if not args.gateway_url:
        raise ValueError("--gateway-url or HALLU_GATEWAY_URL is required")
    import yaml

    from .gemini import GeminiFeedbackProducer, runtime_config
    from .llm import OpenAICompatibleCorrector
    from .runner import run_arms

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("config must be a YAML mapping")
    relation_dir = external(args.relation_dir)
    if run_dir is not None and (relation_dir == run_dir or relation_dir.is_relative_to(run_dir) or run_dir.is_relative_to(relation_dir)):
        raise ValueError("relation and correction run directories must be separate")
    gateway_owner = relation_dir if args.command == "extract-r" else run_dir
    gateway_manifest = pinned_gateway_manifest(
        gateway_owner, args.gateway_url, base["llm"]["model"],
        Path(args.gateway_manifest).resolve() if args.gateway_manifest else (
            relation_dir / "gateway_manifest.json" if args.command == "run" else None
        ),
    )
    os.environ["EXPECTED_GATEWAY_MANIFEST_SHA256"] = digest(gateway_manifest)
    cfg = runtime_config(base, gateway_manifest, args.gateway_url, args.embedding_path, cache_root)
    from .core.cache import evaluation_runtime_metadata
    gateway_path = gateway_owner / "gateway_manifest.json"
    if gateway_path.exists() and json.loads(gateway_path.read_text(encoding="utf-8")) != gateway_manifest:
        raise ValueError("run directory has a different gateway revision")
    atomic_json(gateway_path, gateway_manifest)
    relation_runtime = {
        "source_sha256": file_digest(source_path), "answer_sha256": file_digest(answer_path),
        "config_sha256": file_digest(args.config), "gateway_manifest_sha256": digest(gateway_manifest),
        "extractor_fingerprint": digest({
            "llm_runtime": cfg.llm.runtime_fingerprint, "model_revision": cfg.llm.model_revision,
            "extraction": cfg.extraction.to_dict(),
        }),
        "cache_namespace": dataset.CACHE_NAMESPACE, "embedding_revision": cfg.matching.embedding_model_revision,
        "gemini_runtime": evaluation_runtime_metadata(cfg),
    }
    relation_runtime_path = relation_dir / "runtime_config.json"
    if args.command == "extract-r":
        if relation_runtime_path.exists() and read_json(relation_runtime_path) != relation_runtime:
            raise ValueError("R directory has different runtime settings")
        atomic_json(relation_runtime_path, relation_runtime)
        from .relations import run_relations
        from .core.extract import KGExtractor

        r_summary = run_relations(
            selected, KGExtractor(cfg),
            run_dir=relation_dir, kg_cache_root=cache_root / "kg",
            input_manifest_sha256=file_digest(manifest_path), max_sources=args.max_sources,
        )
        print(json.dumps(r_summary, sort_keys=True))
        return
    from .relations import verify_relations
    r_summary = verify_relations(
        relation_dir, cache_root / "kg", selected,
        input_manifest_sha256=file_digest(manifest_path),
    )
    if read_json(relation_runtime_path) != relation_runtime:
        raise ValueError("R runtime differs from correction runtime")
    if not args.vllm_url or not args.checkpoint:
        raise ValueError("local vLLM URL and exact checkpoint are required")
    corrector = OpenAICompatibleCorrector(
        base_url=args.vllm_url, model=base["corrector"]["model"], checkpoint=args.checkpoint,
        cache_dir=cache_root / "generation", seed=int(base["corrector"]["seed"]),
        temperature=float(base["corrector"]["temperature"]),
        max_tokens=int(base["corrector"]["max_tokens"]),
    )
    served_model = corrector.preflight()
    runtime_record = {
        "config_sha256": file_digest(args.config), "gateway_manifest": gateway_manifest,
        "cache_namespace": dataset.CACHE_NAMESPACE,
        "embedding_path": str(Path(args.embedding_path).resolve()),
        "embedding_revision": cfg.matching.embedding_model_revision,
        "gemini_runtime": evaluation_runtime_metadata(cfg),
        "relation_run_fingerprint": r_summary["run_fingerprint"],
        "relation_identity_sha256": file_digest(relation_dir / "run_identity.json"),
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
    producer = GeminiFeedbackProducer(cfg, gateway_manifest, cache_root / "feedback", entity_cache_only=True)
    summary = run_arms(
        selected, producer, corrector, run_dir=run_dir, cache_root=cache_root,
        input_manifest_sha256=file_digest(manifest_path), iterations=1,
        max_sources=args.max_sources,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
