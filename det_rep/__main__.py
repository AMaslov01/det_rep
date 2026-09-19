"""Prepare immutable inputs, run a train smoke, or replay artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .contracts import Example
from .dataset import load_prepared, prepare
from .util import atomic_json, file_digest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def external(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value == PROJECT_ROOT or PROJECT_ROOT in value.parents:
        raise ValueError("data, work, and run paths must be outside det_rep")
    return value


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
    smoke.add_argument("--gateway-url", default=os.environ.get("HALLU_GATEWAY_URL", "https://hallu-vertex-gateway-453887629111.europe-west4.run.app"))
    smoke.add_argument("--embedding-path", required=True)
    smoke.add_argument("--vllm-url", default=os.environ.get("DET_REP_VLLM_BASE_URL"))
    smoke.add_argument("--checkpoint", default=os.environ.get("DET_REP_VLLM_CHECKPOINT"))
    smoke.add_argument("--count", type=int, default=10)
    smoke.add_argument("--iterations", type=int, choices=(1, 2, 3), default=1)

    replay_cmd = sub.add_parser("replay")
    replay_cmd.add_argument("--sources", required=True)
    replay_cmd.add_argument("--answers", required=True)
    replay_cmd.add_argument("--manifest", required=True)
    replay_cmd.add_argument("--work-dir", required=True)
    replay_cmd.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    source_path = external(args.sources)
    answer_path = external(args.answers)

    if args.command == "prepare":
        manifest_path, summary = prepare(source_path, answer_path, external(args.work_dir))
        print(json.dumps({"manifest": str(manifest_path), **{key: summary[key] for key in ("sources", "available", "pending", "train_available", "test_available")}}, sort_keys=True))
        return

    work_dir = external(args.work_dir)
    run_dir = external(args.run_dir)
    manifest_path = Path(args.manifest).resolve()
    if work_dir not in manifest_path.parents:
        raise ValueError("manifest must belong to the external work directory")
    examples = load_prepared(source_path, answer_path, manifest_path)
    if args.command == "replay":
        from .runner import replay

        report = replay(run_dir, work_dir / "cache", examples)
        print(json.dumps(report, sort_keys=True))
        return

    if not args.vllm_url or not args.checkpoint:
        raise ValueError("local vLLM URL and exact checkpoint are required")
    import yaml

    from .gemini import GeminiFeedbackProducer, fetch_gateway_manifest, runtime_config
    from .llm import OpenAICompatibleCorrector
    from .runner import run_arms

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("config must be a YAML mapping")
    selected = smoke_selection(examples, args.count)
    corrector = OpenAICompatibleCorrector(
        base_url=args.vllm_url, model=base["corrector"]["model"], checkpoint=args.checkpoint,
        cache_dir=work_dir / "cache" / "generation", seed=int(base["corrector"]["seed"]),
        temperature=float(base["corrector"]["temperature"]),
        max_tokens=int(base["corrector"]["max_tokens"]),
    )
    served_model = corrector.preflight()
    gateway_manifest = fetch_gateway_manifest(args.gateway_url, base["llm"]["model"])
    cfg = runtime_config(base, gateway_manifest, args.gateway_url, args.embedding_path, work_dir / "cache")
    from .core.cache import evaluation_runtime_metadata
    gateway_path = run_dir / "gateway_manifest.json"
    if gateway_path.exists() and json.loads(gateway_path.read_text(encoding="utf-8")) != gateway_manifest:
        raise ValueError("run directory has a different gateway revision")
    atomic_json(gateway_path, gateway_manifest)
    runtime_record = {
        "config_sha256": file_digest(args.config), "gateway_manifest": gateway_manifest,
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
    producer = GeminiFeedbackProducer(cfg, gateway_manifest, work_dir / "cache" / "feedback")
    summary = run_arms(
        selected, producer, corrector, run_dir=run_dir, cache_root=work_dir / "cache",
        input_manifest_sha256=file_digest(manifest_path), iterations=args.iterations,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
