"""Existing Gemini KG/claim pipeline adapted to versioned correction feedback."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

from . import __version__
from .contracts import EvidencePack, Example, FeedbackRecord
from .core.config import Config
from .core.critical import CriticalClaimPipeline, CriticalClaimVerifier
from .core.extract import KGExtractor
from .core.matching import RefGraph, SBERTEmbedder
from .core.metrics import score_response
from .core.veriscore import VeriScorePipeline, veriscore_protocol
from .util import atomic_json, digest, file_digest, read_json, text_digest


GATEWAY_PROTOCOL = "hallu-vertex-openai-gateway-v1"
FEEDBACK_PROTOCOL = "det-rep-gemini-feedback-v1"
CLAIM_METHODS = ("support-critical", "veriscore")


def validate_gateway_manifest(manifest: dict[str, Any], model: str) -> str:
    expected = {
        "protocol": GATEWAY_PROTOCOL, "api_path": "/v1", "logical_model": model,
        "vertex_model": model.removeprefix("openai/"), "vertex_location": "eu",
    }
    if not model.startswith("openai/"):
        raise ValueError("Gemini logical model must use openai/ prefix")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"gateway manifest mismatch: {key}")
    for key in ("gateway_release", "cloud_run_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"gateway manifest missing {key}")
    return digest(manifest)


def fetch_gateway_manifest(url: str, model: str, *, client: Any = None) -> dict[str, Any]:
    import httpx

    secret = os.environ.get("HALLU_GATEWAY_API_KEY")
    if not secret:
        raise ValueError("HALLU_GATEWAY_API_KEY is required for live Gemini inference")
    base_url = url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    sender = client or httpx.Client(timeout=30)
    try:
        response = sender.get(base_url + "/v1/hallu/manifest", headers={"Authorization": f"Bearer {secret}"})
        response.raise_for_status()
        manifest = response.json()
        if not isinstance(manifest, dict):
            raise ValueError("gateway manifest is not an object")
        validate_gateway_manifest(manifest, model)
        return manifest
    finally:
        if client is None:
            sender.close()


def runtime_config(base: dict[str, Any], manifest: dict[str, Any], gateway_url: str, embedding_path: str | Path, cache_root: str | Path) -> Config:
    value = copy.deepcopy(base)
    llm = value["llm"]
    manifest_sha = validate_gateway_manifest(manifest, llm["model"])
    embedding = Path(embedding_path).resolve()
    if not (embedding / "config.json").is_file():
        raise ValueError("embedding path must be a local S-BERT snapshot")
    embedding_files = {
        str(path.relative_to(embedding)): file_digest(path)
        for path in sorted(embedding.rglob("*"))
        if path.is_file() and path.name != ".DS_Store" and ".git" not in path.parts
    }
    embedding_sha = digest(embedding_files)
    package_root = Path(__file__).resolve().parent
    code_sha = digest({
        name: file_digest(package_root / name)
        for name in (
            "gemini.py", "contracts.py", "evidence.py", "core/cache.py", "core/config.py",
            "core/critical.py", "core/dspy_adapter.py",
            "core/extract.py", "core/matching.py", "core/metrics.py", "core/retry.py", "core/verifier.py",
            "core/veriscore.py",
        )
    })
    base_url = gateway_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    llm.update({
        "api_base": base_url + "/v1", "api_key_env": "HALLU_GATEWAY_API_KEY",
        "model_revision": f"{manifest['vertex_model']}:{manifest['gateway_release']}:{manifest['cloud_run_revision']}",
        "runtime_fingerprint": f"vertex-gateway:{digest({'manifest': manifest_sha, 'code': code_sha, 'det_rep': __version__})}",
        "structured_output_transport": "response_format", "structured_output_backend": "vertex",
        "structured_output_request_backend": None, "concurrency": 1,
    })
    matching = value["matching"]
    matching.update({
        "embedding_model_path": str(embedding), "embedding_device": "cpu", "local_files_only": True,
        "embedding_snapshot_sha256": embedding_sha,
        "embedding_model_revision": f"{matching['embedding_model_revision']}:{embedding_sha}",
    })
    root = Path(cache_root).resolve()
    value["cache_dir"] = str(root / "kg")
    value["cache_read_dirs"] = []
    value["relation_verifier"]["cache_dir"] = str(root / "verdicts")
    value["relation_verifier"]["cache_read_dirs"] = []
    for name, folder in (("claim_extractor", "critical_claims"), ("coverage_reviewer", "critical_coverage"), ("claim_verifier", "critical_verdicts")):
        value["support_critical"][name]["cache_dir"] = str(root / folder)
        value["support_critical"][name]["cache_read_dirs"] = []
    if "veriscore" in value:
        for name, folder in (("claim_extractor", "veriscore_claims"), ("claim_verifier", "veriscore_verdicts")):
            value["veriscore"][name]["cache_dir"] = str(root / folder)
            value["veriscore"][name]["cache_read_dirs"] = []
    return Config(value)


class GeminiFeedbackProducer:
    def __init__(self, cfg: Config, manifest: dict[str, Any], cache_dir: str | Path, *, cache_only: bool = False):
        self.cfg = cfg
        self.cache_only = cache_only
        self.manifest_sha = validate_gateway_manifest(manifest, cfg.llm.model)
        self.claim_method = str(cfg.get("claim_method", "support-critical"))
        if self.claim_method not in CLAIM_METHODS:
            raise ValueError(f"unknown claim method: {self.claim_method}")
        claim_protocol: dict[str, Any] = {
            name: {key: value for key, value in getattr(cfg.support_critical, name).to_dict().items() if "cache" not in key}
            for name in ("claim_extractor", "coverage_reviewer", "claim_verifier")
        }
        if self.claim_method == "veriscore":
            claim_protocol["veriscore"] = veriscore_protocol(cfg)
        self.fingerprint = digest({
            "protocol": FEEDBACK_PROTOCOL, "gateway_manifest": self.manifest_sha,
            "llm_runtime": cfg.llm.runtime_fingerprint,
            "llm_model_revision": cfg.llm.model_revision,
            "extraction": cfg.extraction.to_dict(), "matching": cfg.matching.to_dict(),
            "claim_protocol": claim_protocol,
            "relation_protocol": {key: value for key, value in cfg.relation_verifier.to_dict().items() if "cache" not in key},
        })
        self.cache_dir = Path(cache_dir)
        if not cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = KGExtractor(cfg, cache_only=cache_only)
        self.embedder = SBERTEmbedder(
            cfg.matching.embedding_model,
            model_revision=cfg.matching.embedding_model_revision,
            model_path=cfg.matching.embedding_model_path,
            device="cpu", local_files_only=True,
        )
        self.relation_verifier = CriticalClaimVerifier(cfg, cache_only=cache_only, embedder=self.embedder)
        pipeline_type = VeriScorePipeline if self.claim_method == "veriscore" else CriticalClaimPipeline
        self.claim_pipeline = pipeline_type(cfg, cache_only=cache_only, embedder=self.embedder)

    def produce(self, example: Example, answer: str, evidence: EvidencePack) -> FeedbackRecord:
        answer_sha = text_digest(answer)
        evidence_sha = digest(evidence.to_dict())
        key = digest({"fingerprint": self.fingerprint, "source_id": example.source_id, "answer": answer_sha, "evidence": evidence_sha})
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            stored = read_json(path)
            if stored.get("key") != key:
                raise ValueError("feedback cache identity mismatch")
            record = FeedbackRecord.from_dict(stored["feedback"])
            if (record.source_id != example.source_id or record.answer_sha256 != answer_sha
                or record.evidence_sha256 != evidence_sha or record.producer_fingerprint != self.fingerprint):
                raise ValueError("feedback cache content mismatch")
            return record
        if self.cache_only:
            raise ValueError("cache-only feedback miss")
        g_context, g_query = self.extractor.extract_reference(example.context, example.query)
        g_answer = self.extractor.extract(answer, kind="response")
        ref = g_context.union(g_query)
        refgraph = RefGraph(ref.entities, ref.relations, self.cfg.matching, self.embedder)
        score = score_response(
            g_answer, refgraph, g_context, g_query, context=example.context,
            query=example.query, verifier=self.relation_verifier,
            answer_text=answer, critical_pipeline=self.claim_pipeline,
        )
        entities: list[dict[str, Any]] = []
        for entity in sorted(g_answer.entities):
            match = refgraph.match_entity(entity)
            occurrence = re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", answer, re.IGNORECASE)
            entities.append({
                "name": entity, "start": occurrence.start() if occurrence else None,
                "end": occurrence.end() if occurrence else None,
                "grounded": match.matched, "reference": match.ref, "method": match.method,
            })
        relations = tuple({
            "triple": item["answer_edge"], "canonical": item.get("canonical_edge"),
            "status": item.get("status"), "verdict": item.get("verdict"),
            "strict_alignment": item.get("strict_alignment"),
        } for item in score.relation_audits)
        claims: list[dict[str, Any]] = []
        links: list[dict[str, Any]] = []
        valid_ids = {sentence.id for sentence in evidence.sentences}
        for index, item in enumerate((score.critical or {}).get("claim_audits", [])):
            claim_id = f"c{index}"
            claim = item["claim"]
            entry = {
                "id": claim_id, "text": claim["text"], "start": claim["start"], "end": claim["end"],
                "verdict": item["verdict"], "candidate_sources": claim.get("sources", []),
                "protocol_fallback": bool(item.get("verifier_protocol_fallback")),
            }
            if "sentence_id" in claim:
                entry["sentence_id"] = claim["sentence_id"]
            if "label" in item:
                entry["label"] = item["label"]
            claims.append(entry)
            linked = sorted({
                ("C" if span.get("source") == "context" else "Q") + str(span.get("index"))
                for span in item.get("evidence", [])
            } & valid_ids)
            links.append({"claim_id": claim_id, "sentence_ids": linked})
        record = FeedbackRecord(
            source_id=example.source_id, answer_sha256=answer_sha, evidence_sha256=evidence_sha,
            producer_fingerprint=self.fingerprint, entities=tuple(entities), relations=relations,
            claims=tuple(claims), links=tuple(links),
            graph_status="unscorable" if score.unscorable else "scorable",
        )
        atomic_json(path, {"key": key, "feedback": record.to_dict()})
        return record
