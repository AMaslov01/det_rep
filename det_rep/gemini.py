"""Independent Gemini entity and VeriScore claim diagnostics for correction."""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

from . import __version__
from .contracts import EvidencePack, Example
from .core.config import Config
from .core.extract import KGExtractor
from .core.matching import EntityMatcher, SBERTEmbedder
from .core.veriscore import VeriScorePipeline, veriscore_protocol
from .util import atomic_json, digest, file_digest, read_json, text_digest


GATEWAY_PROTOCOL = "hallu-vertex-openai-gateway-v1"
FEEDBACK_PROTOCOL = "det-rep-ec-feedback-v1"


class FeedbackComponentError(RuntimeError):
    """A failed diagnostic component; the cause remains private to the caller."""

    component: str

    def __init__(self, component: str):
        self.component = component
        super().__init__(f"{component} feedback failed")


class EntityFeedbackError(FeedbackComponentError):
    def __init__(self):
        super().__init__("entities")


class ClaimFeedbackError(FeedbackComponentError):
    def __init__(self):
        super().__init__("claims")


def _reject_claim_fallbacks(items: tuple[dict[str, Any], ...]) -> None:
    if any(
        item.get("protocol_fallback")
        or item.get("fallback_reason") is not None
        or "veriscore_fallback_sentence" in item.get("candidate_sources", ())
        for item in items
    ):
        raise ValueError("claim feedback contains a fallback rather than verified VeriScore claims")


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
            "core/evidence_spans.py", "core/extract.py", "core/matching.py", "core/retry.py",
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
    value["critical"]["claim_verifier"]["cache_dir"] = str(root / "critical_verdicts")
    value["critical"]["claim_verifier"]["cache_read_dirs"] = []
    value["veriscore"]["claim_extractor"]["cache_dir"] = str(root / "veriscore_claims")
    value["veriscore"]["claim_extractor"]["cache_read_dirs"] = []
    return Config(value)


class GeminiFeedbackProducer:
    """Produce E and C separately, with durable component-specific caches."""

    def __init__(self, cfg: Config, manifest: dict[str, Any], cache_dir: str | Path, *, cache_only: bool = False):
        self.cfg = cfg
        self.cache_only = cache_only
        self.manifest_sha = validate_gateway_manifest(manifest, cfg.llm.model)
        if cfg.veriscore.claim_verifier.labels != "critical":
            raise ValueError("E/C correction requires the four-way critical claim verifier")
        common = {
            "gateway_manifest": self.manifest_sha,
            "llm_runtime": cfg.llm.runtime_fingerprint,
            "llm_model_revision": cfg.llm.model_revision,
        }
        self.entity_fingerprint = digest({
            **common, "protocol": FEEDBACK_PROTOCOL, "component": "entities",
            "extraction": cfg.extraction.to_dict(), "matching": cfg.matching.to_dict(),
        })
        self.claim_fingerprint = digest({
            **common, "protocol": FEEDBACK_PROTOCOL, "component": "claims",
            "veriscore": veriscore_protocol(cfg),
            "claim_verifier": {k: v for k, v in cfg.critical.claim_verifier.to_dict().items() if "cache" not in k},
            "matching": cfg.matching.to_dict(),
        })
        self.fingerprint = digest({
            "protocol": FEEDBACK_PROTOCOL, "entities": self.entity_fingerprint,
            "claims": self.claim_fingerprint,
        })
        self.cache_dir = Path(cache_dir)
        if not cache_only:
            (self.cache_dir / "entities").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "claims").mkdir(parents=True, exist_ok=True)
        self._extractor: KGExtractor | Any | None = None
        self.embedder = SBERTEmbedder(
            cfg.matching.embedding_model,
            model_revision=cfg.matching.embedding_model_revision,
            model_path=cfg.matching.embedding_model_path,
            device="cpu", local_files_only=True,
        )
        self._claim_pipeline: VeriScorePipeline | Any | None = None

    @property
    def extractor(self) -> Any:
        if self._extractor is None:
            self._extractor = KGExtractor(self.cfg, cache_only=self.cache_only)
        return self._extractor

    @extractor.setter
    def extractor(self, value: Any) -> None:
        self._extractor = value

    @property
    def claim_pipeline(self) -> Any:
        if self._claim_pipeline is None:
            self._claim_pipeline = VeriScorePipeline(
                self.cfg, cache_only=self.cache_only, embedder=self.embedder
            )
        return self._claim_pipeline

    @claim_pipeline.setter
    def claim_pipeline(self, value: Any) -> None:
        self._claim_pipeline = value

    def _cache_identity(self, component: str, example: Example, answer: str, evidence: EvidencePack) -> tuple[str, dict[str, str], Path]:
        if evidence.source_id != example.source_id:
            raise ValueError("cross-source evidence")
        fingerprint = self.entity_fingerprint if component == "entities" else self.claim_fingerprint
        identity = {
            "protocol": FEEDBACK_PROTOCOL, "component": component, "source_id": example.source_id,
            "answer_sha256": text_digest(answer), "evidence_sha256": digest(evidence.to_dict()),
            "component_fingerprint": fingerprint,
        }
        key = digest(identity)
        return key, identity, self.cache_dir / component / f"{key}.json"

    def _load(self, component: str, example: Example, answer: str, evidence: EvidencePack) -> tuple[tuple[dict[str, Any], ...] | None, str, dict[str, str], Path]:
        key, identity, path = self._cache_identity(component, example, answer, evidence)
        if path.exists():
            stored = read_json(path)
            if stored.get("key") != key or stored.get("identity") != identity:
                raise ValueError(f"{component} feedback cache identity mismatch")
            items = stored.get("items")
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise ValueError(f"{component} feedback cache content mismatch")
            if component == "claims":
                audits = stored.get("audits")
                if not isinstance(audits, list) or len(audits) != len(items):
                    raise ValueError("claims feedback cache has no matching verifier audits")
                _reject_claim_fallbacks(tuple(items))
            return tuple(items), key, identity, path
        if self.cache_only:
            raise ValueError(f"cache-only {component} feedback miss")
        return None, key, identity, path

    @staticmethod
    def _save(
        items: tuple[dict[str, Any], ...], key: str, identity: dict[str, str], path: Path,
        *, audits: list[dict[str, Any]] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"key": key, "identity": identity, "items": list(items)}
        if audits is not None:
            payload["audits"] = audits
        atomic_json(path, payload)

    def produce_entities(self, example: Example, answer: str, evidence: EvidencePack) -> tuple[dict[str, Any], ...]:
        try:
            cached, key, identity, path = self._load("entities", example, answer, evidence)
            if cached is not None:
                return cached
            g_context, g_query = self.extractor.extract_reference(example.context, example.query)
            g_answer = self.extractor.extract(answer, kind="response")
            reference = g_context.union(g_query)
            matcher = EntityMatcher(reference.entities, self.cfg.matching, self.embedder)
            entities: list[dict[str, Any]] = []
            for entity in sorted(g_answer.entities):
                match = matcher.match_entity(entity)
                occurrence = re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", answer, re.IGNORECASE)
                entities.append({
                    "name": entity, "start": occurrence.start() if occurrence else None,
                    "end": occurrence.end() if occurrence else None,
                    "grounded": match.matched, "reference": match.ref, "method": match.method,
                })
            result = tuple(entities)
            self._save(result, key, identity, path)
            return result
        except Exception as exc:
            raise EntityFeedbackError() from exc

    def produce_claims(self, example: Example, answer: str, evidence: EvidencePack) -> tuple[dict[str, Any], ...]:
        try:
            cached, key, identity, path = self._load("claims", example, answer, evidence)
            if cached is not None:
                return cached
            audits = self.claim_pipeline.assess(answer, example.context, example.query)
            claims = tuple({
                "id": f"c{index}", "text": item["claim"]["text"],
                "sentence_id": item["claim"]["sentence_id"],
                "sentence_start": item["claim"]["sentence_start"],
                "sentence_end": item["claim"]["sentence_end"],
                "verdict": item["verdict"],
                "candidate_sources": item["claim"].get("sources", []),
                "protocol_fallback": bool(item.get("verifier_protocol_fallback")),
                "fallback_reason": item.get("verifier_fallback_reason"),
            } for index, item in enumerate(audits))
            _reject_claim_fallbacks(claims)
            self._save(claims, key, identity, path, audits=audits)
            return claims
        except Exception as exc:
            raise ClaimFeedbackError() from exc
