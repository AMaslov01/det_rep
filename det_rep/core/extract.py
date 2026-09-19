"""KGGen extraction wrapper: disk cache, retries, chunking, cost logging.

Design notes / adaptation to the real kg-gen API (documented in README):
  - The §3 snippet shows ``kg.generate(input_data=text, cluster=...)`` and a graph with
    ``.entities`` (set[str]) and ``.relations`` (set[tuple]). The installed kg-gen matches
    this and additionally exposes ``.edges`` and ``*_clusters``; we only need entities +
    relations.
  - Long-context chunking is native: ``kg.generate(chunk_size=..., cluster=True)`` chunks,
    aggregates across chunks, and runs one clustering pass -- exactly the behaviour §3 asks
    for. We therefore map ``extraction.context_chunk_chars`` -> ``chunk_size`` rather than
    merging chunks by hand.
  - The backend is injectable (``backend=``) so tests / offline smoke runs can pass a
    ``FakeExtractor`` without importing kg-gen or hitting an API.
"""
from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import random
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from tenacity import Retrying, retry_if_exception, stop_after_attempt

from .cache import CacheOnlyMissError, config_value, llm_runtime_fingerprint
from .dspy_adapter import (
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    StructuredOutputTruncatedError,
    install_dspy_completion_guard,
    is_retryable_llm_exception,
    structured_output_settings,
)
from .retry import RetryHeartbeat, StopAfterAttemptsExceptRateLimit, WaitRetryAfterOrExponentialJitter


# --------------------------------------------------------------------------------------
# Graph container (decouples the rest of the pipeline from kg-gen's object)
# --------------------------------------------------------------------------------------
@dataclass
class Graph:
    entities: set[str] = field(default_factory=set)
    relations: set[tuple[str, str, str]] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": sorted(self.entities),
            "relations": sorted([list(r) for r in self.relations]),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Graph":
        return cls(
            entities={str(e) for e in d.get("entities", [])},
            relations={tuple(str(x) for x in r) for r in d.get("relations", []) if len(r) == 3},
        )

    @classmethod
    def empty(cls) -> "Graph":
        return cls(set(), set())

    def union(self, other: "Graph") -> "Graph":
        return Graph(self.entities | other.entities, self.relations | other.relations)


class ExtractionError(RuntimeError):
    pass


class ClusteringCollapseError(ExtractionError):
    """KGGen clustering discarded more of a probe graph than allowed."""


CLUSTER_CONTEXT_PROTOCOL = "kggen-native-strict-equivalence-v2"

CLUSTER_EQUIVALENCE_POLICY = """

Strict clustering contract:
- A cluster contains only surface variants that denote the same entity, or
  relation labels with the same truth conditions. Aliases, spelling/case
  variants, inflections, and direct paraphrases may be merged.
- Topical relatedness, co-occurrence, a shared subject/object, comparison,
  class membership, part-whole relations, and causal association are not
  equivalence. Keep such items in separate clusters.
- For example, "Paris" and "France" are related but are not the same entity;
  "was born in" and "won an award" can share a subject but are not the same
  relation.
- When no two candidates are genuinely equivalent, an empty proposed cluster
  is the correct answer. When uncertain, keep candidates separate.
"""


# --------------------------------------------------------------------------------------
# Cost / usage logging (best-effort; token accounting depends on LiteLLM being reachable)
# --------------------------------------------------------------------------------------
class UsageLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.calls = 0
        self.total_requests = 0
        self.cache_hits = 0
        self.cost = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.retries = 0
        self.retry_reasons: dict[str, int] = {}
        self._emitted_prompt_tokens = 0
        self._emitted_completion_tokens = 0
        self._litellm_hooked = False
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def try_hook_litellm(self) -> None:
        """Attach a LiteLLM success callback to accumulate cost/tokens if available."""
        if self._litellm_hooked:
            return
        try:
            import litellm  # type: ignore

            def _cb(kwargs, completion_response, start_time, end_time):  # pragma: no cover
                try:
                    usage = getattr(completion_response, "usage", None)
                    if usage:
                        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                    # A localhost vLLM deployment has no meaningful USD price.
                    # LiteLLM 1.60's generic cost calculator serialises its
                    # OpenAI-compatible response through Pydantic. Some guided
                    # JSON responses from a local 8B model can spin there after
                    # vLLM has already completed, leaving the paid GPU idle.
                    # Preserve an explicitly supplied provider cost, but never
                    # invoke the calculator merely for telemetry.
                    cost = kwargs.get("response_cost")
                    if cost is not None:
                        self.cost += float(cost)
                except Exception:
                    pass

            if _cb not in (litellm.success_callback or []):
                litellm.success_callback = list(litellm.success_callback or []) + [_cb]
            self._litellm_hooked = True
        except Exception:
            # LiteLLM not importable (e.g. offline/fake mode) -- fall back to call counts.
            self._litellm_hooked = False

    def record_call(self, kind: str, cache_key: str, seconds: float, cached: bool) -> None:
        with self._lock:
            self.total_requests += 1
            self.cache_hits += int(cached)
            self.calls += 0 if cached else 1
            prompt_tokens = self.prompt_tokens - self._emitted_prompt_tokens
            completion_tokens = self.completion_tokens - self._emitted_completion_tokens
            self._emitted_prompt_tokens = self.prompt_tokens
            self._emitted_completion_tokens = self.completion_tokens
            if not self.path:
                return
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": kind, "cache_key": cache_key, "seconds": round(seconds, 4),
                    "cached": cached, "cum_calls": self.calls,
                    "cum_requests": self.total_requests, "cum_cache_hits": self.cache_hits,
                    "cum_retries": self.retries,
                    "cum_cost_usd": round(self.cost, 6),
                    "cum_prompt_tokens": self.prompt_tokens,
                    "cum_completion_tokens": self.completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }) + "\n")

    def record_retry(self, kind: str, exc: BaseException) -> None:
        """Persist an aggregate-safe retry event without prompts or responses."""
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        reason = f"http_{status}" if status is not None else type(exc).__name__
        with self._lock:
            self.retries += 1
            self.retry_reasons[reason] = self.retry_reasons.get(reason, 0) + 1
            if not self.path:
                return
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "kind": kind, "event": "retry", "retry_reason": reason,
                    "cum_calls": self.calls, "cum_requests": self.total_requests,
                    "cum_cache_hits": self.cache_hits, "cum_retries": self.retries,
                    "cum_cost_usd": round(self.cost, 6),
                    "cum_prompt_tokens": self.prompt_tokens,
                    "cum_completion_tokens": self.completion_tokens,
                }) + "\n")

    def summary(self) -> dict[str, Any]:
        return {
            "api_calls": self.calls,
            "requests_total": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / self.total_requests, 6)
            if self.total_requests else 0.0,
            "retries": self.retries,
            "retry_reasons": dict(sorted(self.retry_reasons.items())),
            "estimated_cost_usd": round(self.cost, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# --------------------------------------------------------------------------------------
# Extractor
# --------------------------------------------------------------------------------------
class KGExtractor:
    def __init__(
        self,
        cfg,
        backend: Any | None = None,
        usage: UsageLogger | None = None,
        *,
        cache_only: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        before_live_request: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        # KGGen 0.4 defaults to 16k output tokens for GPT-5-family models.
        # Keep that upstream default for the main config unless a caller
        # explicitly supplies a narrower cap (the micro demo does).
        self.max_tokens = (
            cfg.llm.get("max_tokens", None)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "max_tokens", None)
        )
        self.cluster = cfg.extraction.cluster
        self.cluster_context_mode = str(
            config_value(cfg.extraction, "cluster_context_mode", "empty")
        )
        if self.cluster_context_mode not in {"empty", "source_text"}:
            raise ValueError(
                "extraction.cluster_context_mode must be 'empty' or 'source_text'"
            )
        raw_cluster_max_items = (
            cfg.extraction.get("cluster_max_items", None)
            if hasattr(cfg.extraction, "get")
            else getattr(cfg.extraction, "cluster_max_items", None)
        )
        self.cluster_max_items = (
            int(raw_cluster_max_items) if raw_cluster_max_items is not None else None
        )
        if self.cluster_max_items is not None and self.cluster_max_items <= 0:
            raise ValueError("extraction.cluster_max_items must be positive or null")
        raw_min_retention = config_value(
            cfg.extraction, "cluster_min_retention_ratio", None
        )
        self.cluster_min_retention_ratio = (
            float(raw_min_retention) if raw_min_retention is not None else None
        )
        if self.cluster_min_retention_ratio is not None and not (
            0.0 <= self.cluster_min_retention_ratio <= 1.0
        ):
            raise ValueError(
                "extraction.cluster_min_retention_ratio must be between 0 and 1 or null"
            )
        self.cluster_retention_min_items = int(
            config_value(cfg.extraction, "cluster_retention_min_items", 5)
        )
        if self.cluster_retention_min_items <= 0:
            raise ValueError("extraction.cluster_retention_min_items must be positive")
        self.chunk_chars = cfg.extraction.context_chunk_chars
        self.serial_chunking = (
            cfg.extraction.get("serial_chunking", False)
            if hasattr(cfg.extraction, "get")
            else getattr(cfg.extraction, "serial_chunking", False)
        )
        self.explicit_clustering = (
            cfg.extraction.get("explicit_clustering", False)
            if hasattr(cfg.extraction, "get")
            else getattr(cfg.extraction, "explicit_clustering", False)
        )
        # Structured output is a transport/runtime property; ``llm.model``
        # remains the sole model slug.  ``guided_json`` is retained only as a
        # deprecated compatibility route for old caches and is never selected
        # by the research DataSphere profile.
        self.structured_output = structured_output_settings(cfg.llm)
        self.vllm_guided_json = self.structured_output.transport == "guided_json"
        raw_dump_after = os.environ.get("DATASPHERE_KGGEN_DUMP_AFTER_SECONDS", "")
        self.debug_dump_after_s = float(raw_dump_after) if raw_dump_after else None
        if self.debug_dump_after_s is not None and self.debug_dump_after_s <= 0:
            raise ValueError("DATASPHERE_KGGEN_DUMP_AFTER_SECONDS must be positive when set")
        self.max_retries = int(cfg.llm.max_retries)
        if self.max_retries < 0:
            raise ValueError("llm.max_retries must be non-negative")
        self.backoff_base = cfg.llm.retry_backoff_base_s
        self.backoff_max = float(
            cfg.llm.get("retry_backoff_max_s", 60)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "retry_backoff_max_s", 60)
        )
        if self.backoff_max < float(self.backoff_base):
            raise ValueError("llm.retry_backoff_max_s must be at least retry_backoff_base_s")
        self.rate_limit_cooldown_max_s = float(
            cfg.llm.get("rate_limit_cooldown_max_s", 900)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "rate_limit_cooldown_max_s", 900)
        )
        if self.rate_limit_cooldown_max_s < self.backoff_max:
            raise ValueError(
                "llm.rate_limit_cooldown_max_s must be at least retry_backoff_max_s"
            )
        self.rate_limit_retry_deadline_s = float(
            cfg.llm.get("rate_limit_retry_deadline_s", 1800)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "rate_limit_retry_deadline_s", 1800)
        )
        self.retry_deadline_s = float(
            cfg.llm.get("retry_deadline_s", 90)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "retry_deadline_s", 90)
        )
        if self.rate_limit_retry_deadline_s <= 0:
            raise ValueError("llm.rate_limit_retry_deadline_s must be positive")
        if self.retry_deadline_s <= 0:
            raise ValueError("llm.retry_deadline_s must be positive")
        self.max_protocol_retries = int(
            config_value(cfg.extraction, "max_protocol_retries", 0)
        )
        if self.max_protocol_retries < 0:
            raise ValueError("extraction.max_protocol_retries must be non-negative")
        raw_max_tokens_ceiling = config_value(
            cfg.extraction, "max_tokens_ceiling", self.max_tokens
        )
        self.max_tokens_ceiling = (
            int(raw_max_tokens_ceiling)
            if raw_max_tokens_ceiling is not None
            else None
        )
        if self.max_tokens is not None and self.max_tokens_ceiling is not None:
            if self.max_tokens_ceiling < int(self.max_tokens):
                raise ValueError(
                    "extraction.max_tokens_ceiling must be at least llm.max_tokens"
                )
        self.request_timeout_s = float(
            cfg.llm.get("request_timeout_s", 90)
            if hasattr(cfg.llm, "get")
            else getattr(cfg.llm, "request_timeout_s", 90)
        )
        if self.request_timeout_s <= 0:
            raise ValueError("llm.request_timeout_s must be positive")
        self.cache_dir = Path(cfg.cache_dir)
        raw_read_dirs = config_value(cfg, "cache_read_dirs", []) or []
        self.cache_read_dirs = [Path(str(path)) for path in raw_read_dirs]
        self.cache_only = bool(cache_only)
        if not self.cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = backend  # if None, lazily construct KGGen on first use
        self._token_budget_lock = threading.RLock()
        self._cluster_audit_lock = threading.Lock()
        self.last_cluster_audit: dict[str, Any] | None = None
        # Never let an offline FakeKGGen smoke artifact masquerade as a live
        # KGGen graph.  A cache-only production replay uses ``backend=None``
        # and therefore receives the same stable ``kggen`` namespace as the
        # original live extraction without constructing that backend.
        self.backend_fingerprint = (
            "kggen"
            if backend is None
            else f"{type(backend).__module__}.{type(backend).__qualname__}"
        )
        self.usage = usage or UsageLogger(None)
        # Optional redacted instrumentation for long document-level jobs.  It
        # receives counts and phase names only; callers must never pass source
        # text, prompts, completions, cache keys, or credentials through it.
        self.progress_callback = progress_callback
        # ``UsageLogger.record_call`` is document-level. A caller that needs a
        # hard cost ceiling can reserve every raw KGGen operation here instead.
        self.before_live_request = before_live_request

    def _emit_progress(self, **payload: Any) -> None:
        if self.progress_callback is not None:
            self.progress_callback(dict(payload))

    def _reserve_live_request(self, kind: str, operation: str) -> None:
        if self.before_live_request is not None:
            self.before_live_request(operation)
        self._emit_progress(event="live_request_reserved", kind=kind, phase=operation)

    # -- backend --------------------------------------------------------------
    def _get_backend(self):
        if self.cache_only:
            raise RuntimeError("cache-only mode forbids constructing a KGGen/LLM backend")
        if self._backend is None:
            from kg_gen import KGGen  # lazy import; not needed for offline tests
            from .config import resolve_api_key

            kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            api_key = resolve_api_key(self.cfg)
            if api_key:
                kwargs["api_key"] = api_key
            api_base = getattr(self.cfg.llm, "api_base", None)
            if api_base:
                kwargs["api_base"] = api_base
            self._backend = KGGen(**kwargs)
            # KGGen 0.4 does not expose DSPy's HTTP timeout in its constructor.
            # Bound the underlying local-vLLM request anyway: a request that is
            # never accepted by the server must surface as a retryable error,
            # not occupy a paid GPU indefinitely.  KGExtractor's tenacity loop
            # remains the single retry policy, hence DSPy's own retries are off.
            lm = getattr(self._backend, "lm", None)
            if lm is not None:
                lm.kwargs["timeout"] = self.request_timeout_s
                lm.num_retries = 0
                install_dspy_completion_guard(lm)
            self.usage.try_hook_litellm()
        return self._backend

    @contextmanager
    def _temporary_token_budget(self, max_tokens: int | None):
        """Temporarily raise one KGGen request's output ceiling.

        Gemini may use its initial budget on hidden reasoning and return a
        strictly invalid truncated document.  The configured ceiling is a
        bounded retry control, not a changed extraction protocol: only a
        ``finish_reason=length`` response enters this branch.  The lock keeps
        this per-request override isolated if a caller configures extraction
        concurrency above one.
        """
        if max_tokens is None or max_tokens == self.max_tokens:
            yield
            return
        backend = self._get_backend()
        lm = getattr(backend, "lm", None)
        kwargs = getattr(lm, "kwargs", None)
        if not isinstance(kwargs, dict):
            raise RuntimeError("KGGen backend has no mutable LM token budget")
        with self._token_budget_lock:
            had_previous = "max_tokens" in kwargs
            previous = kwargs.get("max_tokens")
            kwargs["max_tokens"] = max_tokens
            try:
                yield
            finally:
                if had_previous:
                    kwargs["max_tokens"] = previous
                else:
                    kwargs.pop("max_tokens", None)

    # -- cache ----------------------------------------------------------------
    def _cache_key(self, text: str) -> str:
        params = {
            "v": 11,
            "extractor_protocol": "kggen-0.4-strict-cache-v5-vertex-runtime-contracts",
            "backend": self.backend_fingerprint,
            "llm": llm_runtime_fingerprint(self.cfg),
            "api_base": config_value(self.cfg.llm, "api_base"),
            "max_tokens": self.max_tokens,
            "cluster": self.cluster,
            "cluster_context_mode": self.cluster_context_mode,
            "cluster_context_protocol": CLUSTER_CONTEXT_PROTOCOL,
            "cluster_max_items": self.cluster_max_items,
            "cluster_min_retention_ratio": self.cluster_min_retention_ratio,
            "cluster_retention_min_items": self.cluster_retention_min_items,
            "chunk_chars": self.chunk_chars,
            "serial_chunking": self.serial_chunking,
            "explicit_clustering": self.explicit_clustering,
            "vllm_guided_json": self.vllm_guided_json,
        }
        payload = json.dumps(params, sort_keys=True) + "\x00" + text
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_candidates(self, key: str):
        """Yield the writable cache first, then immutable read-through roots."""
        yield "primary", self._cache_path(key)
        for index, root in enumerate(self.cache_read_dirs, start=1):
            yield f"read-through-{index}", root / f"{key}.json"

    @staticmethod
    def _read_cache_file(path: Path, key: str) -> Graph | None:
        """Return a structurally valid cache entry without any network access."""
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(envelope, dict) or set(envelope) != {
                "protocol", "cache_key", "graph", "graph_sha256"
            }:
                return None
            if envelope["protocol"] != "hallu-kg-cache-v2" or envelope["cache_key"] != key:
                return None
            graph_payload = envelope["graph"]
            if not isinstance(graph_payload, dict) or set(graph_payload) != {
                "entities", "relations"
            }:
                return None
            canonical = json.dumps(
                graph_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != envelope["graph_sha256"]:
                return None
            graph = Graph.from_dict(graph_payload)
            # Round-tripping catches malformed/dropped relation rows.
            return graph if graph.to_dict() == graph_payload else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def cache_location(self, key: str) -> tuple[str, Path] | None:
        """Locate a valid entry across primary/read-through roots.

        This is intentionally validation-aware: a file merely existing is not
        evidence that a cache-only experiment can reproduce it.  A corrupt
        primary entry must also not hide a valid immutable historical entry.
        """
        for origin, path in self._cache_candidates(key):
            if self._read_cache_file(path, key) is not None:
                return origin, path
        return None

    def _load_cache(self, key: str) -> Graph | None:
        # The primary directory is writable for this run.  Read-through roots
        # are historical, content-addressed graph namespaces: they are never
        # modified and an envelope/key check still rejects incompatible graphs.
        # Invalid files are cache misses at that root, not a reason to skip a
        # valid later read-through root.
        for _, path in self._cache_candidates(key):
            graph = self._read_cache_file(path, key)
            if graph is not None:
                return graph
        return None

    def _save_cache(self, key: str, graph: Graph) -> None:
        dest = self._cache_path(key)
        # Unique temp name per writer: concurrent threads extracting identical text share a
        # cache key, so a fixed tmp name would race on replace() (FileNotFoundError). A unique
        # tmp + atomic os.replace makes concurrent writes of the same key safe and idempotent.
        import os

        graph_payload = graph.to_dict()
        canonical = json.dumps(
            graph_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        envelope = {
            "protocol": "hallu-kg-cache-v2",
            "cache_key": key,
            "graph": graph_payload,
            "graph_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        }
        tmp = dest.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, dest)  # atomic -> crash-safe / resumable

    # -- generation -----------------------------------------------------------
    @staticmethod
    def _split_text(text: str, chunk_size: int) -> list[str]:
        """Use KGGen's own sentence-aware splitter without its thread pool."""
        from kg_gen.utils.chunk_text import chunk_text

        return chunk_text(text, chunk_size)

    def _should_cluster_backend_graph(self, graph: Any) -> bool:
        """Return whether optional KGGen canonicalisation is safe to run.

        KGGen creates dynamic ``Literal[...]`` schemas for all entities and
        predicates before asking the model to cluster them.  That is useful
        canonicalisation, but a small local model can occasionally emit a very
        large candidate list.  Building that schema becomes CPU-bound before a
        new vLLM request is sent.  A finite limit retains every raw triple and
        simply skips this optional post-processing for that outlier.  ``None``
        is deliberately the default so non-DataSphere behaviour is unchanged.
        """
        if not self.cluster:
            return False
        if self.cluster_max_items is None:
            return True
        entities = getattr(graph, "entities", set()) or set()
        edges = getattr(graph, "edges", set()) or {
            relation[1] for relation in (getattr(graph, "relations", set()) or set())
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        largest_group = max(len(entities), len(edges))
        if largest_group <= self.cluster_max_items:
            return True
        print(
            "[kg] skipping optional KGGen clustering: "
            f"entities={len(entities)} predicates={len(edges)} "
            f"limit={self.cluster_max_items}; raw triples are retained",
            flush=True,
        )
        return False

    def _cluster_context(self, source_text: str) -> str:
        """Build KGGen's documented clustering context without role leakage."""
        if self.cluster_context_mode == "empty":
            return ""
        return CLUSTER_EQUIVALENCE_POLICY + "\nSource evidence:\n" + source_text

    @staticmethod
    def _cluster_map(value: Any) -> dict[str, list[str]] | None:
        if value is None:
            return None
        return {
            str(representative): sorted(str(member) for member in (members or set()))
            for representative, members in sorted(
                value.items(), key=lambda item: str(item[0])
            )
        }

    def _write_cluster_audit(self, record: dict[str, Any]) -> None:
        """Persist and emit a complete, source-text-free clustering trail."""
        self.last_cluster_audit = record
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        # ``record`` is deliberately detailed enough to diagnose a bad native
        # KGGen remap, so it contains graph members and the content-addressed
        # cache key.  Keep that local persistent-cache audit off stdout: the
        # controlled GCP runner retains stdout as its redacted terminal log.
        # Counts and structural status are sufficient progress diagnostics.
        raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
        clustered = (
            record.get("clustered")
            if isinstance(record.get("clustered"), dict)
            else {}
        )
        failures = record.get("failures")
        progress = {
            "protocol": "hallu-kg-cluster-progress-v1",
            "kind": record.get("kind"),
            "context_mode": record.get("context_mode"),
            "status": record.get("status"),
            "source_text_chars": record.get("source_text_chars"),
            "failure_count": len(failures) if isinstance(failures, list) else 0,
            "raw_counts": {
                "entities": len(raw.get("entities", [])),
                "predicates": len(raw.get("predicates", [])),
                "relations": len(raw.get("relations", [])),
            },
            "clustered_counts": {
                "entities": len(clustered.get("entities", [])),
                "predicates": len(clustered.get("predicates", [])),
                "relations": len(clustered.get("relations", [])),
            },
        }
        print(
            "[kg] cluster:audit "
            + json.dumps(progress, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        path = self.cache_dir.parent / "cluster-audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._cluster_audit_lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    @staticmethod
    def _cluster_mapping_checks(
        *,
        raw_items: set[str],
        clustered_items: set[str],
        clusters: dict[str, list[str]] | None,
    ) -> dict[str, Any]:
        """Audit structural invariants without constraining semantic choices."""
        if clusters is None:
            return {
                "available": False,
                "representatives_match_clustered_items": False,
                "representatives_are_members": False,
                "members_cover_raw_items": False,
                "members_are_disjoint": False,
                "duplicate_members": [],
            }
        members = [member for values in clusters.values() for member in values]
        duplicate_members = sorted(
            member for member in set(members) if members.count(member) > 1
        )
        checks = {
            "available": True,
            "representatives_match_clustered_items": set(clusters) == clustered_items,
            "representatives_are_members": all(
                representative in cluster_members
                for representative, cluster_members in clusters.items()
            ),
            "members_cover_raw_items": set(members) == raw_items,
            "members_are_disjoint": not duplicate_members,
            "duplicate_members": duplicate_members,
        }
        return checks

    @staticmethod
    def _cluster_member_map(
        clusters: dict[str, list[str]] | None,
    ) -> dict[str, str]:
        if clusters is None:
            return {}
        return {
            member: representative
            for representative, members in clusters.items()
            for member in members
        }

    def _cluster_backend_graph(
        self,
        backend: Any,
        graph: Any,
        *,
        source_text: str = "",
        cache_key: str | None = None,
        kind: str = "graph",
    ) -> Any:
        """Run KGGen's own cluster pass with observable phase boundaries."""
        if not self._should_cluster_backend_graph(graph):
            return graph
        entities = getattr(graph, "entities", set()) or set()
        raw_relations = getattr(graph, "relations", set()) or set()
        edges = getattr(graph, "edges", set()) or {
            relation[1] for relation in raw_relations
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        print(
            f"[kg] cluster:start entities={len(entities)} "
            f"predicates={len(edges)} relations={len(raw_relations)}",
            flush=True,
        )
        cluster_context = self._cluster_context(source_text)
        self._reserve_live_request(kind, "clustering")
        try:
            clustered = backend.cluster(graph, context=cluster_context)
        except Exception as exc:
            # KGGen builds a dynamic Literal schema for cluster candidates.
            # Vertex can reject an otherwise valid cluster request when the
            # added source-evidence context combines with that schema.  Retry
            # the *same full KGGen cluster pass* without the optional context;
            # never replace it with raw triples or an ``unknown`` result.
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            try:
                is_bad_request = int(status) == 400
            except (TypeError, ValueError):
                is_bad_request = type(exc).__name__ == "BadRequestError"
            if self.cluster_context_mode != "source_text" or not is_bad_request:
                raise
            print("[kg] cluster:retry context=empty reason=vertex_bad_request", flush=True)
            self._reserve_live_request(kind, "clustering")
            try:
                clustered = backend.cluster(graph, context="")
            except Exception as empty_exc:
                empty_status = (
                    getattr(empty_exc, "status_code", None)
                    or getattr(empty_exc, "code", None)
                )
                try:
                    empty_is_bad_request = int(empty_status) == 400
                except (TypeError, ValueError):
                    empty_is_bad_request = type(empty_exc).__name__ == "BadRequestError"
                if not empty_is_bad_request:
                    raise
                # Vertex has a finite structured-schema complexity limit.  This is
                # not a raw-graph fallback: every entity and predicate is sent
                # through KGGen's native cluster pass in deterministic groups and
                # every resulting remap is checked below.  The groups are only a
                # transport segmentation for the dynamic Literal schemas.
                print("[kg] cluster:retry segmented reason=vertex_bad_request", flush=True)
                clustered = self._cluster_backend_graph_segmented(
                    backend,
                    graph,
                    kind=kind,
                )
        clustered_entities = getattr(clustered, "entities", set()) or set()
        clustered_relations = getattr(clustered, "relations", set()) or set()
        clustered_edges = getattr(clustered, "edges", set()) or {
            relation[1] for relation in clustered_relations
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        entity_retention = len(clustered_entities) / len(entities) if entities else 1.0
        predicate_retention = len(clustered_edges) / len(edges) if edges else 1.0
        print(
            f"[kg] cluster:done entities={len(clustered_entities)} "
            f"predicates={len(clustered_edges)} relations={len(clustered_relations)}",
            flush=True,
        )
        entity_clusters = self._cluster_map(
            getattr(clustered, "entity_clusters", None)
        )
        edge_clusters = self._cluster_map(getattr(clustered, "edge_clusters", None))
        raw_entities = {str(value) for value in entities}
        raw_edges = {str(value) for value in edges}
        clustered_entity_names = {str(value) for value in clustered_entities}
        clustered_edge_names = {str(value) for value in clustered_edges}
        raw_triples = {
            tuple(str(value) for value in relation)
            for relation in raw_relations
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        clustered_triples = {
            tuple(str(value) for value in relation)
            for relation in clustered_relations
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        entity_checks = self._cluster_mapping_checks(
            raw_items={str(value) for value in entities},
            clustered_items=clustered_entity_names,
            clusters=entity_clusters,
        )
        predicate_checks = self._cluster_mapping_checks(
            raw_items=raw_edges,
            clustered_items=clustered_edge_names,
            clusters=edge_clusters,
        )
        entity_member_map = self._cluster_member_map(entity_clusters)
        edge_member_map = self._cluster_member_map(edge_clusters)
        expected_clustered_triples = {
            (
                subject
                if subject in clustered_entity_names
                else entity_member_map.get(subject, subject),
                predicate
                if predicate in clustered_edge_names
                else edge_member_map.get(predicate, predicate),
                obj
                if obj in clustered_entity_names
                else entity_member_map.get(obj, obj),
            )
            for subject, predicate, obj in raw_triples
        }
        relation_checks = {
            "raw_rows_are_triples": len(raw_triples) == len(raw_relations),
            "raw_endpoints_in_entities": all(
                subject in raw_entities and obj in raw_entities
                for subject, _, obj in raw_triples
            ),
            "raw_predicates_in_edges": all(
                predicate in raw_edges for _, predicate, _ in raw_triples
            ),
            "clustered_rows_are_triples": len(clustered_triples) == len(clustered_relations),
            "clustered_endpoints_in_entities": all(
                subject in clustered_entity_names and obj in clustered_entity_names
                for subject, _, obj in clustered_triples
            ),
            "clustered_predicates_in_edges": all(
                predicate in clustered_edge_names
                for _, predicate, _ in clustered_triples
            ),
            "relations_match_cluster_remap": (
                expected_clustered_triples == clustered_triples
            ),
        }
        failures: list[str] = []
        for label, checks in (
            ("entity", entity_checks),
            ("predicate", predicate_checks),
        ):
            if not all(
                checks[name]
                for name in (
                    "available",
                    "representatives_match_clustered_items",
                    "representatives_are_members",
                    "members_cover_raw_items",
                    "members_are_disjoint",
                )
            ):
                failures.append(f"inconsistent {label} mapping")
        if not all(relation_checks.values()):
            failures.append("inconsistent clustered relations")
        if raw_relations and not clustered_relations:
            failures.append("all relations disappeared")
        if self.cluster_min_retention_ratio is not None:
            if (
                len(entities) >= self.cluster_retention_min_items
                and entity_retention < self.cluster_min_retention_ratio
            ):
                failures.append(f"entity retention={entity_retention:.6f}")
            if (
                len(edges) >= self.cluster_retention_min_items
                and predicate_retention < self.cluster_min_retention_ratio
            ):
                failures.append(f"predicate retention={predicate_retention:.6f}")
        audit_record = {
            "protocol": CLUSTER_CONTEXT_PROTOCOL,
            "context_mode": self.cluster_context_mode,
            "cache_key": cache_key,
            "kind": kind,
            "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "source_text_chars": len(source_text),
            "status": "error" if failures else "ready",
            "failures": failures,
            "raw": {
                "entities": sorted(raw_entities),
                "predicates": sorted(raw_edges),
                "relations": sorted([list(relation) for relation in raw_triples]),
            },
            "clustered": {
                "entities": sorted(clustered_entity_names),
                "predicates": sorted(clustered_edge_names),
                "relations": sorted([list(relation) for relation in clustered_triples]),
            },
            "entity_clusters": entity_clusters,
            "edge_clusters": edge_clusters,
            "structural_checks": {
                "entities": entity_checks,
                "predicates": predicate_checks,
                "relations": relation_checks,
            },
            "retention": {
                "entities": entity_retention,
                "predicates": predicate_retention,
                "relations": (
                    len(clustered_relations) / len(raw_relations)
                    if raw_relations else 1.0
                ),
            },
        }
        self._write_cluster_audit(audit_record)
        print(
            "[kg] cluster:retention "
            f"entities={len(clustered_entities)}/{len(entities)} "
            f"({entity_retention:.6f}) predicates={len(clustered_edges)}/{len(edges)} "
            f"({predicate_retention:.6f}) relations={len(clustered_relations)}/{len(raw_relations)}",
            flush=True,
        )
        if failures:
            raise ClusteringCollapseError(
                "KGGen clustering structural/retention gate failed: "
                + "; ".join(failures)
            )
        return clustered

    def _cluster_backend_graph_segmented(
        self,
        backend: Any,
        graph: Any,
        *,
        kind: str,
    ) -> Any:
        """Run native KGGen clustering in schema-safe deterministic groups.

        KGGen's cluster signatures enumerate every candidate in a JSON Schema.
        A Vertex 400 after both context variants means that schema, rather than
        the document, exceeds a service limit.  Groups retain all candidates;
        relations are remapped only after their endpoint and predicate groups
        return structurally complete native cluster maps.
        """
        from kg_gen.models import Graph

        group_size = 32
        raw_entities = sorted(str(value) for value in (getattr(graph, "entities", set()) or set()))
        raw_edges = sorted(str(value) for value in (getattr(graph, "edges", set()) or set()))
        raw_relations = {
            tuple(str(value) for value in relation)
            for relation in (getattr(graph, "relations", set()) or set())
            if isinstance(relation, (tuple, list)) and len(relation) == 3
        }
        entity_groups = [raw_entities[index : index + group_size] for index in range(0, len(raw_entities), group_size)] or [[]]
        edge_groups = [raw_edges[index : index + group_size] for index in range(0, len(raw_edges), group_size)] or [[]]
        group_count = max(len(entity_groups), len(edge_groups))
        entity_member_map: dict[str, str] = {}
        edge_member_map: dict[str, str] = {}
        entity_clusters: dict[str, set[str]] = {}
        edge_clusters: dict[str, set[str]] = {}

        for index in range(group_count):
            entities = set(entity_groups[index]) if index < len(entity_groups) else set()
            edges = set(edge_groups[index]) if index < len(edge_groups) else set()
            self._reserve_live_request(kind, "clustering")
            grouped = backend.cluster(
                Graph(entities=entities, edges=edges, relations=set()), context=""
            )
            grouped_entity_clusters = getattr(grouped, "entity_clusters", None) or {}
            grouped_edge_clusters = getattr(grouped, "edge_clusters", None) or {}
            for representative, members in grouped_entity_clusters.items():
                member_set = {str(member) for member in members}
                entity_clusters[str(representative)] = member_set
                entity_member_map.update({member: str(representative) for member in member_set})
            for representative, members in grouped_edge_clusters.items():
                member_set = {str(member) for member in members}
                edge_clusters[str(representative)] = member_set
                edge_member_map.update({member: str(representative) for member in member_set})

        if set(entity_member_map) != set(raw_entities) or set(edge_member_map) != set(raw_edges):
            raise ClusteringCollapseError("segmented KGGen clustering returned an incomplete map")
        relations = {
            (entity_member_map[subject], edge_member_map[predicate], entity_member_map[obj])
            for subject, predicate, obj in raw_relations
        }
        return Graph(
            entities=set(entity_clusters),
            edges=set(edge_clusters),
            relations=relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
        )

    @contextmanager
    def dspy_context(self, backend: Any):
        """Scope the optional local-vLLM adapter around a full KGGen call.

        KGGen creates nested ``dspy.context(lm=...)`` blocks itself.  DSPy's
        context is additive, so this outer block retains the adapter during
        raw extraction *and* official KGGen LLM clustering.
        """
        if self.structured_output.transport == "none":
            yield
            return
        import dspy

        from .dspy_adapter import strict_json_schema_adapter, vllm_guided_json_adapter

        lm = getattr(backend, "lm", None)
        if lm is None:
            raise RuntimeError(
                f"{self.structured_output.transport} requires a KGGen backend with a DSPy LM"
            )
        adapter = (
            strict_json_schema_adapter(
                request_backend=self.structured_output.request_backend
            )
            if self.structured_output.transport == "response_format"
            else vllm_guided_json_adapter()
        )
        with dspy.context(lm=lm, adapter=adapter):
            yield

    def _call_backend(
        self,
        text: str,
        *,
        cache_key: str | None = None,
        kind: str = "graph",
    ) -> Graph:
        backend = self._get_backend()
        with self.dspy_context(backend):
            if self.serial_chunking and len(text) > self.chunk_chars:
                # KGGen 0.4 implements ``chunk_size`` with an unbounded nested
                # ThreadPoolExecutor over one shared DSPy LM.  Combining that with
                # response-level parallelism can deadlock a local vLLM client after
                # some successful calls.  Preserve KGGen's algorithm (chunk,
                # aggregate, then cluster) but schedule the chunks serially.
                chunks = self._split_text(text, self.chunk_chars)
                graphs = []
                for index, chunk in enumerate(chunks, start=1):
                    self._emit_progress(
                        event="kg_chunk", kind=kind, completed=index - 1,
                        total=len(chunks), phase="extracting",
                    )
                    self._reserve_live_request(kind, "generating")
                    graphs.append(backend.generate(input_data=chunk, cluster=False))
                    self._emit_progress(
                        event="kg_chunk", kind=kind, completed=index,
                        total=len(chunks), phase="extracting",
                    )
                g = backend.aggregate(graphs)
                self._emit_progress(
                    event="kg_cluster", kind=kind, completed=len(chunks),
                    total=len(chunks), phase="clustering",
                )
                g = self._cluster_backend_graph(
                    backend,
                    g,
                    source_text=text,
                    cache_key=cache_key,
                    kind=kind,
                )
            else:
                # ``KGGen.generate(cluster=True)`` is implemented as raw extraction
                # followed by ``KGGen.cluster(graph)``.  Keep that exact upstream
                # cluster routine, but make the boundary explicit in the local
                # profile so a client-side stall can be distinguished from a vLLM
                # generation stall. A finite cap also needs raw triples first.
                explicit_cluster_phase = self.cluster and (
                    self.explicit_clustering
                    or self.cluster_max_items is not None
                    or self.cluster_min_retention_ratio is not None
                )
                gen_kwargs: dict[str, Any] = {
                    "input_data": text,
                    "cluster": False if explicit_cluster_phase else self.cluster,
                }
                if self.cluster and not explicit_cluster_phase:
                    gen_kwargs["context"] = self._cluster_context(text)
                if len(text) > self.chunk_chars:
                    gen_kwargs["chunk_size"] = self.chunk_chars
                self._reserve_live_request(kind, "generating")
                g = backend.generate(**gen_kwargs)
                if explicit_cluster_phase:
                    self._emit_progress(
                        event="kg_cluster", kind=kind, completed=0, total=1,
                        phase="clustering",
                    )
                    g = self._cluster_backend_graph(
                        backend,
                        g,
                        source_text=text,
                        cache_key=cache_key,
                        kind=kind,
                    )
        entities = {str(e) for e in getattr(g, "entities", set())}
        relations = {
            tuple(str(x) for x in r)
            for r in getattr(g, "relations", set())
            if isinstance(r, (list, tuple)) and len(r) == 3
        }
        return Graph(entities, relations)

    def _call_backend_with_transient_retries(
        self, text: str, *, cache_key: str, kind: str, max_tokens: int | None = None
    ) -> Graph:
        """Make one schema-valid graph attempt, retrying transient provider errors.

        Protocol failures are deliberately handled by ``extract`` rather than
        this loop: a malformed completion is not a transport failure, and it
        must stay bounded even when ``llm.max_retries=0`` means retry transient
        429/5xx errors until the enclosing Job deadline.
        """
        graph: Graph | None = None
        retry_heartbeat = RetryHeartbeat(
            kind,
            self.usage,
            progress_callback=lambda payload: self._emit_progress(**payload),
        )
        for attempt in Retrying(
            # ``0`` leaves non-capacity transient retries to the enclosing
            # DataSphere Job. A continuous 429 streak has an explicit local
            # deadline; completed graph calls remain atomic and resumable.
            stop=(
                StopAfterAttemptsExceptRateLimit(
                    None if self.max_retries == 0 else self.max_retries,
                    rate_limit_retry_deadline_seconds=self.rate_limit_retry_deadline_s,
                    retry_deadline_seconds=self.retry_deadline_s,
                )
            ),
            # Honour gateway Retry-After where supplied and otherwise use
            # bounded full-jitter exponential backoff to avoid a quota herd.
            wait=WaitRetryAfterOrExponentialJitter(
                self.backoff_base,
                self.backoff_max,
                rate_limit_cooldown_max_seconds=self.rate_limit_cooldown_max_s,
                rate_limit_retry_deadline_seconds=self.rate_limit_retry_deadline_s,
                retry_deadline_seconds=self.retry_deadline_s,
            ),
            retry=retry_if_exception(is_retryable_llm_exception),
            before_sleep=retry_heartbeat,
            reraise=True,
        ):
            with attempt:
                if self.debug_dump_after_s is not None:
                    # This is intentionally timer-based rather than SIGUSR1:
                    # the prior external signal dump occasionally segfaulted
                    # Pydantic while we were diagnosing the local runtime.
                    faulthandler.dump_traceback_later(
                        self.debug_dump_after_s, repeat=False, exit=False
                    )
                try:
                    with self._temporary_token_budget(max_tokens):
                        graph = self._call_backend(text, cache_key=cache_key, kind=kind)
                finally:
                    if self.debug_dump_after_s is not None:
                        faulthandler.cancel_dump_traceback_later()
        # With reraise=True, we only reach here on success, so graph is bound.
        assert graph is not None
        return graph

    def extract(self, text: str, kind: str = "graph") -> Graph:
        """Extract a graph from a single text, using the disk cache.

        Returns an empty graph (no API call) for empty/whitespace text -- this is how
        empty queries (Data2txt / Summary) yield G_q = empty.
        """
        text = (text or "").strip()
        if not text:
            return Graph.empty()

        key = self._cache_key(text)
        cached = self._load_cache(key)
        if cached is not None:
            self.usage.record_call(kind, key, 0.0, cached=True)
            return cached
        if self.cache_only:
            raise CacheOnlyMissError(kind, key, self._cache_path(key))

        start = time.perf_counter()
        protocol_attempt = 0
        token_budget = self.max_tokens
        while True:
            try:
                graph = self._call_backend_with_transient_retries(
                    text, cache_key=key, kind=kind, max_tokens=token_budget
                )
                break
            except StructuredOutputTruncatedError as exc:
                if (
                    token_budget is not None
                    and self.max_tokens_ceiling is not None
                    and token_budget < self.max_tokens_ceiling
                ):
                    next_budget = min(token_budget * 2, self.max_tokens_ceiling)
                    protocol_attempt += 1
                    self.usage.record_retry(kind, exc)
                    print(
                        "[kg] token-budget retry "
                        f"kind={kind} max_tokens={next_budget}",
                        flush=True,
                    )
                    token_budget = next_budget
                    continue
                if protocol_attempt >= self.max_protocol_retries:
                    raise
                protocol_attempt += 1
                self.usage.record_retry(kind, exc)
                print(
                    "[kg] structured-output retry "
                    f"kind={kind} attempt={protocol_attempt}/{self.max_protocol_retries}",
                    flush=True,
                )
                ceiling = min(
                    self.backoff_max,
                    float(self.backoff_base) * (2 ** (protocol_attempt - 1)),
                )
                if ceiling > 0:
                    time.sleep(random.uniform(0, ceiling))
            except (StructuredOutputParseError, StructuredOutputSchemaError) as exc:
                if protocol_attempt >= self.max_protocol_retries:
                    raise
                protocol_attempt += 1
                self.usage.record_retry(kind, exc)
                # This is a fresh native-schema request, never JSON repair and
                # never an acceptance of the malformed document. Keep retries
                # bounded and jittered so four isolated misses do not create a
                # quota herd after a long cache-backed resume.
                ceiling = min(
                    self.backoff_max,
                    float(self.backoff_base) * (2 ** (protocol_attempt - 1)),
                )
                print(
                    "[kg] structured-output retry "
                    f"kind={kind} attempt={protocol_attempt}/{self.max_protocol_retries}",
                    flush=True,
                )
                if ceiling > 0:
                    time.sleep(random.uniform(0, ceiling))
        elapsed = time.perf_counter() - start
        self._save_cache(key, graph)
        self.usage.record_call(kind, key, elapsed, cached=False)
        return graph

    def extract_reference(self, context: str, query: str | None) -> tuple[Graph, Graph]:
        """Extract (G_c, G_q). G_q is empty when query is empty."""
        g_c = self.extract(context, kind="context")
        g_q = self.extract(query or "", kind="query")
        return g_c, g_q


# --------------------------------------------------------------------------------------
# Offline fake backend (for tests and `run.py --fake-extractor` plumbing checks)
# --------------------------------------------------------------------------------------
class FakeKGGen:
    """A deterministic, dependency-free stand-in for kg_gen.KGGen.

    Extracts a toy graph from text so the full pipeline runs offline. It is NOT a real
    extractor -- only for exercising plumbing / cache determinism, never for real metrics.
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _tokens(text: str) -> list[str]:
        import re

        words = re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
        # keep capitalized-ish / longer tokens as pseudo entities
        return [w for w in words if len(w) >= 4]

    def generate(self, input_data, cluster=True, chunk_size=None, context=None):  # noqa: ARG002
        toks = self._tokens(input_data if isinstance(input_data, str) else str(input_data))
        uniq = list(dict.fromkeys(toks))[:12]
        rels = set()
        for a, b in zip(uniq, uniq[1:]):
            rels.add((a, "co_occurs_with", b))

        class _G:
            entities = set(uniq)
            relations = rels

        return _G()
