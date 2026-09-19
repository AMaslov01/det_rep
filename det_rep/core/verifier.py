"""Evidence selection and a cache-backed relation entailment verifier.

The verifier is deliberately separate from KG extraction: it receives only a
canonical triple plus short spans from the original context/query and returns
one of entailed, contradicted, or unknown.  Exhausted transient transport and
structured-output failures are persisted as explicitly marked ``unknown``
decisions: this keeps a long cache-backed run resumable without treating a
failed model call as positive evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tenacity import Retrying, retry_if_exception, stop_after_attempt

from .cache import CacheOnlyMissError, config_value, llm_runtime_fingerprint
from .config import resolve_api_key
from .dspy_adapter import (
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    is_retryable_llm_exception,
    json_schema_response_format,
    strict_json_loads,
    structured_output_settings,
    validate_gateway_identity,
    validate_json_document,
)
from .matching import normalize
from .retry import RetryHeartbeat, StopAfterAttemptsExceptRateLimit, WaitRetryAfterOrExponentialJitter


VERDICTS = frozenset({"entailed", "contradicted", "unknown"})
FALLBACK_REASONS = frozenset({"structured_output_exhausted", "transient_exhausted"})
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["entailed", "contradicted", "unknown"],
        }
    },
    "required": ["verdict"],
    "additionalProperties": False,
}
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_PREDICATE_STOPWORDS = {
    "a", "an", "the", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "has", "have", "had", "in", "is", "it", "of", "on",
    "or", "to", "was", "were", "with",
}


class RelationVerifierError(RuntimeError):
    """A verifier failure is distinct from the semantic ``unknown`` verdict."""


@dataclass(frozen=True)
class EvidenceSpan:
    source: str
    index: int
    start: int
    end: int
    text: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class RelationVerdict:
    verdict: str
    evidence: tuple[EvidenceSpan, ...]
    cache_hit: bool = False
    protocol_fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unsupported relation verdict {self.verdict!r}")
        if self.protocol_fallback != (self.fallback_reason is not None):
            raise ValueError("relation fallback provenance must be explicit")
        if self.fallback_reason is not None:
            if self.verdict != "unknown" or self.fallback_reason not in FALLBACK_REASONS:
                raise ValueError("relation fallback must be an allowed unknown verdict")


def select_evidence(
    context: str,
    query: str | None,
    canonical_triple: tuple[str, str, str],
    *,
    max_sentences: int = 4,
    stopwords: Iterable[str] = (),
) -> list[EvidenceSpan]:
    """Pick up to ``max_sentences`` deterministic evidence spans.

    Rank = 2*subject_phrase + 2*object_phrase + predicate-token hits.  The
    predicate tokens only retrieve evidence; the LLM decides entailment.
    """
    subject, predicate, obj = (normalize(x) for x in canonical_triple)
    blocked = set(stopwords) | _PREDICATE_STOPWORDS
    predicate_tokens = [
        token for token in _WORD_RE.findall(predicate)
        if token and token not in blocked
    ]
    spans: list[EvidenceSpan] = []
    for source, text in (("context", context or ""), ("query", query or "")):
        for index, start, end, sentence in _sentences(text):
            norm_sentence = normalize(sentence)
            subject_hit = int(_contains_phrase(norm_sentence, subject))
            object_hit = int(_contains_phrase(norm_sentence, obj))
            predicate_hits = sum(int(_contains_token(norm_sentence, token)) for token in predicate_tokens)
            rank = 2 * subject_hit + 2 * object_hit + predicate_hits
            if rank:
                spans.append(EvidenceSpan(source, index, start, end, sentence, rank))
    # Context precedes query for a stable, document-order tie-breaker.
    spans.sort(key=lambda s: (-s.rank, 0 if s.source == "context" else 1, s.start, s.index))
    return spans[: max(0, int(max_sentences))]


def _sentences(text: str) -> list[tuple[int, int, int, str]]:
    out: list[tuple[int, int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(text):
        out.extend(_append_sentence(text, cursor, match.start(), len(out)))
        cursor = match.end()
    out.extend(_append_sentence(text, cursor, len(text), len(out)))
    return out


def _append_sentence(text: str, start: int, end: int, index: int) -> list[tuple[int, int, int, str]]:
    piece = text[start:end]
    left = len(piece) - len(piece.lstrip())
    right = len(piece.rstrip())
    if right <= left:
        return []
    actual_start = start + left
    actual_end = start + right
    return [(index, actual_start, actual_end, text[actual_start:actual_end])]


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _contains_token(text: str, token: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", text))


class RelationVerifier:
    """Direct LiteLLM verifier that shares the pipeline's single ``llm.model``."""

    def __init__(self, cfg, usage=None, *, cache_only: bool = False):
        verifier_cfg = getattr(cfg, "relation_verifier", None)
        if verifier_cfg is None:
            raise ValueError("relation_verifier config is required for support scoring")
        self.cfg = cfg
        self.model = cfg.llm.model
        self.api_base = getattr(cfg.llm, "api_base", None)
        self.structured_output = structured_output_settings(cfg.llm)
        self.temperature = float(cfg.llm.temperature)
        # Keep verifier and extractor on the same configured output budget.  A
        # previous hard-coded 32-token limit bypassed the CPU Vertex profile's
        # ``llm.max_tokens=1024`` and can leave Gemini 2.5 Flash no room for a
        # final structured response after its internal reasoning.
        configured_max_tokens = getattr(cfg.llm, "max_tokens", None)
        self.max_tokens = 128 if configured_max_tokens is None else int(configured_max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("llm.max_tokens must be positive when configured")
        self.max_retries = int(cfg.llm.max_retries)
        if self.max_retries < 0:
            raise ValueError("llm.max_retries must be non-negative")
        self.backoff_base = float(cfg.llm.retry_backoff_base_s)
        self.backoff_max = float(getattr(cfg.llm, "retry_backoff_max_s", 60))
        if self.backoff_max < self.backoff_base:
            raise ValueError("llm.retry_backoff_max_s must be at least retry_backoff_base_s")
        self.rate_limit_cooldown_max_s = float(
            getattr(cfg.llm, "rate_limit_cooldown_max_s", 900)
        )
        self.rate_limit_retry_deadline_s = float(
            getattr(cfg.llm, "rate_limit_retry_deadline_s", 1800)
        )
        self.retry_deadline_s = float(getattr(cfg.llm, "retry_deadline_s", 90))
        if self.rate_limit_cooldown_max_s < self.backoff_max:
            raise ValueError(
                "llm.rate_limit_cooldown_max_s must be at least retry_backoff_max_s"
            )
        if self.rate_limit_retry_deadline_s <= 0:
            raise ValueError("llm.rate_limit_retry_deadline_s must be positive")
        if self.retry_deadline_s <= 0:
            raise ValueError("llm.retry_deadline_s must be positive")
        self.request_timeout_s = float(getattr(cfg.llm, "request_timeout_s", 90))
        if self.request_timeout_s <= 0:
            raise ValueError("llm.request_timeout_s must be positive")
        self.max_sentences = int(verifier_cfg.max_evidence_sentences)
        self.max_protocol_retries = int(config_value(verifier_cfg, "max_protocol_retries", 3))
        if self.max_protocol_retries < 1:
            raise ValueError("relation_verifier.max_protocol_retries must be positive")
        self.prompt_version = str(verifier_cfg.prompt_version)
        self.stopwords = set(getattr(cfg.matching, "stopwords", []) or [])
        self.cache_dir = Path(verifier_cfg.cache_dir)
        raw_read_dirs = config_value(verifier_cfg, "cache_read_dirs", []) or []
        self.cache_read_dirs = [Path(str(path)) for path in raw_read_dirs]
        self.cache_only = bool(cache_only)
        if not self.cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = usage

    def verify(
        self,
        canonical_triple: tuple[str, str, str],
        context: str,
        query: str | None,
        *,
        matching_params: dict[str, Any] | None = None,
    ) -> RelationVerdict:
        evidence = select_evidence(
            context, query, canonical_triple,
            max_sentences=self.max_sentences,
            stopwords=self.stopwords,
        )
        if not evidence:
            return RelationVerdict("unknown", tuple(), cache_hit=False)
        key = self._cache_key(canonical_triple, evidence, matching_params or {})
        legacy_key = self._legacy_cache_key(canonical_triple, evidence, matching_params or {})
        cached = self._load_cache(key, legacy_key)
        if cached is not None:
            if self.usage is not None:
                self.usage.record_call("relation_verifier", key, 0.0, cached=True)
            verdict, protocol_fallback, fallback_reason, cache_key = cached
            # Earlier cache entries unnecessarily included matching thresholds
            # even though the LLM receives only the already-canonical triple
            # and selected text evidence. Promote a compatible old entry only
            # during a live/cache-fill run; a cache-only acceptance replay must
            # leave every cache inventory byte-for-byte unchanged.
            if cache_key != key and not self.cache_only:
                self._save_cache(
                    key,
                    verdict,
                    protocol_fallback=protocol_fallback,
                    fallback_reason=fallback_reason,
                )
            return RelationVerdict(
                verdict,
                tuple(evidence),
                cache_hit=True,
                protocol_fallback=protocol_fallback,
                fallback_reason=fallback_reason,
            )
        if self.cache_only:
            raise CacheOnlyMissError(
                "relation_verifier", key, self.cache_dir / f"{key}.json"
            )

        start = time.perf_counter()
        try:
            verdict, protocol_fallback, fallback_reason = self._resolve_verdict(
                canonical_triple, evidence
            )
        except Exception as exc:  # noqa: BLE001
            raise RelationVerifierError(f"relation verifier failed for {canonical_triple!r}") from exc
        elapsed = time.perf_counter() - start
        self._save_cache(
            key,
            verdict,
            protocol_fallback=protocol_fallback,
            fallback_reason=fallback_reason,
        )
        if self.usage is not None:
            self.usage.record_call("relation_verifier", key, elapsed, cached=False)
        return RelationVerdict(
            verdict,
            tuple(evidence),
            cache_hit=False,
            protocol_fallback=protocol_fallback,
            fallback_reason=fallback_reason,
        )

    def _resolve_verdict(
        self,
        canonical_triple: tuple[str, str, str],
        evidence: list[EvidenceSpan],
    ) -> tuple[str, bool, str | None]:
        """Return a scalar verdict without letting one bad response kill a Job.

        Transport retries and malformed-output retries are intentionally
        separate.  ``429`` remains unlimited through the transport retry
        policy; a timeout/5xx is bounded.  Neither class ever becomes
        ``entailed`` merely because the provider did not return a valid answer.
        """
        for protocol_attempt in range(self.max_protocol_retries):
            try:
                return self._call_with_transient_retries(canonical_triple, evidence), False, None
            except (StructuredOutputParseError, StructuredOutputSchemaError) as exc:
                if protocol_attempt + 1 >= self.max_protocol_retries:
                    return "unknown", True, "structured_output_exhausted"
                if self.usage is not None:
                    self.usage.record_retry("relation_verifier", exc)
                ceiling = min(
                    self.backoff_max,
                    self.backoff_base * (2 ** protocol_attempt),
                )
                if ceiling > 0:
                    time.sleep(random.uniform(0.0, ceiling))
            except Exception as exc:  # noqa: BLE001
                # A finite transient retry budget protects against requests
                # which may have reached the upstream model but timed out.
                # Authentication, model, request, and other deterministic
                # configuration errors remain hard failures.
                if is_retryable_llm_exception(exc):
                    return "unknown", True, "transient_exhausted"
                raise
        raise AssertionError("unreachable relation verifier protocol state")

    def _call_with_transient_retries(
        self,
        canonical_triple: tuple[str, str, str],
        evidence: list[EvidenceSpan],
    ) -> str:
        verdict: str | None = None
        for attempt in Retrying(
            # ``0`` delegates non-capacity transient retries to the enclosing
            # Job; a continuous 429 streak has an explicit local deadline.
            stop=(
                StopAfterAttemptsExceptRateLimit(
                    None if self.max_retries == 0 else self.max_retries,
                    rate_limit_retry_deadline_seconds=self.rate_limit_retry_deadline_s,
                    retry_deadline_seconds=self.retry_deadline_s,
                )
            ),
            wait=WaitRetryAfterOrExponentialJitter(
                self.backoff_base,
                self.backoff_max,
                rate_limit_cooldown_max_seconds=self.rate_limit_cooldown_max_s,
                rate_limit_retry_deadline_seconds=self.rate_limit_retry_deadline_s,
                retry_deadline_seconds=self.retry_deadline_s,
            ),
            retry=retry_if_exception(is_retryable_llm_exception),
            before_sleep=RetryHeartbeat("relation_verifier", self.usage),
            reraise=True,
        ):
            with attempt:
                verdict = self._call_llm(canonical_triple, evidence)
        assert verdict is not None
        return verdict

    def _call_llm(self, triple: tuple[str, str, str], evidence: list[EvidenceSpan]) -> str:
        try:
            from litellm import completion  # type: ignore
        except Exception as exc:  # pragma: no cover - live-only dependency
            raise RelationVerifierError("litellm is required for relation verification") from exc
        evidence_text = "\n".join(
            f"[{span.source}:{span.index}] {span.text}" for span in evidence
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify whether the claim follows only from the supplied evidence. "
                    "Do not use outside knowledge. Return JSON only: "
                    '{"verdict":"entailed"|"contradicted"|"unknown"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Claim:\n"
                    f"({triple[0]}, {triple[1]}, {triple[2]})\n\n"
                    f"Evidence:\n{evidence_text}"
                ),
            },
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.request_timeout_s,
            # Tenacity above is the verifier's only retry policy.  LiteLLM's
            # OpenAI client otherwise applies its own default retries, making
            # the configured attempt bound and usage accounting inaccurate.
            "num_retries": 0,
        }
        api_key = resolve_api_key(self.cfg)
        if api_key:
            kwargs["api_key"] = api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.structured_output.transport == "response_format":
            kwargs["response_format"] = json_schema_response_format(
                VERDICT_SCHEMA, name="relation_verdict"
            )
            if self.structured_output.request_backend is not None:
                kwargs["extra_body"] = {
                    "guided_decoding_backend": self.structured_output.request_backend
                }
        elif self.structured_output.transport == "guided_json":
            # Deprecated compatibility mode for artifacts created by the old
            # vLLM 0.6 runtime.  New research Jobs use response_format.
            kwargs["extra_body"] = {"guided_json": VERDICT_SCHEMA}
        response = completion(**kwargs)
        validate_gateway_identity(response, label="relation verifier completion")
        choices = _response_field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise StructuredOutputParseError(
                "verifier completion must contain exactly one choice"
            )
        choice = choices[0]
        finish_reason = _response_field(choice, "finish_reason")
        if finish_reason != "stop":
            raise StructuredOutputParseError(
                f"verifier completion did not finish cleanly: {finish_reason!r}"
            )
        message = _response_field(choice, "message")
        content = _response_field(message, "content")
        return _parse_verdict(content)

    def _cache_key(
        self, triple: tuple[str, str, str], evidence: list[EvidenceSpan], matching_params: dict[str, Any]
    ) -> str:
        """Key a textual verdict by the inputs actually sent to the LLM.

        ``matching_params`` determines which answer edge reaches this method,
        but it has already done so through the canonical triple. It never
        changes the evidence prompt or the returned textual entailment label,
        so putting it in this cache key caused duplicate paid calls across the
        train-only threshold grid.
        """
        payload = {
            "v": 5,
            "verifier_protocol": "relation-entailment-closed-schema-v4-matching-independent",
            "llm": llm_runtime_fingerprint(self.cfg),
            "api_base": self.api_base,
            "prompt_version": self.prompt_version,
            "max_tokens": self.max_tokens,
            "max_evidence_sentences": self.max_sentences,
            "embedding_model": config_value(self.cfg.matching, "embedding_model"),
            "embedding_model_revision": config_value(
                self.cfg.matching, "embedding_model_revision"
            ),
            "triple": list(triple),
            "evidence": [span.to_dict() for span in evidence],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _legacy_cache_key(
        self,
        triple: tuple[str, str, str],
        evidence: list[EvidenceSpan],
        matching_params: dict[str, Any],
    ) -> str:
        """Locate v4 entries produced before matching-independent verdict keys."""
        payload = {
            "v": 4,
            "verifier_protocol": "relation-entailment-closed-schema-v3",
            "llm": llm_runtime_fingerprint(self.cfg),
            "api_base": self.api_base,
            "prompt_version": self.prompt_version,
            "max_tokens": self.max_tokens,
            "max_evidence_sentences": self.max_sentences,
            "embedding_model": config_value(self.cfg.matching, "embedding_model"),
            "embedding_model_revision": config_value(
                self.cfg.matching, "embedding_model_revision"
            ),
            "matching": matching_params,
            "triple": list(triple),
            "evidence": [span.to_dict() for span in evidence],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_cache(
        self,
        key: str,
        *compatible_legacy_keys: str,
    ) -> tuple[str, bool, str | None, str] | None:
        # Historical support-verdict namespaces are immutable read-through
        # inputs.  A malformed primary entry must not hide a compatible,
        # validated verdict in a later root.
        for root in [self.cache_dir, *self.cache_read_dirs]:
            for candidate_key in (key, *compatible_legacy_keys):
                try:
                    payload = json.loads(
                        (root / f"{candidate_key}.json").read_text(encoding="utf-8")
                    )
                    verdict = payload.get("verdict")
                    if verdict not in VERDICTS:
                        continue
                    protocol_fallback = bool(payload.get("_hallu_protocol_fallback", False))
                    fallback_reason = payload.get("_hallu_fallback_reason")
                    if protocol_fallback:
                        if fallback_reason not in FALLBACK_REASONS or verdict != "unknown":
                            continue
                        return str(verdict), True, str(fallback_reason), candidate_key
                    if fallback_reason is not None:
                        continue
                    return str(verdict), False, None, candidate_key
                except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                    continue
        return None

    def _save_cache(
        self,
        key: str,
        verdict: str,
        *,
        protocol_fallback: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        if protocol_fallback != (fallback_reason is not None):
            raise ValueError("relation cache fallback provenance must be explicit")
        if protocol_fallback and (verdict != "unknown" or fallback_reason not in FALLBACK_REASONS):
            raise ValueError("relation fallback cache must be an allowed unknown verdict")
        dest = self.cache_dir / f"{key}.json"
        tmp = dest.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        payload: dict[str, Any] = {"verdict": verdict}
        if protocol_fallback:
            payload["_hallu_protocol_fallback"] = True
            payload["_hallu_fallback_reason"] = fallback_reason
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        os.replace(tmp, dest)


class FakeRelationVerifier:
    """Deterministic verifier for fully offline tests and fake pipeline runs."""

    def __init__(self, verdicts: dict[tuple[str, str, str], str] | None = None, default: str = "unknown"):
        self.verdicts = {tuple(normalize(x) for x in k): v for k, v in (verdicts or {}).items()}
        if default not in VERDICTS:
            raise ValueError(default)
        self.default = default
        self.calls = 0

    def verify(
        self,
        canonical_triple: tuple[str, str, str],
        context: str,
        query: str | None,
        *,
        matching_params: dict[str, Any] | None = None,  # noqa: ARG002 - interface parity
    ) -> RelationVerdict:
        self.calls += 1
        triple = tuple(normalize(x) for x in canonical_triple)
        evidence = select_evidence(context, query, triple)
        return RelationVerdict(self.verdicts.get(triple, self.default), tuple(evidence), cache_hit=False)


def _parse_verdict(content: Any) -> str:
    if not isinstance(content, str):
        raise StructuredOutputParseError("verifier response has no text content")
    payload = strict_json_loads(content.strip(), label="verifier response")
    try:
        validate_json_document(payload, VERDICT_SCHEMA)
    except StructuredOutputSchemaError:
        raise
    verdict = payload["verdict"]
    return str(verdict)


def _response_field(value: Any, field: str) -> Any:
    """Read a LiteLLM/OpenAI response field without accepting malformed shapes."""
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)
