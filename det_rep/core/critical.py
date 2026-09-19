"""Claim-level evidence checks used by the ``support-critical`` detector.

This module intentionally does not alter the historical relation verifier.
It adds a separate, versioned closed-world protocol in which a clear factual
claim that has no direct textual support is ``unsupported`` rather than the
legacy verifier's catch-all ``unknown``.  Every LLM-backed component is cached
independently so a completed critical run can be replayed fully offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
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
from .matching import Embedder, normalize
from .retry import (
    RequestPacer,
    RetryHeartbeat,
    StopAfterAttemptsExceptRateLimit,
    WaitRetryAfterOrExponentialJitter,
)
from .verifier import EvidenceSpan, _sentences


CRITICAL_VERDICTS = frozenset({"entailed", "unknown", "unsupported", "contradicted"})
CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
                "required": ["text", "start", "end"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": sorted(CRITICAL_VERDICTS)},
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "has", "have", "had", "in", "is", "it", "of",
    "on", "or", "the", "to", "was", "were", "with",
}
_T = TypeVar("_T")


class CriticalProtocolError(RuntimeError):
    """Raised when a critical component cannot produce its strict artifact."""


class CriticalCompletionTruncatedError(StructuredOutputParseError):
    """A schema response was cut off by the provider's output-token ceiling."""


class CriticalRetryableTruncation(CriticalCompletionTruncatedError):
    """A token-ceiling retry remains available for this otherwise strict response."""


class CriticalOutputLimitError(StructuredOutputParseError):
    """A list response needs deterministic input segmentation, not another retry.

    Retrying the same request after Gemini has exhausted the configured output
    ceiling cannot make its JSON array shorter.  List-producing components
    catch this signal and bisect the answer while preserving absolute offsets.
    Scalar verdict components intentionally do not catch it.
    """


@dataclass(frozen=True)
class AtomicClaim:
    text: str
    start: int
    end: int
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end, "sources": list(self.sources)}


@dataclass(frozen=True)
class CriticalVerdict:
    verdict: str
    evidence: tuple[EvidenceSpan, ...]
    cache_hit: bool = False
    protocol_fallback: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in CRITICAL_VERDICTS:
            raise ValueError(f"unsupported critical verdict {self.verdict!r}")
        if self.protocol_fallback != (self.fallback_reason is not None):
            raise ValueError("critical fallback provenance must be explicit")
        if self.fallback_reason is not None and self.verdict != "unknown":
            raise ValueError("critical fallback must be an unknown verdict")


def claim_risk(verdict: str | None, unknown_risk: float) -> float:
    """Map a final claim verdict to the critical detector's fixed risk scale."""
    if verdict == "entailed":
        return 0.0
    if verdict in {"unsupported", "contradicted"}:
        return 1.0
    if verdict == "unknown":
        return float(unknown_risk)
    # Ungrounded graph edges are strict candidates without a textual verdict.
    return 1.0


def merge_claims(*collections: Iterable[AtomicClaim]) -> list[AtomicClaim]:
    """Deduplicate exact response spans while retaining candidate provenance."""
    merged: dict[tuple[int, int, str], AtomicClaim] = {}
    for collection in collections:
        for claim in collection:
            key = (int(claim.start), int(claim.end), normalize(claim.text))
            if not key[2]:
                continue
            previous = merged.get(key)
            if previous is None:
                merged[key] = claim
                continue
            merged[key] = replace(
                previous,
                sources=tuple(sorted(set(previous.sources) | set(claim.sources))),
            )
    return sorted(merged.values(), key=lambda c: (c.start, c.end, normalize(c.text)))


@dataclass(frozen=True)
class _AnswerChunk:
    """An exact contiguous slice of an answer, addressed in answer offsets."""

    text: str
    start: int
    end: int


def _answer_chunks(text: str, max_chars: int) -> list[_AnswerChunk]:
    """Split an answer deterministically, retaining every character exactly once.

    We prefer a sentence boundary in the latter half of a window, then a
    whitespace boundary, and use a hard boundary only as a final fallback.
    This avoids a single pathological long answer producing an unbounded JSON
    claim list.  Chunks neither overlap nor drop whitespace, which makes their
    local offsets safely convertible to the original answer offsets.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [_AnswerChunk(text=text, start=0, end=len(text))]

    chunks: list[_AnswerChunk] = []
    start = 0
    length = len(text)
    while start < length:
        target = min(length, start + max_chars)
        if target == length:
            end = target
        else:
            floor = start + max(1, (target - start) // 2)
            window = text[start:target]
            # Include separator whitespace in the prior segment.  It gives
            # the model natural sentence-shaped input without changing text.
            sentence_ends = [
                start + match.end()
                for match in re.finditer(r"(?:[.!?;:][\"'\)\]\}]*)(?:\s+)|\n+", window)
                if start + match.end() >= floor
            ]
            whitespace_ends = [
                start + match.end()
                for match in re.finditer(r"\s+", window)
                if start + match.end() >= floor
            ]
            end = (sentence_ends or whitespace_ends or [target])[-1]
        if end <= start:  # Defensive: a malformed boundary must not loop.
            end = min(length, start + max_chars)
        chunks.append(_AnswerChunk(text=text[start:end], start=start, end=end))
        start = end
    return chunks


def _bisect_chunk(chunk: _AnswerChunk, min_chars: int) -> list[_AnswerChunk] | None:
    """Return smaller answer-offset chunks, or ``None`` at the safety floor."""
    if min_chars <= 0:
        raise ValueError("min_chars must be positive")
    if len(chunk.text) <= min_chars:
        return None
    local = _answer_chunks(chunk.text, max(min_chars, (len(chunk.text) + 1) // 2))
    if len(local) < 2:
        return None
    return [
        _AnswerChunk(text=piece.text, start=chunk.start + piece.start, end=chunk.start + piece.end)
        for piece in local
    ]


def _literal_spans(response: str, text: str) -> list[tuple[int, int]]:
    """Return every literal occurrence without guessing semantic content."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    offset = 0
    while True:
        start = response.find(text, offset)
        if start < 0:
            return spans
        spans.append((start, start + len(text)))
        offset = start + 1


def _repair_claim_span(response: str, text: str) -> tuple[str, int, int] | None:
    """Recover only a mechanical offset error for a unique literal fragment.

    The model still supplies the claim text. This helper merely locates that
    same text in the answer, never paraphrases it, and never guesses between
    repeated occurrences. The stored audit remains byte-exact and checkable.
    """
    candidates: set[tuple[int, int]] = set(_literal_spans(response, text))
    stripped = text.strip()
    if stripped and stripped != text:
        candidates.update(_literal_spans(response, stripped))
    if not candidates:
        # Newline-versus-space is a serialisation error, not a semantic edit.
        # Persist the original answer slice only when this match is unique.
        tokens = stripped.split()
        if tokens:
            pattern = r"\s+".join(re.escape(token) for token in tokens)
            candidates.update((match.start(), match.end()) for match in re.finditer(pattern, response))
    if len(candidates) != 1:
        return None
    start, end = next(iter(candidates))
    return response[start:end], start, end


def _validate_claims(payload: Any, response: str, source: str) -> list[AtomicClaim]:
    try:
        validate_json_document(payload, CLAIM_SCHEMA)
    except StructuredOutputSchemaError:
        raise
    claims: list[AtomicClaim] = []
    for raw in payload["claims"]:
        start, end, text = int(raw["start"]), int(raw["end"]), str(raw["text"])
        if not (0 <= start < end <= len(response)) or response[start:end] != text:
            repaired = _repair_claim_span(response, text)
            if repaired is None:
                raise StructuredOutputParseError(
                    f"{source} claim does not identify one exact response substring"
                )
            text, start, end = repaired
        claims.append(AtomicClaim(text=text, start=start, end=end, sources=(source,)))
    return merge_claims(claims)


_FALLBACK_SPANS_FIELD = "_hallu_fallback_sentence_spans"


def _sentence_fallback_claims(response: str, source: str) -> list[AtomicClaim]:
    """Create exact, conservative claim candidates without an LLM response.

    This is a continuity mechanism for a provider that repeatedly violates the
    JSON/offset contract.  It does not invent factual content: each candidate
    is a verbatim non-empty answer sentence (or line).  The ordinary strict
    evidence verifier still decides its four-way verdict.  Thus a malformed
    model artifact cannot abort the entire experiment or silently become
    ``entailed``.
    """
    return [
        AtomicClaim(text=text, start=start, end=end, sources=(source,))
        for _, start, end, text in _sentences(response)
    ]


def _claim_cache_payload(claims: Sequence[AtomicClaim]) -> dict[str, Any]:
    """Serialize claims while retaining deterministic-fallback provenance."""
    payload: dict[str, Any] = {
        "claims": [{"text": claim.text, "start": claim.start, "end": claim.end} for claim in claims]
    }
    fallback_spans = sorted({
        (claim.start, claim.end)
        for claim in claims
        if any(source.endswith("_fallback_sentence") for source in claim.sources)
    })
    if fallback_spans:
        # This private cache field is intentionally not part of the LLM schema.
        # It preserves audit provenance across cache-only replay.
        payload[_FALLBACK_SPANS_FIELD] = [[start, end] for start, end in fallback_spans]
    return payload


def _claims_from_cache(payload: dict[str, Any], response: str, source: str) -> list[AtomicClaim]:
    """Validate a claim cache entry and restore any fallback provenance."""
    core = {"claims": payload.get("claims")}
    claims = _validate_claims(core, response, source)
    raw_spans = payload.get(_FALLBACK_SPANS_FIELD, [])
    if not isinstance(raw_spans, list):
        raise StructuredOutputParseError("critical claim cache has malformed fallback spans")
    fallback_spans: set[tuple[int, int]] = set()
    for raw in raw_spans:
        if not isinstance(raw, list) or len(raw) != 2:
            raise StructuredOutputParseError("critical claim cache has malformed fallback span")
        fallback_spans.add((int(raw[0]), int(raw[1])))
    if not fallback_spans:
        return claims
    fallback_source = f"{source}_fallback_sentence"
    return [
        replace(claim, sources=(fallback_source,))
        if (claim.start, claim.end) in fallback_spans else claim
        for claim in claims
    ]


def _response_field(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


def _validated_critical_verdict(payload: dict[str, Any]) -> str:
    validate_json_document(payload, VERDICT_SCHEMA)
    return str(payload["verdict"])


class _CachedComponent:
    """Common cache, transport, and retry contract for critical components."""

    component: str = "critical_component"
    protocol: str = "support-critical-v1"

    def __init__(
        self,
        cfg,
        section_name: str,
        usage=None,
        *,
        cache_only: bool = False,
        request_pacer: RequestPacer | None = None,
    ):
        critical_cfg = getattr(cfg, "support_critical", None)
        if critical_cfg is None:
            raise ValueError("support_critical config is required")
        section = getattr(critical_cfg, section_name, None)
        if section is None:
            raise ValueError(f"support_critical.{section_name} config is required")
        self.cfg = cfg
        self.section = section
        self.model = cfg.llm.model
        self.api_base = getattr(cfg.llm, "api_base", None)
        self.structured_output = structured_output_settings(cfg.llm)
        self.temperature = float(cfg.llm.temperature)
        self.max_tokens = int(config_value(section, "max_tokens", 512))
        # Gemini can consume output budget on hidden reasoning before emitting a
        # tiny JSON object.  Keep the configured value as the first attempt,
        # then make a bounded transport retry on ``finish_reason=length``.
        self.max_tokens_ceiling = int(
            config_value(section, "max_tokens_ceiling", max(self.max_tokens, 8192))
        )
        self.max_retries = int(cfg.llm.max_retries)
        self.max_protocol_retries = int(config_value(section, "max_protocol_retries", 4))
        self.backoff_base = float(cfg.llm.retry_backoff_base_s)
        self.backoff_max = float(getattr(cfg.llm, "retry_backoff_max_s", 60))
        self.request_timeout_s = float(getattr(cfg.llm, "request_timeout_s", 90))
        self.rate_limit_cooldown_max_s = float(
            getattr(cfg.llm, "rate_limit_cooldown_max_s", 900)
        )
        self.rate_limit_retry_deadline_s = float(
            getattr(cfg.llm, "rate_limit_retry_deadline_s", 1800)
        )
        self.retry_deadline_s = float(getattr(cfg.llm, "retry_deadline_s", 1800))
        self.request_min_interval_s = float(getattr(cfg.llm, "request_min_interval_s", 0))
        self.prompt_version = str(config_value(section, "prompt_version", "v1"))
        self.cache_dir = Path(str(config_value(section, "cache_dir")))
        raw_read_dirs = config_value(section, "cache_read_dirs", []) or []
        self.cache_read_dirs = [Path(str(path)) for path in raw_read_dirs]
        self.cache_only = bool(cache_only)
        self.usage = usage
        if (
            self.max_tokens <= 0
            or self.max_retries < 0
            or self.request_timeout_s <= 0
            or self.rate_limit_cooldown_max_s < self.backoff_max
            or self.rate_limit_retry_deadline_s <= 0
            or self.retry_deadline_s <= 0
            or self.request_min_interval_s < 0
        ):
            raise ValueError(f"invalid {self.component} runtime limits")
        if self.max_protocol_retries <= 0:
            raise ValueError(f"{self.component}.max_protocol_retries must be positive")
        if self.max_tokens_ceiling < self.max_tokens:
            raise ValueError(f"{self.component}.max_tokens_ceiling must be at least max_tokens")
        if self.backoff_max < self.backoff_base:
            raise ValueError("llm.retry_backoff_max_s must be at least retry_backoff_base_s")
        self.request_pacer = request_pacer or RequestPacer(self.request_min_interval_s)
        if not self.cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        envelope = {
            "protocol": self.protocol,
            "component": self.component,
            "prompt_version": self.prompt_version,
            # The key contains the initial request budget.  The ceiling is a
            # transport-only recovery limit: a completed response produced on
            # the first attempt remains valid after a code update adds bounded
            # retries, so this deliberately preserves durable partial caches.
            "max_tokens": self.max_tokens,
            "llm": llm_runtime_fingerprint(self.cfg),
            "api_base": self.api_base,
            "payload": payload,
        }
        return hashlib.sha256(
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _load(self, key: str) -> dict[str, Any] | None:
        # The primary root is writable for the active protocol namespace;
        # prior roots are read-through only. This lets a retrying Job reuse
        # completed claim/verdict artifacts after its source commit changes.
        for root in [self.cache_dir, *self.cache_read_dirs]:
            try:
                payload = json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                continue
        return None

    def _save(self, key: str, payload: dict[str, Any]) -> None:
        dest = self.cache_dir / f"{key}.json"
        tmp = dest.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, dest)

    def _load_claims(self, key: str, response: str, source: str) -> list[AtomicClaim] | None:
        """Load a valid claim artifact, treating a corrupt live cache as a miss.

        A completed cache file is an optimisation, never authority to make a
        live Job fail.  In cache-only mode it remains strict: a malformed
        artifact is reported as a cache miss rather than prompting the model.
        """
        cached = self._load(key)
        if cached is None:
            return None
        try:
            claims = _claims_from_cache(cached, response, source)
        except (StructuredOutputParseError, StructuredOutputSchemaError, TypeError, ValueError) as exc:
            if self.cache_only:
                raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json") from exc
            return None
        self._record(key, 0.0, cached=True)
        return claims

    def _save_claims(self, key: str, claims: Sequence[AtomicClaim]) -> None:
        self._save(key, _claim_cache_payload(claims))

    def _record(self, key: str, elapsed: float, cached: bool) -> None:
        if self.usage is not None:
            self.usage.record_call(self.component, key, elapsed, cached=cached)

    def _call_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        name: str,
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        try:
            from litellm import completion  # type: ignore
        except Exception as exc:  # pragma: no cover - live-only dependency
            raise CriticalProtocolError("litellm is required for critical verification") from exc
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens),
            "timeout": self.request_timeout_s,
            "num_retries": 0,
        }
        api_key = resolve_api_key(self.cfg)
        if api_key:
            kwargs["api_key"] = api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.structured_output.transport == "response_format":
            kwargs["response_format"] = json_schema_response_format(schema, name=name)
            if self.structured_output.request_backend is not None:
                kwargs["extra_body"] = {
                    "guided_decoding_backend": self.structured_output.request_backend
                }
        elif self.structured_output.transport == "guided_json":
            kwargs["extra_body"] = {"guided_json": schema}
        self.request_pacer.wait_for_turn()
        response = completion(**kwargs)
        validate_gateway_identity(response, label=f"{self.component} completion")
        choices = _response_field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise StructuredOutputParseError(f"{self.component} completion must contain one choice")
        choice = choices[0]
        finish_reason = str(_response_field(choice, "finish_reason") or "").lower()
        if finish_reason == "length":
            raise CriticalCompletionTruncatedError(
                f"{self.component} completion hit max_tokens={max_tokens}"
            )
        if finish_reason != "stop":
            raise StructuredOutputParseError(
                f"{self.component} completion did not finish cleanly: {finish_reason or 'missing'}"
            )
        content = _response_field(_response_field(choice, "message"), "content")
        if not isinstance(content, str):
            raise StructuredOutputParseError(f"{self.component} response has no text content")
        return strict_json_loads(content.strip(), label=f"{self.component} response")

    def _retry_json(self, messages: list[dict[str, str]], schema: dict[str, Any], name: str) -> dict[str, Any]:
        result: dict[str, Any] | None = None
        token_budget = self.max_tokens

        def should_retry(exc: BaseException) -> bool:
            return isinstance(exc, CriticalRetryableTruncation) or is_retryable_llm_exception(exc)

        for attempt in Retrying(
            # ``0`` leaves non-capacity transient retries to the enclosing
            # Job, while a continuous 429 streak has its own explicit limit.
            # Each completed artifact is atomically cached before the next
            # claim begins.
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
            retry=retry_if_exception(should_retry),
            before_sleep=RetryHeartbeat(self.component, self.usage),
            reraise=True,
        ):
            with attempt:
                try:
                    result = self._call_json(messages, schema, name, max_tokens=token_budget)
                except CriticalCompletionTruncatedError as exc:
                    if token_budget >= self.max_tokens_ceiling:
                        raise CriticalOutputLimitError(
                            f"{self.component} remained truncated at max_tokens={token_budget}"
                        ) from exc
                    next_budget = min(token_budget * 2, self.max_tokens_ceiling)
                    token_budget = next_budget
                    raise CriticalRetryableTruncation(
                        f"{self.component} retrying after token truncation with max_tokens={next_budget}"
                    ) from exc
        assert result is not None
        return result

    def _retry_validated_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        name: str,
        validator: Callable[[dict[str, Any]], _T],
    ) -> _T:
        """Bounded recovery for malformed structured artifacts.

        Transport errors are retried by ``_retry_json`` according to the
        long-lived provider policy. A malformed payload is not a transport
        error, so reissue it only a small, explicit number of times with a
        schema correction. Deterministic offset recovery is attempted before
        this fallback.
        """
        retry_messages = list(messages)
        for protocol_attempt in range(self.max_protocol_retries):
            payload = self._retry_json(retry_messages, schema, name)
            try:
                return validator(payload)
            except (StructuredOutputParseError, StructuredOutputSchemaError):
                if protocol_attempt + 1 >= self.max_protocol_retries:
                    raise
                retry_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "The preceding structured answer was rejected. Return one JSON object matching the "
                            "given schema exactly. For every claim, copy text verbatim from the answer and use "
                            "zero-based, end-exclusive Python offsets for that exact substring."
                        ),
                    },
                ]
        raise AssertionError("unreachable structured-output retry state")


class AtomicClaimExtractor(_CachedComponent):
    component = "critical_claim_extractor"
    protocol = "support-critical-claims-v1"

    def __init__(
        self, cfg, usage=None, *, cache_only: bool = False, request_pacer: RequestPacer | None = None
    ):
        super().__init__(
            cfg, "claim_extractor", usage, cache_only=cache_only, request_pacer=request_pacer
        )
        self.chunk_chars = int(config_value(self.section, "chunk_chars", 0))
        self.min_chunk_chars = int(config_value(self.section, "min_chunk_chars", 160))
        if self.chunk_chars < 0 or self.min_chunk_chars <= 0:
            raise ValueError("claim_extractor chunk limits must be positive")

    def _segment_key_payload(self, segment: str) -> dict[str, Any]:
        # The segment protocol is deliberately part of the key instead of a
        # protocol bump: full-answer entries from earlier interrupted Jobs
        # remain readable, while new chunks are independently resumable.
        return {"segment_protocol": "answer-chunks-v1", "answer_segment": segment}

    def _extract_segment(self, segment: str) -> list[AtomicClaim]:
        key = self._cache_key(self._segment_key_payload(segment))
        cached = self._load_claims(key, segment, "atomic")
        if cached is not None:
            return cached
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        messages = [
            {
                "role": "system",
                "content": (
                    "Extract every minimal independently checkable factual claim in the answer segment. "
                    "Keep quantities, dates, negation, comparison, condition, modality, and scope in the claim. "
                    "Do not include greetings, opinions, or purely stylistic text. Each text value must be an exact "
                    "substring of the answer segment and start/end are zero-based Python string offsets, end exclusive."
                ),
            },
            {"role": "user", "content": f"Answer segment:\n{segment}"},
        ]
        start = time.perf_counter()
        try:
            claims = self._retry_validated_json(
                messages,
                CLAIM_SCHEMA,
                "atomic_claims",
                lambda payload: _validate_claims(payload, segment, "atomic"),
            )
        except CriticalOutputLimitError:
            # Let a larger segment bisect first. At the deterministic floor,
            # exact sentence candidates preserve coverage without more futile
            # output-budget retries.
            if len(segment) > self.min_chunk_chars:
                raise
            claims = _sentence_fallback_claims(segment, "atomic_fallback_sentence")
        except (StructuredOutputParseError, StructuredOutputSchemaError):
            # The provider has already had max_protocol_retries chances to
            # repair the schema/offsets. Do not discard the entire 100-QA run
            # because it paraphrased a single claim.
            claims = _sentence_fallback_claims(segment, "atomic_fallback_sentence")
        except Exception as exc:  # noqa: BLE001
            if not is_retryable_llm_exception(exc):
                raise
            # A list-producing component must not pretend the missing model
            # result was empty: sentence candidates preserve every factual
            # span for the later four-way verifier, and the cache records the
            # deterministic fallback for cache-only replay.
            claims = _sentence_fallback_claims(segment, "atomic_fallback_sentence")
        self._save_claims(key, claims)
        self._record(key, time.perf_counter() - start, cached=False)
        return claims

    def _extract_chunk(self, chunk: _AnswerChunk) -> list[AtomicClaim]:
        try:
            local = self._extract_segment(chunk.text)
        except CriticalOutputLimitError:
            children = _bisect_chunk(chunk, self.min_chunk_chars)
            if children is None:
                raise
            claims: list[AtomicClaim] = []
            for child in children:
                claims.extend(self._extract_chunk(child))
            return claims
        return [
            replace(claim, start=claim.start + chunk.start, end=claim.end + chunk.start)
            for claim in local
        ]

    def extract(self, response: str) -> list[AtomicClaim]:
        if not response.strip():
            return []
        # Preserve an exact completed full-answer artifact from previous Jobs.
        # This is especially valuable for the partial cache produced before
        # segmentation existed.
        key = self._cache_key({"response": response})
        cached = self._load_claims(key, response, "atomic")
        if cached is not None:
            return cached
        # Short responses retain the historical one-request cache identity;
        # only long answers take the new resumable segmented path.
        if not self.chunk_chars or len(response) <= self.chunk_chars:
            if self.cache_only:
                raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract every minimal independently checkable factual claim in the answer. "
                        "Keep quantities, dates, negation, comparison, condition, modality, and scope in the claim. "
                        "Do not include greetings, opinions, or purely stylistic text. Each text value must be an exact "
                        "substring of the answer and start/end are zero-based Python string offsets, end exclusive."
                    ),
                },
                {"role": "user", "content": f"Answer:\n{response}"},
            ]
            start = time.perf_counter()
            try:
                claims = self._retry_validated_json(
                    messages,
                    CLAIM_SCHEMA,
                    "atomic_claims",
                    lambda payload: _validate_claims(payload, response, "atomic"),
                )
            except CriticalOutputLimitError:
                # A provider can hit an output ceiling even for a nominally
                # short but claim-dense answer. Fall through to the same
                # deterministic bisection machinery rather than fail late.
                claims = (
                    _sentence_fallback_claims(response, "atomic_fallback_sentence")
                    if len(response) <= self.min_chunk_chars
                    else self._extract_chunk(_AnswerChunk(response, 0, len(response)))
                )
                merged = merge_claims(claims)
                # A compact answer-level index makes cache-only replay work
                # even when a provider unexpectedly forced bisection below
                # the configured chunk threshold. Leaf caches remain useful
                # for resuming an interrupted live attempt.
                self._save_claims(key, merged)
                return merged
            except (StructuredOutputParseError, StructuredOutputSchemaError):
                claims = _sentence_fallback_claims(response, "atomic_fallback_sentence")
                self._save_claims(key, claims)
                self._record(key, time.perf_counter() - start, cached=False)
                return claims
            except Exception as exc:  # noqa: BLE001
                if is_retryable_llm_exception(exc):
                    claims = _sentence_fallback_claims(response, "atomic_fallback_sentence")
                    self._save_claims(key, claims)
                    self._record(key, time.perf_counter() - start, cached=False)
                    return claims
                raise CriticalProtocolError("atomic claim extraction failed") from exc
            self._save_claims(key, claims)
            self._record(key, time.perf_counter() - start, cached=False)
            return claims
        try:
            claims = []
            for chunk in _answer_chunks(response, self.chunk_chars):
                claims.extend(self._extract_chunk(chunk))
        except Exception as exc:  # noqa: BLE001
            if is_retryable_llm_exception(exc):
                # A transport failure can occur while recursively handling a
                # preceding output-limit exception.  Do not let that unusual
                # exception context discard the whole experiment: preserve
                # all answer spans in an explicit, cacheable fallback.
                claims = _sentence_fallback_claims(response, "atomic_fallback_sentence")
                self._save_claims(key, claims)
                return claims
            raise CriticalProtocolError("atomic claim extraction failed") from exc
        merged = merge_claims(claims)
        # Index the deterministic leaf artifacts under the historical
        # full-answer key as well. This is an atomic replay marker, not the
        # source of truth during an interrupted live extraction.
        self._save_claims(key, merged)
        return merged


class FullContextReviewer(_CachedComponent):
    component = "critical_coverage_reviewer"
    protocol = "support-critical-coverage-v1"

    def __init__(
        self, cfg, usage=None, *, cache_only: bool = False, request_pacer: RequestPacer | None = None
    ):
        super().__init__(
            cfg, "coverage_reviewer", usage, cache_only=cache_only, request_pacer=request_pacer
        )
        self.chunk_chars = int(config_value(self.section, "chunk_chars", 0))
        self.min_chunk_chars = int(config_value(self.section, "min_chunk_chars", 160))
        if self.chunk_chars < 0 or self.min_chunk_chars <= 0:
            raise ValueError("coverage_reviewer chunk limits must be positive")

    @staticmethod
    def _claims_for_chunk(claims: Sequence[AtomicClaim], chunk: _AnswerChunk) -> list[AtomicClaim]:
        """Select known candidates touching this segment in a stable order."""
        return [
            claim for claim in claims
            if claim.start < chunk.end and claim.end > chunk.start
        ]

    def _segment_key_payload(
        self, segment: str, context: str, query: str | None, known_claims: Sequence[AtomicClaim]
    ) -> dict[str, Any]:
        return {
            "segment_protocol": "answer-chunks-v1",
            "answer_segment": segment,
            "context": context,
            "query": query or "",
            # The prompt reads claim text only. Excluding absolute offsets lets
            # an identical segment resume even when it occurs in a new answer.
            "known_claims": [claim.text for claim in known_claims],
        }

    def _review_segment(
        self, segment: str, context: str, query: str | None, known_claims: Sequence[AtomicClaim]
    ) -> list[AtomicClaim]:
        key = self._cache_key(self._segment_key_payload(segment, context, query, known_claims))
        cached = self._load_claims(key, segment, "global_review")
        if cached is not None:
            return cached
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        known = "\n".join(f"- {claim.text}" for claim in known_claims) or "(none)"
        messages = [
            {
                "role": "system",
                "content": (
                    "Read the entire source and answer segment under a closed-world rule. Find factual answer "
                    "fragments that may be unsupported or contradicted by the source, especially a single subtle "
                    "extra fact. Return only exact answer-segment substrings with Python offsets relative to the "
                    "answer segment. Do not return claims that are directly supported. This is a candidate-generation "
                    "review, not a final verdict."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuery:\n{query or ''}\n\nAnswer segment:\n{segment}\n\n"
                f"Already extracted claims touching this segment:\n{known}",
            },
        ]
        start = time.perf_counter()
        try:
            claims = self._retry_validated_json(
                messages,
                CLAIM_SCHEMA,
                "coverage_candidates",
                lambda payload: _validate_claims(payload, segment, "global_review"),
            )
        except CriticalOutputLimitError:
            if len(segment) > self.min_chunk_chars:
                raise
            claims = _sentence_fallback_claims(segment, "global_review_fallback_sentence")
        except (StructuredOutputParseError, StructuredOutputSchemaError):
            # Coverage is an additional adversarial review. Its protocol
            # failure must not erase the answer-level candidate coverage.
            claims = _sentence_fallback_claims(segment, "global_review_fallback_sentence")
        except Exception as exc:  # noqa: BLE001
            if not is_retryable_llm_exception(exc):
                raise
            # Preserve coverage under an exhausted network budget instead of
            # silently omitting factual answer spans from the claim layer.
            claims = _sentence_fallback_claims(segment, "global_review_fallback_sentence")
        self._save_claims(key, claims)
        self._record(key, time.perf_counter() - start, cached=False)
        return claims

    def _review_chunk(
        self, chunk: _AnswerChunk, context: str, query: str | None, known_claims: Sequence[AtomicClaim]
    ) -> list[AtomicClaim]:
        local_known = self._claims_for_chunk(known_claims, chunk)
        try:
            local = self._review_segment(chunk.text, context, query, local_known)
        except CriticalOutputLimitError:
            children = _bisect_chunk(chunk, self.min_chunk_chars)
            if children is None:
                raise
            claims: list[AtomicClaim] = []
            for child in children:
                claims.extend(self._review_chunk(child, context, query, known_claims))
            return claims
        return [
            replace(claim, start=claim.start + chunk.start, end=claim.end + chunk.start)
            for claim in local
        ]

    def review(
        self, response: str, context: str, query: str | None, known_claims: Sequence[AtomicClaim]
    ) -> list[AtomicClaim]:
        if not response.strip():
            return []
        key = self._cache_key({
            "response": response,
            "context": context,
            "query": query or "",
            "known_claims": [c.to_dict() for c in known_claims],
        })
        cached = self._load_claims(key, response, "global_review")
        if cached is not None:
            return cached
        if not self.chunk_chars or len(response) <= self.chunk_chars:
            if self.cache_only:
                raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
            known = "\n".join(f"- {claim.text}" for claim in known_claims) or "(none)"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Read the entire source and answer under a closed-world rule. Find factual answer fragments that "
                        "may be unsupported or contradicted by the source, especially a single subtle extra fact. "
                        "Return only exact answer substrings with Python offsets. Do not return claims that are directly "
                        "supported. This is a candidate-generation review, not a final verdict."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuery:\n{query or ''}\n\nAnswer:\n{response}\n\n"
                    f"Already extracted claims:\n{known}",
                },
            ]
            start = time.perf_counter()
            try:
                claims = self._retry_validated_json(
                    messages,
                    CLAIM_SCHEMA,
                    "coverage_candidates",
                    lambda payload: _validate_claims(payload, response, "global_review"),
                )
            except CriticalOutputLimitError:
                claims = (
                    _sentence_fallback_claims(response, "global_review_fallback_sentence")
                    if len(response) <= self.min_chunk_chars
                    else self._review_chunk(
                        _AnswerChunk(response, 0, len(response)), context, query, known_claims
                    )
                )
                merged = merge_claims(claims)
                self._save_claims(key, merged)
                return merged
            except (StructuredOutputParseError, StructuredOutputSchemaError):
                claims = _sentence_fallback_claims(response, "global_review_fallback_sentence")
                self._save_claims(key, claims)
                self._record(key, time.perf_counter() - start, cached=False)
                return claims
            except Exception as exc:  # noqa: BLE001
                if is_retryable_llm_exception(exc):
                    claims = _sentence_fallback_claims(response, "global_review_fallback_sentence")
                    self._save_claims(key, claims)
                    self._record(key, time.perf_counter() - start, cached=False)
                    return claims
                raise CriticalProtocolError("full-context coverage review failed") from exc
            self._save_claims(key, claims)
            self._record(key, time.perf_counter() - start, cached=False)
            return claims
        try:
            claims = []
            for chunk in _answer_chunks(response, self.chunk_chars):
                claims.extend(self._review_chunk(chunk, context, query, known_claims))
        except Exception as exc:  # noqa: BLE001
            if is_retryable_llm_exception(exc):
                # Keep the coverage layer restartable when a provider outage
                # interrupts recursive output-limit segmentation.  The
                # fallback remains explicit and cache-only replayable.
                claims = _sentence_fallback_claims(response, "global_review_fallback_sentence")
                self._save_claims(key, claims)
                return claims
            raise CriticalProtocolError("full-context coverage review failed") from exc
        merged = merge_claims(claims)
        self._save_claims(key, merged)
        return merged


def select_claim_evidence(
    context: str,
    query: str | None,
    claim: str,
    *,
    max_sentences: int,
    stopwords: Iterable[str] = (),
    embedder: Embedder | None = None,
) -> list[EvidenceSpan]:
    """Deterministically rank lexical and local-S-BERT sentence evidence."""
    blocked = set(stopwords) | _STOPWORDS
    claim_norm = normalize(claim)
    tokens = [t for t in _WORD_RE.findall(claim_norm) if t not in blocked]
    candidates: list[EvidenceSpan] = []
    for source, text in (("context", context or ""), ("query", query or "")):
        for index, start, end, sentence in _sentences(text):
            sentence_norm = normalize(sentence)
            token_hits = sum(
                int(re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", sentence_norm) is not None)
                for token in tokens
            )
            phrase = int(bool(claim_norm and claim_norm in sentence_norm))
            candidates.append(EvidenceSpan(source, index, start, end, sentence, 4 * phrase + token_hits))
    if not candidates:
        return []
    semantic = np.zeros(len(candidates), dtype=float)
    if embedder is not None:
        try:
            vectors = embedder.encode([claim] + [span.text for span in candidates])
            if len(vectors) == len(candidates) + 1:
                semantic = vectors[1:] @ vectors[0]
        except Exception:
            # Retrieval can remain lexical if an optional test embedder cannot encode a value.
            semantic = np.zeros(len(candidates), dtype=float)
    ranked = list(enumerate(candidates))
    ranked.sort(
        key=lambda item: (
            -item[1].rank,
            -float(semantic[item[0]]),
            0 if item[1].source == "context" else 1,
            item[1].start,
            item[1].index,
        )
    )
    return [span for _, span in ranked[: max(0, int(max_sentences))]]


class CriticalClaimVerifier(_CachedComponent):
    component = "critical_claim_verifier"
    protocol = "support-critical-verdict-v1"

    def __init__(
        self,
        cfg,
        usage=None,
        *,
        cache_only: bool = False,
        embedder: Embedder | None = None,
        request_pacer: RequestPacer | None = None,
    ):
        super().__init__(
            cfg, "claim_verifier", usage, cache_only=cache_only, request_pacer=request_pacer
        )
        self.max_sentences = int(config_value(self.section, "max_evidence_sentences", 8))
        self.stopwords = set(getattr(cfg.matching, "stopwords", []) or [])
        self.embedder = embedder

    def verify_claim(self, claim: str, context: str, query: str | None) -> CriticalVerdict:
        evidence = select_claim_evidence(
            context, query, claim, max_sentences=self.max_sentences,
            stopwords=self.stopwords, embedder=self.embedder,
        )
        key = self._cache_key({
            "claim": claim,
            "evidence": [span.to_dict() for span in evidence],
            "max_evidence_sentences": self.max_sentences,
            "embedding_model": config_value(self.cfg.matching, "embedding_model"),
            "embedding_model_revision": config_value(self.cfg.matching, "embedding_model_revision"),
        })
        cached = self._load(key)
        if cached is not None and cached.get("verdict") in CRITICAL_VERDICTS:
            self._record(key, 0.0, cached=True)
            return CriticalVerdict(
                str(cached["verdict"]), tuple(evidence), cache_hit=True,
                protocol_fallback=bool(cached.get("_hallu_protocol_fallback", False)),
                fallback_reason=(
                    str(cached["_hallu_fallback_reason"])
                    if cached.get("_hallu_protocol_fallback", False)
                    and isinstance(cached.get("_hallu_fallback_reason"), str)
                    else ("structured_output_exhausted" if cached.get("_hallu_protocol_fallback", False) else None)
                ),
            )
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        evidence_text = "\n".join(f"[{span.source}:{span.index}] {span.text}" for span in evidence)
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the claim using only the supplied evidence. Entailed requires direct support for every "
                    "material detail, including quantity, date, negation, condition, modality, comparison, and scope. "
                    "Contradicted requires direct incompatible evidence. Unsupported means this is a clear factual claim "
                    "but the evidence does not directly establish it; do not use world knowledge or plausible inference. "
                    "Unknown is reserved for a non-factual or genuinely ambiguous fragment, not merely missing evidence."
                ),
            },
            {"role": "user", "content": f"Claim:\n{claim}\n\nEvidence:\n{evidence_text or '(no evidence retrieved)'}"},
        ]
        start = time.perf_counter()
        try:
            verdict = self._retry_validated_json(
                messages,
                VERDICT_SCHEMA,
                "critical_claim_verdict",
                _validated_critical_verdict,
            )
            protocol_fallback = False
            fallback_reason = None
        except (CriticalOutputLimitError, StructuredOutputParseError, StructuredOutputSchemaError):
            # There is no defensible positive/negative inference when a scalar
            # verifier repeatedly violates its own schema. ``unknown`` is the
            # only conservative four-way result and remains train-tunable.
            verdict = "unknown"
            protocol_fallback = True
            fallback_reason = "structured_output_exhausted"
        except Exception as exc:  # noqa: BLE001
            if is_retryable_llm_exception(exc):
                # After the finite timeout/5xx budget, a scalar factual
                # decision has no defensible polarity. Preserve the completed
                # cache work and let train-tuned ``unknown`` handling absorb
                # this explicitly marked outcome.
                verdict = "unknown"
                protocol_fallback = True
                fallback_reason = "transient_exhausted"
            else:
                raise CriticalProtocolError(f"critical claim verification failed for {claim!r}") from exc
        payload: dict[str, Any] = {"verdict": verdict}
        if protocol_fallback:
            payload["_hallu_protocol_fallback"] = True
            payload["_hallu_fallback_reason"] = fallback_reason
        self._save(key, payload)
        self._record(key, time.perf_counter() - start, cached=False)
        return CriticalVerdict(
            verdict,
            tuple(evidence),
            cache_hit=False,
            protocol_fallback=protocol_fallback,
            fallback_reason=fallback_reason,
        )

    # Interface parity with RelationVerifier for graph-edge scoring.
    def verify(
        self,
        canonical_triple: tuple[str, str, str],
        context: str,
        query: str | None,
        *,
        matching_params: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> CriticalVerdict:
        claim = f"{canonical_triple[0]} {canonical_triple[1]} {canonical_triple[2]}."
        return self.verify_claim(claim, context, query)


class CriticalClaimPipeline:
    """Build, review, deduplicate, and verify response-level atomic claims."""

    def __init__(self, cfg, usage=None, *, cache_only: bool = False, embedder: Embedder | None = None):
        pacer = RequestPacer(float(getattr(cfg.llm, "request_min_interval_s", 0)))
        self.extractor = AtomicClaimExtractor(
            cfg, usage, cache_only=cache_only, request_pacer=pacer
        )
        self.reviewer = FullContextReviewer(
            cfg, usage, cache_only=cache_only, request_pacer=pacer
        )
        self.verifier = CriticalClaimVerifier(
            cfg, usage, cache_only=cache_only, embedder=embedder, request_pacer=pacer
        )

    def assess(
        self,
        response: str,
        context: str,
        query: str | None,
        *,
        progress_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        def emit(phase: str, completed: int = 0, total: int = 0) -> None:
            if progress_hook is not None:
                progress_hook({"phase": phase, "completed": completed, "total": total})

        emit("claim_extraction")
        atomic = self.extractor.extract(response)
        emit("coverage_review", len(atomic), len(atomic))
        reviewed = self.reviewer.review(response, context, query, atomic)
        claims = merge_claims(atomic, reviewed)
        claim_total = len(claims)
        claim_interval = max(1, claim_total // 10) if claim_total else 1
        emit("claim_verification", 0, claim_total)
        audits: list[dict[str, Any]] = []
        for completed, claim in enumerate(claims, start=1):
            decision = self.verifier.verify_claim(claim.text, context, query)
            audits.append({
                "claim": claim.to_dict(),
                "evidence": [span.to_dict() for span in decision.evidence],
                "verdict": decision.verdict,
                "verifier_cache_hit": decision.cache_hit,
                "verifier_protocol_fallback": decision.protocol_fallback,
                "verifier_fallback_reason": decision.fallback_reason,
            })
            if completed == 1 or completed == claim_total or completed % claim_interval == 0:
                emit("claim_verification", completed, claim_total)
        emit("claim_verification_done", claim_total, claim_total)
        return audits


class FakeCriticalClaimPipeline:
    """Deterministic offline fixture for ``support-critical`` tests."""

    def __init__(self, claims: Sequence[AtomicClaim] | None = None, verdicts: dict[str, str] | None = None):
        self.claims = list(claims or [])
        self.verdicts = {normalize(k): v for k, v in (verdicts or {}).items()}

    def assess(
        self,
        response: str,
        context: str,
        query: str | None,
        *,
        progress_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:  # noqa: ARG002
        if progress_hook is not None:
            progress_hook({"phase": "claim_verification_done", "completed": len(self.claims), "total": len(self.claims)})
        return [
            {
                "claim": claim.to_dict(),
                "evidence": [],
                "verdict": self.verdicts.get(normalize(claim.text), "unknown"),
                "verifier_cache_hit": False,
            }
            for claim in self.claims
        ]
