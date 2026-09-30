"""Four-way Gemini verification of VeriScore claims with durable component caches."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

import numpy as np
from tenacity import Retrying, retry_if_exception

from .cache import CacheOnlyMissError, config_value, llm_runtime_fingerprint
from .config import litellm_transport_model, resolve_api_key
from .dspy_adapter import (
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    is_retryable_llm_exception,
    json_schema_response_format,
    strict_json_loads,
    validate_structured_output_settings,
    validate_gateway_identity,
    validate_json_document,
)
from .evidence_spans import EvidenceSpan, _sentences
from .matching import Embedder, normalize
from .retry import (
    RequestPacer,
    RetryHeartbeat,
    StopAfterAttemptsExceptRateLimit,
    WaitRetryAfterOrExponentialJitter,
)

CRITICAL_VERDICTS = frozenset({"entailed", "unknown", "unsupported", "contradicted"})
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": sorted(CRITICAL_VERDICTS)}},
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


def _response_field(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


def _validated_critical_verdict(payload: dict[str, Any]) -> str:
    validate_json_document(payload, VERDICT_SCHEMA)
    return str(payload["verdict"])


class _CachedComponent:
    """Common cache, transport, and retry contract for critical components."""

    component: str = "critical_component"
    protocol: str = "det-rep-component-v1"
    config_root: str = "critical"
    repair_message: str = (
        "The preceding structured answer was rejected. Return one JSON object "
        "matching the given schema exactly."
    )

    def __init__(
        self,
        cfg,
        section_name: str,
        usage=None,
        *,
        cache_only: bool = False,
        request_pacer: RequestPacer | None = None,
    ):
        critical_cfg = getattr(cfg, self.config_root, None)
        if critical_cfg is None:
            raise ValueError(f"{self.config_root} config is required")
        section = getattr(critical_cfg, section_name, None)
        if section is None:
            raise ValueError(f"{self.config_root}.{section_name} config is required")
        self.cfg = cfg
        self.section = section
        self.model = cfg.llm.model
        self.api_base = getattr(cfg.llm, "api_base", None)
        validate_structured_output_settings(cfg.llm)
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
        try:
            payload = json.loads((self.cache_dir / f"{key}.json").read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return None

    def _save(self, key: str, payload: dict[str, Any]) -> None:
        dest = self.cache_dir / f"{key}.json"
        tmp = dest.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, dest)

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
            "model": litellm_transport_model(self.cfg),
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
        kwargs["response_format"] = json_schema_response_format(schema, name=name)
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
                    {"role": "system", "content": self.repair_message},
                ]
        raise AssertionError("unreachable structured-output retry state")


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
        vectors = embedder.encode([claim] + [span.text for span in candidates])
        if len(vectors) != len(candidates) + 1:
            raise ValueError("claim evidence embedder returned an incomplete result")
        semantic = vectors[1:] @ vectors[0]
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
    protocol = "det-rep-four-way-verdict-v2"

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
        if cached is not None and (
            cached.get("_hallu_protocol_fallback")
            or cached.get("_hallu_fallback_reason") is not None
        ):
            raise CriticalProtocolError("critical verifier cache contains a fallback verdict")
        if cached is not None and cached.get("verdict") in CRITICAL_VERDICTS:
            self._record(key, 0.0, cached=True)
            return CriticalVerdict(
                str(cached["verdict"]), tuple(evidence), cache_hit=True,
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
        except (CriticalOutputLimitError, StructuredOutputParseError, StructuredOutputSchemaError) as exc:
            raise CriticalProtocolError("critical claim verification protocol exhausted") from exc
        except Exception as exc:  # noqa: BLE001
            if is_retryable_llm_exception(exc):
                raise CriticalProtocolError("critical claim verification transport exhausted") from exc
            raise CriticalProtocolError("critical claim verification failed") from exc
        self._save(key, {"verdict": verdict})
        self._record(key, time.perf_counter() - start, cached=False)
        return CriticalVerdict(
            verdict,
            tuple(evidence),
            cache_hit=False,
        )
