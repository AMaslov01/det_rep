"""Shared transient-retry timing for gateway-backed components.

This module deliberately contains no provider client code.  It extracts a
standard ``Retry-After`` header from common wrapped HTTP exceptions and combines
it with bounded exponential full-jitter backoff.  The caller still decides
which exception classes are transient.
"""
from __future__ import annotations

import json
import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any


class RateLimitRetryDeadlineExceeded(RuntimeError):
    """A single request remained capacity-blocked beyond its safe wait budget."""


class RetryDeadlineExceeded(RuntimeError):
    """A retryable request exceeded its total safe recovery budget.

    This differs from :class:`RateLimitRetryDeadlineExceeded`: alternating
    429, 5xx, and truncated-response failures are still one unavailable
    request from the experiment's perspective.  They must not reset the
    budget and pin a source-level progress counter indefinitely.
    """


class RequestPacer:
    """Serialize request admission at a stable minimum interval.

    ``concurrency=1`` prevents overlap but not an overly fast serial request
    loop.  Reserving turns under a lock lets sibling components share one
    sustainable request rate without changing prompts, caches, or semantics.
    """

    def __init__(self, minimum_interval_seconds: float = 0.0):
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        if self.minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds must be non-negative")
        self._lock = threading.Lock()
        self._next_permit_at = 0.0

    def wait_for_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            permit_at = max(now, self._next_permit_at)
            self._next_permit_at = permit_at + self.minimum_interval_seconds
        remaining = permit_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


def transient_http_status(exc: BaseException | None) -> int | None:
    """Return an explicit HTTP status from a wrapped transient exception."""
    for item in _exception_chain(exc):
        response = getattr(item, "response", None)
        status = getattr(item, "status_code", None)
        if status is None and response is not None:
            status = getattr(response, "status_code", None)
        if status is None:
            continue
        try:
            return int(status)
        except (TypeError, ValueError):
            continue
    return None


def is_rate_limit_error(exc: BaseException | None) -> bool:
    """Recognise a provider capacity/rate rejection, including wrapped 429s."""
    if transient_http_status(exc) == 429:
        return True
    for item in _exception_chain(exc):
        if type(item).__name__ == "RateLimitError":
            return True
    message = " ".join(str(item) for item in _exception_chain(exc)).lower()
    return "rate limit" in message or "rate_limit" in message


def _exception_chain(exc: BaseException | None) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def retry_after_seconds(exc: BaseException | None) -> float | None:
    """Return a non-negative ``Retry-After`` delay from a wrapped response."""
    for item in _exception_chain(exc):
        for owner in (item, getattr(item, "response", None)):
            headers: Any = getattr(owner, "headers", None)
            if headers is None:
                continue
            try:
                value = headers.get("Retry-After") or headers.get("retry-after")
            except (AttributeError, TypeError):
                continue
            if value is None:
                continue
            try:
                return max(0.0, float(str(value).strip()))
            except ValueError:
                try:
                    target = parsedate_to_datetime(str(value))
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=timezone.utc)
                    return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, IndexError, OverflowError):
                    continue
    return None


class WaitRetryAfterOrExponentialJitter:
    """Tenacity wait callable with a bounded ``Retry-After`` override.

    The jitter prevents a synchronized retry burst.  When a gateway gives a
    longer explicit delay, honour it up to the configured maximum pause.
    """

    def __init__(
        self,
        base_seconds: float,
        max_seconds: float,
        *,
        rate_limit_cooldown_max_seconds: float | None = None,
        rate_limit_retry_deadline_seconds: float | None = None,
        retry_deadline_seconds: float | None = None,
    ):
        self.base_seconds = max(0.0, float(base_seconds))
        self.max_seconds = max(0.0, float(max_seconds))
        self.rate_limit_cooldown_max_seconds = max(
            self.max_seconds,
            float(rate_limit_cooldown_max_seconds)
            if rate_limit_cooldown_max_seconds is not None
            else self.max_seconds,
        )
        self.rate_limit_retry_deadline_seconds = (
            None
            if rate_limit_retry_deadline_seconds is None
            else float(rate_limit_retry_deadline_seconds)
        )
        self.retry_deadline_seconds = (
            None if retry_deadline_seconds is None else float(retry_deadline_seconds)
        )
        if (
            self.rate_limit_retry_deadline_seconds is not None
            and self.rate_limit_retry_deadline_seconds <= 0
        ):
            raise ValueError("rate_limit_retry_deadline_seconds must be positive when set")
        if self.retry_deadline_seconds is not None and self.retry_deadline_seconds <= 0:
            raise ValueError("retry_deadline_seconds must be positive when set")

    @staticmethod
    def _rate_limit_started_at(retry_state: Any, now: float) -> float:
        started = getattr(retry_state, "_hallu_rate_limit_started_at", None)
        if started is None:
            started = now
            setattr(retry_state, "_hallu_rate_limit_started_at", started)
        return float(started)

    @staticmethod
    def _retry_started_at(retry_state: Any, now: float) -> float:
        started = getattr(retry_state, "_hallu_retry_started_at", None)
        if started is None:
            started = now
            setattr(retry_state, "_hallu_retry_started_at", started)
        return float(started)

    def __call__(self, retry_state: Any) -> float:
        attempt = max(1, int(getattr(retry_state, "attempt_number", 1)))
        outcome = getattr(retry_state, "outcome", None)
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        is_rate_limited = is_rate_limit_error(exception)
        cap_limit = (
            self.rate_limit_cooldown_max_seconds
            if is_rate_limited
            else self.max_seconds
        )
        now = time.monotonic()
        if self.retry_deadline_seconds is not None:
            remaining = self.retry_deadline_seconds - (
                now - self._retry_started_at(retry_state, now)
            )
            cap_limit = min(cap_limit, max(0.0, remaining))
        if is_rate_limited and self.rate_limit_retry_deadline_seconds is not None:
            remaining = self.rate_limit_retry_deadline_seconds - (
                now - self._rate_limit_started_at(retry_state, now)
            )
            cap_limit = min(cap_limit, max(0.0, remaining))
        elif not is_rate_limited:
            setattr(retry_state, "_hallu_rate_limit_started_at", None)
        cap = min(cap_limit, self.base_seconds * (2 ** (attempt - 1)))
        jittered = random.uniform(0.0, cap) if cap > 0 else 0.0
        retry_after = retry_after_seconds(exception)
        if retry_after is None:
            return jittered
        if is_rate_limited:
            # The gateway's admission-control delay is more informative than
            # a large exponential cap. Add only a small de-synchronisation
            # jitter, rather than idling an extra fifteen minutes after it has
            # explicitly invited a retry in (say) one second.
            cooldown_jitter = random.uniform(0.0, min(5.0, max(1.0, retry_after * 0.1)))
            return min(cap_limit, retry_after + cooldown_jitter)
        return min(cap_limit, max(jittered, retry_after))


class StopAfterAttemptsExceptRateLimit:
    """Bound potentially billable retries while allowing 429 cooldown/recovery.

    A 429 is provider admission control, normally returned before generation.
    It can wait beyond the ordinary timeout budget, but a single continuous
    capacity outage still has an explicit wall-time deadline.
    """

    def __init__(
        self,
        max_attempts: int | None,
        *,
        rate_limit_retry_deadline_seconds: float | None = None,
        retry_deadline_seconds: float | None = None,
    ):
        if max_attempts is not None and int(max_attempts) < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = None if max_attempts is None else int(max_attempts)
        self.rate_limit_retry_deadline_seconds = (
            None
            if rate_limit_retry_deadline_seconds is None
            else float(rate_limit_retry_deadline_seconds)
        )
        self.retry_deadline_seconds = (
            None if retry_deadline_seconds is None else float(retry_deadline_seconds)
        )
        if (
            self.rate_limit_retry_deadline_seconds is not None
            and self.rate_limit_retry_deadline_seconds <= 0
        ):
            raise ValueError("rate_limit_retry_deadline_seconds must be positive when set")
        if self.retry_deadline_seconds is not None and self.retry_deadline_seconds <= 0:
            raise ValueError("retry_deadline_seconds must be positive when set")

    def __call__(self, retry_state: Any) -> bool:
        outcome = getattr(retry_state, "outcome", None)
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        now = time.monotonic()
        started = getattr(retry_state, "_hallu_retry_started_at", None)
        if started is None:
            started = now
            setattr(retry_state, "_hallu_retry_started_at", started)
        if (
            self.retry_deadline_seconds is not None
            and now - float(started) >= self.retry_deadline_seconds
        ):
            raise RetryDeadlineExceeded(
                "retryable request recovery exceeded "
                f"{self.retry_deadline_seconds:.0f}s"
            )
        if is_rate_limit_error(exception):
            started = getattr(retry_state, "_hallu_rate_limit_started_at", None)
            if started is None:
                started = now
                setattr(retry_state, "_hallu_rate_limit_started_at", started)
            if (
                self.rate_limit_retry_deadline_seconds is not None
                and now - float(started) >= self.rate_limit_retry_deadline_seconds
            ):
                raise RateLimitRetryDeadlineExceeded(
                    "continuous HTTP 429 capacity wait exceeded "
                    f"{self.rate_limit_retry_deadline_seconds:.0f}s"
                )
            return False
        setattr(retry_state, "_hallu_rate_limit_started_at", None)
        return (
            self.max_attempts is not None
            and int(getattr(retry_state, "attempt_number", 1)) >= self.max_attempts
        )


class RetryHeartbeat:
    """Emit redacted, machine-readable retry progress to stdout and usage logs."""

    def __init__(
        self,
        component: str,
        usage: Any | None = None,
        progress_callback: Any | None = None,
    ):
        self.component = str(component)
        self.usage = usage
        self.progress_callback = progress_callback

    def __call__(self, retry_state: Any) -> None:
        outcome = getattr(retry_state, "outcome", None)
        exception = outcome.exception() if outcome is not None and outcome.failed else None
        if exception is None:
            return
        if self.usage is not None:
            self.usage.record_retry(self.component, exception)
        next_action = getattr(retry_state, "next_action", None)
        delay = getattr(next_action, "sleep", None)
        status = transient_http_status(exception)
        payload = {
            "event": "llm_retry_wait",
            "component": self.component,
            "reason": f"http_{status}" if status is not None else type(exception).__name__,
            "attempt": int(getattr(retry_state, "attempt_number", 0)),
            "sleep_seconds": round(float(delay or 0.0), 3),
        }
        retry_started = getattr(retry_state, "_hallu_retry_started_at", None)
        if retry_started is not None:
            payload["retry_seconds"] = round(max(0.0, time.monotonic() - float(retry_started)), 3)
        started = getattr(retry_state, "_hallu_rate_limit_started_at", None)
        if started is not None:
            payload["continuous_429_seconds"] = round(max(0.0, time.monotonic() - float(started)), 3)
        print("[heartbeat] " + json.dumps(payload, sort_keys=True), flush=True)
        if self.progress_callback is not None:
            self.progress_callback(dict(payload))
