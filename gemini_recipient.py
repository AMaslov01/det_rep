#!/usr/bin/env python3
"""A dependency-free client for the shared Gemini gateway.

Send this single file and a recipient-specific ``gk_…`` key.  The recipient
needs only Python 3 and can run::

    GEMINI_GATEWAY_API_KEY='gk_alice_…' python3 gemini_recipient.py 'Hello'

For use from another Python file, import :func:`ask_gemini`.  The bearer key is
read from ``GEMINI_GATEWAY_API_KEY`` by default; keeping it out of the file
makes the file safe to update and share without accidentally sharing a key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "openai/gemini-3.5-flash"
# This is a public Cloud Run URL; the bearer key remains the access boundary.
DEFAULT_GATEWAY_URL = "https://gemini-openai-gateway-pilgizlacq-ez.a.run.app"


class GatewayError(RuntimeError):
    """An error response returned by the Gemini gateway."""


def ask_gemini(
    prompt: str,
    *,
    api_key: str | None = None,
    gateway_url: str | None = None,
    system: str | None = None,
    max_tokens: int | None = None,
    timeout_seconds: float = 120,
) -> str:
    """Send one text prompt and return the assistant's text reply.

    ``api_key`` defaults to ``GEMINI_GATEWAY_API_KEY`` and ``gateway_url`` to
    ``GEMINI_GATEWAY_URL`` (or this gateway's stable public URL).  The function
    deliberately implements only the simple text-request path, so the file has
    no third-party dependencies and no access to the sender's repository.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    key = api_key or os.environ.get("GEMINI_GATEWAY_API_KEY")
    if not key:
        raise ValueError("set GEMINI_GATEWAY_API_KEY or pass api_key=")

    base_url = gateway_url or os.environ.get("GEMINI_GATEWAY_URL") or DEFAULT_GATEWAY_URL
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {"model": MODEL, "messages": messages}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    request = Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - caller controls the URL
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise GatewayError(f"gateway returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise GatewayError(f"could not reach gateway: {exc.reason}") from exc

    try:
        decoded = json.loads(body)
        content = decoded["choices"][0]["message"]["content"]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise GatewayError("gateway returned an invalid chat-completions response") from exc
    if not isinstance(content, str):
        raise GatewayError("gateway returned a non-text response")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask Gemini through the shared gateway.")
    parser.add_argument("prompt", help="Text prompt to send")
    parser.add_argument("--api-key", help="Bearer key; defaults to GEMINI_GATEWAY_API_KEY")
    parser.add_argument("--gateway-url", help="Override the deployed gateway URL")
    parser.add_argument("--system", help="Optional system message")
    parser.add_argument("--max-tokens", type=int, help="Maximum output tokens")
    args = parser.parse_args()
    try:
        print(
            ask_gemini(
                args.prompt,
                api_key=args.api_key,
                gateway_url=args.gateway_url,
                system=args.system,
                max_tokens=args.max_tokens,
            )
        )
    except (GatewayError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
