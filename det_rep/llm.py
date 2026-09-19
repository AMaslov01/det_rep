"""Stateless local vLLM corrector with content-addressed generation cache."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .arms import PROMPT_VERSION, SYSTEM_PROMPT
from .contracts import Generation
from .util import atomic_json, digest, read_json


class LocalLLMError(RuntimeError):
    pass


class OpenAICompatibleCorrector:
    def __init__(
        self, *, base_url: str, model: str, checkpoint: str, cache_dir: str | Path,
        seed: int = 42, temperature: float = 0.0, max_tokens: int = 1024,
        timeout_s: float = 180.0, cache_only: bool = False, api_key_env: str = "DET_REP_VLLM_API_KEY",
        client: Any = None,
    ) -> None:
        if not base_url or not model or not checkpoint:
            raise ValueError("vLLM endpoint, model, and exact checkpoint are required")
        if max_tokens <= 0 or timeout_s <= 0:
            raise ValueError("invalid vLLM output or timeout budget")
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.checkpoint = checkpoint
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.cache_only = cache_only
        self.cache_dir = Path(cache_dir)
        self.api_key_env = api_key_env
        self.client = client
        self.served_model_metadata: dict[str, Any] | None = None
        if not cache_only:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def fingerprint(self) -> str:
        return digest({
            "protocol": "det-rep-vllm-v1", "base_url": self.base_url,
            "model": self.model, "checkpoint": self.checkpoint, "seed": self.seed,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "system_prompt": SYSTEM_PROMPT, "prompt_version": PROMPT_VERSION,
            "served_model": self.served_model_metadata,
        })

    def preflight(self) -> dict[str, Any]:
        """Check that the configured model is actually exposed by this server."""
        import httpx

        headers = {}
        secret = os.environ.get(self.api_key_env)
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        client = self.client or httpx.Client(timeout=self.timeout_s)
        try:
            response = client.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(models, list):
                raise LocalLLMError("vLLM models response has no data list")
            match = next((item for item in models if isinstance(item, dict) and item.get("id") == self.model), None)
            if match is None:
                raise LocalLLMError("configured correction model is not served by vLLM")
            self.served_model_metadata = {key: match.get(key) for key in ("id", "root", "max_model_len")}
            return dict(self.served_model_metadata)
        finally:
            if self.client is None:
                client.close()

    def generate(self, prompt: str, *, arm: str, iteration: int) -> Generation:
        key = digest({"fingerprint": self.fingerprint, "prompt": prompt, "arm": arm, "iteration": iteration})
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            value = read_json(path)
            if value.get("key") != key:
                raise LocalLLMError("generation cache identity mismatch")
            return Generation(**{**value["generation"], "cache_hit": True})
        if self.cache_only:
            raise LocalLLMError("cache-only generation miss")
        import httpx

        headers = {"Content-Type": "application/json"}
        secret = os.environ.get(self.api_key_env)
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
            "stream": False,
        }
        client = self.client or httpx.Client(timeout=self.timeout_s)
        owned = self.client is None
        started = time.monotonic()
        try:
            for attempt in range(3):
                try:
                    response = client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                        time.sleep(min(8.0, 2.0 ** attempt))
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    choice = payload["choices"][0]
                    answer = choice["message"]["content"]
                    finish_reason = str(choice.get("finish_reason") or "unknown")
                    if not isinstance(answer, str) or not answer.strip() or finish_reason == "length":
                        raise LocalLLMError("vLLM returned empty or truncated revision")
                    usage = payload.get("usage") or {}
                    generation = Generation(
                        answer.strip(), usage.get("prompt_tokens"), usage.get("completion_tokens"),
                        time.monotonic() - started, finish_reason,
                    )
                    atomic_json(path, {"key": key, "generation": generation.__dict__})
                    return generation
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    if attempt == 2:
                        raise LocalLLMError(f"vLLM transport failure: {type(exc).__name__}") from exc
                    time.sleep(min(8.0, 2.0 ** attempt))
        finally:
            if owned:
                client.close()
        raise LocalLLMError("vLLM retries exhausted")
