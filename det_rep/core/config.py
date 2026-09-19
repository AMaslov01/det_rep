"""Configuration loading.

The YAML config is the single source of truth. In particular ``llm.model`` is the
ONLY place the backend LLM is named; no other module hardcodes a model string.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as e:  # pragma: no cover - yaml is a hard dep
    raise RuntimeError("PyYAML is required to load the config") from e


class Config:
    """Attribute-accessible view over a nested dict (``cfg.matching.entity_sim_threshold``).

    Unknown attributes raise ``AttributeError`` so typos fail loudly rather than
    silently returning ``None``.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in self._data.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping")
    return Config(raw)


def resolve_api_key(cfg: Config) -> str | None:
    """Read the API key from the env var named in ``llm.api_key_env``.

    Returns ``None`` if unset (fine for local/ollama models); the extractor will
    surface a clear error if a hosted model actually needs it.
    """
    env_name = getattr(cfg.llm, "api_key_env", None)
    if not env_name:
        return None
    return os.environ.get(env_name)
