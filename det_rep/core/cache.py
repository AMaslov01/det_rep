"""Shared cache fingerprinting and offline replay safeguards.

The graph and relation-verdict caches are scientific artifacts, not merely
performance optimisations.  A key therefore records every runtime input that
can change structured LLM output.  ``CacheOnlyMissError`` is intentionally a
single exception type so a warm-cache replay can prove that no inference path
was entered.
"""
from __future__ import annotations

import os
import platform
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any


class CacheOnlyMissError(RuntimeError):
    """Raised when an offline replay needs an artifact that is not cached."""

    def __init__(self, component: str, key: str, path: str | Path):
        self.component = component
        self.key = key
        self.path = Path(path)
        super().__init__(
            f"cache-only miss for {component}: key={key} expected={self.path}"
        )


def config_value(section: Any, name: str, default: Any = None) -> Any:
    """Read a value from ``Config``, a mapping, or a test SimpleNamespace."""
    if section is None:
        return default
    if hasattr(section, "get"):
        try:
            return section.get(name, default)
        except TypeError:
            pass
    if isinstance(section, dict):
        return section.get(name, default)
    return getattr(section, name, default)


def _structured_output_request_backend(llm: Any) -> str | None:
    configured = config_value(llm, "structured_output_request_backend")
    if configured is not None:
        return str(configured)
    if (
        config_value(llm, "structured_output_transport", "none") == "response_format"
        and config_value(llm, "structured_output_backend", "none") == "xgrammar"
    ):
        from .dspy_adapter import XGRAMMAR_STRICT_REQUEST_BACKEND

        return XGRAMMAR_STRICT_REQUEST_BACKEND
    return None


@lru_cache(maxsize=1)
def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in (
        "kg-gen",
        "dspy",
        "litellm",
        "pydantic",
        "jsonschema",
        "tenacity",
        "sentence-transformers",
        "transformers",
        "torch",
        "numpy",
    ):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def evaluation_runtime_metadata(cfg: Any) -> dict[str, Any]:
    """Return the reproducibility identity written beside evaluation metrics.

    This is intentionally human-readable rather than hashed.  Cache keys use
    the same LLM identity below, while reports retain the embedding asset and
    device settings needed to explain a byte-identical offline replay.
    """
    matching = cfg.matching
    llm = cfg.llm
    extraction = config_value(cfg, "extraction")
    return {
        "llm_model": config_value(llm, "model"),
        "llm_model_revision": config_value(llm, "model_revision"),
        "runtime_fingerprint": config_value(llm, "runtime_fingerprint"),
        "structured_output_transport": config_value(
            llm, "structured_output_transport", "none"
        ),
        "structured_output_backend": config_value(
            llm, "structured_output_backend", "none"
        ),
        "structured_output_request_backend": _structured_output_request_backend(llm),
        "cluster_context_mode": config_value(
            extraction, "cluster_context_mode", "empty"
        ),
        "cluster_context_protocol": "kggen-native-strict-equivalence-v2",
        "embedding_model": config_value(matching, "embedding_model"),
        "embedding_model_revision": config_value(
            matching, "embedding_model_revision"
        ),
        "embedding_model_path": config_value(matching, "embedding_model_path"),
        "embedding_device": config_value(matching, "embedding_device", "cpu"),
        "embedding_local_files_only": bool(
            config_value(matching, "local_files_only", True)
        ),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
        "python_version": platform.python_version(),
        "client_packages": dict(_installed_versions()),
    }


def llm_runtime_fingerprint(cfg: Any) -> dict[str, Any]:
    """Return the common, JSON-serialisable part of LLM-backed cache keys."""
    llm = cfg.llm
    try:
        from .dspy_adapter import STRUCTURED_OUTPUT_PROTOCOL_VERSION
    except (ImportError, AttributeError):
        # Old configs and fully offline unit tests do not import DSPy.  The
        # marker still invalidates artifacts made before the strict protocol.
        STRUCTURED_OUTPUT_PROTOCOL_VERSION = "legacy-adapter"
    return {
        "model": config_value(llm, "model"),
        "model_revision": config_value(llm, "model_revision"),
        "temperature": config_value(llm, "temperature"),
        "concurrency": config_value(llm, "concurrency"),
        "structured_output_transport": config_value(
            llm, "structured_output_transport", "none"
        ),
        "structured_output_backend": config_value(
            llm, "structured_output_backend", "none"
        ),
        "structured_output_request_backend": _structured_output_request_backend(llm),
        "structured_output_protocol": STRUCTURED_OUTPUT_PROTOCOL_VERSION,
        "runtime_fingerprint": config_value(llm, "runtime_fingerprint"),
        "python_version": platform.python_version(),
        "packages": _installed_versions(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "unset"),
    }
