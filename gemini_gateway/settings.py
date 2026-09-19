"""Runtime settings and stable gateway identity."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


LOGICAL_MODEL = "openai/gemini-3.5-flash"
VERTEX_MODEL = "gemini-3.5-flash"
VERTEX_LOCATION = "eu"
PROTOCOL = "hallu-vertex-openai-gateway-v1"


def stable_digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GatewaySettings:
    project_id: str
    key_secret_id: str
    key_cache_ttl_seconds: float = 60.0
    gateway_release: str = "dev"
    cloud_run_revision: str = "local"
    logical_model: str = LOGICAL_MODEL
    vertex_model: str = VERTEX_MODEL
    vertex_location: str = VERTEX_LOCATION

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            project_id=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
            key_secret_id=os.environ.get("GATEWAY_KEY_SECRET", ""),
            key_cache_ttl_seconds=float(os.environ.get("GATEWAY_KEY_CACHE_TTL_SECONDS", "60")),
            gateway_release=os.environ.get("GATEWAY_RELEASE", "dev"),
            cloud_run_revision=os.environ.get("K_REVISION", "local"),
        )

    def manifest(self) -> dict[str, str]:
        return {
            "protocol": PROTOCOL,
            "api_path": "/v1",
            "logical_model": self.logical_model,
            "vertex_model": self.vertex_model,
            "vertex_location": self.vertex_location,
            "gateway_release": self.gateway_release,
            "cloud_run_revision": self.cloud_run_revision,
        }

    @property
    def manifest_sha256(self) -> str:
        return stable_digest(self.manifest())

    @property
    def system_fingerprint(self) -> str:
        return f"{self.manifest_sha256[:16]}:{self.vertex_model}:{self.gateway_release}:{self.cloud_run_revision}"
