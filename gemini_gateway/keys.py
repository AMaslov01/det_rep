"""Revocable bearer-key verification backed by Secret Manager."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


KEY_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


class KeyDocumentError(ValueError):
    """The Secret Manager key document is malformed."""


@dataclass(frozen=True)
class KeyRecord:
    key_id: str
    sha256: str


def key_digest(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def parse_key_document(value: str | bytes | dict[str, Any]) -> dict[str, KeyRecord]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise KeyDocumentError("key secret is not JSON") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise KeyDocumentError("key secret must have version 1")
    rows = value.get("keys")
    if not isinstance(rows, list):
        raise KeyDocumentError("key secret must contain a keys list")
    result: dict[str, KeyRecord] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise KeyDocumentError("key entry is not an object")
        key_id, digest = row.get("id"), row.get("sha256")
        if not isinstance(key_id, str) or not KEY_ID_RE.fullmatch(key_id):
            raise KeyDocumentError("key entry has an invalid id")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise KeyDocumentError("key entry has an invalid sha256 digest")
        if key_id in result:
            raise KeyDocumentError("key entry ids must be unique")
        if not bool(row.get("disabled", False)):
            result[key_id] = KeyRecord(key_id, digest)
    return result


class SecretManagerKeySource:
    """Read only the latest key document; the runtime never writes secrets."""

    def __init__(self, project_id: str, secret_id: str) -> None:
        self.project_id = project_id
        self.secret_id = secret_id
        self._client: Any | None = None

    def __call__(self) -> str:
        if not self.project_id or not self.secret_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT and GATEWAY_KEY_SECRET are required")
        if self._client is None:
            try:
                from google.cloud import secretmanager
            except ImportError as exc:  # pragma: no cover - deployment dependency
                raise RuntimeError("google-cloud-secret-manager is required") from exc
            self._client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{self.project_id}/secrets/{self.secret_id}/versions/latest"
        return self._client.access_secret_version(request={"name": name}).payload.data.decode("utf-8")


class KeyStore:
    """Cache secret reads briefly while permitting prompt, deploy-free revocation."""

    def __init__(self, source: Callable[[], str | bytes | dict[str, Any]], ttl_seconds: float = 60.0, *, now: Callable[[], float] = time.monotonic) -> None:
        if ttl_seconds <= 0:
            raise ValueError("key cache TTL must be positive")
        self.source = source
        self.ttl_seconds = ttl_seconds
        self.now = now
        self._records: dict[str, KeyRecord] | None = None
        self._loaded_at = float("-inf")

    def _load(self) -> dict[str, KeyRecord]:
        if self._records is None or self.now() - self._loaded_at >= self.ttl_seconds:
            self._records = parse_key_document(self.source())
            self._loaded_at = self.now()
        return self._records

    def authenticate(self, authorization: str | None) -> str | None:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        raw_key = authorization.removeprefix("Bearer ").strip()
        parts = raw_key.split("_", 2)
        if len(parts) != 3 or parts[0] != "gk" or not KEY_ID_RE.fullmatch(parts[1]) or not parts[2]:
            return None
        record = self._load().get(parts[1])
        if record is None:
            return None
        return record.key_id if hmac.compare_digest(record.sha256, key_digest(raw_key)) else None
