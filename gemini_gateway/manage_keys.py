"""Create, list, and revoke recipient keys without keeping raw keys in GCP."""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys

from .keys import KEY_ID_RE, key_digest, parse_key_document


def _read(project_id: str, secret_id: str) -> dict:
    result = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret_id}", f"--project={project_id}"],
        check=True, capture_output=True, text=True,
    )
    raw = result.stdout
    # Validate all records before preserving disabled entries for an update.
    parse_key_document(raw)
    value = json.loads(raw)
    assert isinstance(value, dict)
    return value


def _write(project_id: str, secret_id: str, document: dict) -> None:
    data = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    subprocess.run(
        ["gcloud", "secrets", "versions", "add", secret_id, f"--project={project_id}", "--data-file=-"],
        input=data, check=True, capture_output=True,
    )


def create_key(project_id: str, secret_id: str, key_id: str) -> str:
    if not KEY_ID_RE.fullmatch(key_id):
        raise ValueError("key id must start with a lowercase letter and contain lowercase letters, digits, or hyphens")
    document = _read(project_id, secret_id)
    rows = document["keys"]
    assert isinstance(rows, list)
    if any(isinstance(row, dict) and row.get("id") == key_id for row in rows):
        raise ValueError(f"key id {key_id!r} already exists")
    raw_key = f"gk_{key_id}_{secrets.token_urlsafe(32)}"
    rows.append({"id": key_id, "sha256": key_digest(raw_key)})
    _write(project_id, secret_id, document)
    return raw_key


def revoke_key(project_id: str, secret_id: str, key_id: str) -> bool:
    document = _read(project_id, secret_id)
    rows = document["keys"]
    assert isinstance(rows, list)
    changed = False
    for row in rows:
        if isinstance(row, dict) and row.get("id") == key_id and not row.get("disabled", False):
            row["disabled"] = True
            changed = True
    if changed:
        _write(project_id, secret_id, document)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--secret", default="gemini-gateway-keys")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Print a new recipient key exactly once")
    create.add_argument("key_id")
    revoke = sub.add_parser("revoke", help="Disable one recipient key")
    revoke.add_argument("key_id")
    sub.add_parser("list", help="List key ids and status without exposing any key")
    args = parser.parse_args(argv)
    if args.command == "create":
        print(create_key(args.project, args.secret, args.key_id))
        return 0
    if args.command == "revoke":
        if not revoke_key(args.project, args.secret, args.key_id):
            print(f"no active key named {args.key_id!r}", file=sys.stderr)
            return 1
        print(f"revoked {args.key_id}")
        return 0
    document = _read(args.project, args.secret)
    for row in document["keys"]:
        assert isinstance(row, dict)
        print(f"{row['id']}\t{'disabled' if row.get('disabled', False) else 'active'}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command entry point
    raise SystemExit(main())
