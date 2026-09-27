import json

import pytest

from det_rep.__main__ import pinned_gateway_manifest


MANIFEST = {
    "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
    "logical_model": "openai/gemini-3.5-flash", "vertex_model": "gemini-3.5-flash",
    "vertex_location": "eu", "gateway_release": "release", "cloud_run_revision": "revision",
}


def test_resume_uses_pinned_gateway_identity_without_live_gateway(tmp_path, monkeypatch):
    (tmp_path / "gateway_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    monkeypatch.setattr("det_rep.gemini.fetch_gateway_manifest", lambda *_: (_ for _ in ()).throw(RuntimeError("unreachable")))
    assert pinned_gateway_manifest(tmp_path, "https://gateway.example", MANIFEST["logical_model"]) == MANIFEST


def test_rejects_conflicting_supplied_gateway_manifest(tmp_path):
    (tmp_path / "gateway_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    alternate = {**MANIFEST, "cloud_run_revision": "other"}
    path = tmp_path / "other.json"
    path.write_text(json.dumps(alternate), encoding="utf-8")
    with pytest.raises(ValueError, match="different gateway revision"):
        pinned_gateway_manifest(tmp_path, "https://gateway.example", MANIFEST["logical_model"], path)
