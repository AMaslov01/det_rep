from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_CLIENT_PATH = Path(__file__).parents[1] / "gemini_recipient.py"
_SPEC = importlib.util.spec_from_file_location("gemini_recipient", _CLIENT_PATH)
assert _SPEC and _SPEC.loader
client = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(client)


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def test_standalone_recipient_client_calls_gateway(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(b'{"choices":[{"message":{"content":"Bonjour"}}]}')

    monkeypatch.setattr(client, "urlopen", fake_urlopen)

    answer = client.ask_gemini(
        "Hello",
        api_key="gk_alice_example",
        gateway_url="https://example.test/",
        system="Be concise",
        max_tokens=64,
    )

    assert answer == "Bonjour"
    assert captured == {
        "url": "https://example.test/v1/chat/completions",
        "headers": {
            "Authorization": "Bearer gk_alice_example",
            "Content-type": "application/json",
            "Accept": "application/json",
        },
        "body": {
            "model": "openai/gemini-3.5-flash",
            "messages": [
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 64,
        },
        "timeout": 120,
    }
