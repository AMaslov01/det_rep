import json
from pathlib import Path

import pytest
import yaml

from det_rep.core.cache import CacheOnlyMissError
from det_rep.core.critical import (
    CriticalClaimVerifier, CriticalCompletionTruncatedError, CriticalProtocolError,
)
from det_rep.core.veriscore import (
    VeriScorePipeline,
    VerifiableClaimExtractor,
    answer_sentences,
    extraction_messages,
    render_window,
    veriscore_protocol,
)
from det_rep.gemini import runtime_config


MANIFEST = {
    "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
    "logical_model": "openai/gemini-3.5-flash", "vertex_model": "gemini-3.5-flash",
    "vertex_location": "eu", "gateway_release": "fixture", "cloud_run_revision": "fixture",
}
ANSWER = "Frankenstein was written by Mary Shelley. She published it in 1818. I hope this helps!"
CONTEXT = "Frankenstein is a novel by Mary Shelley. It was first published in London in 1818."
QUERY = "who wrote frankenstein"


def build_config(root, *, extractor=None, labels="critical"):
    snapshot = root / "sbert"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}")
    base = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    base["llm"]["request_min_interval_s"] = 0
    base["veriscore"]["claim_extractor"].update(extractor or {})
    base["veriscore"]["claim_verifier"]["labels"] = labels
    return runtime_config(base, MANIFEST, "https://example.invalid", snapshot, root / "cache")


class FakeLLM:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def __call__(self, messages, schema, name, *, max_tokens):
        self.calls.append((messages, name))
        return self.respond(messages, name)


def focus_of(messages):
    return messages[1]["content"].split("Sentence to be focused on: ")[-1]


def paper_claims(messages, name):
    return {"claims": {
        "Frankenstein was written by Mary Shelley.": ["Frankenstein was written by Mary Shelley."],
        "She published it in 1818.": ["Mary Shelley published Frankenstein in 1818.", "No verifiable claim."],
    }.get(focus_of(messages), [])}


def test_answer_sentences_keep_offsets_and_window_context():
    text = " ".join(f"Sentence number {index} is here." for index in range(7))
    sentences = answer_sentences(text)
    assert [sentence.id for sentence in sentences] == [f"A{index}" for index in range(7)]
    assert all(text[sentence.start:sentence.end] == sentence.text for sentence in sentences)
    window = render_window(text, sentences, 5, before=3, after=1)
    assert window == (
        "Sentence number 2 is here. Sentence number 3 is here. Sentence number 4 is here. "
        "<SOS>Sentence number 5 is here.<EOS> Sentence number 6 is here."
    )


def test_window_extraction_is_self_contained_sentence_anchored_and_replayable(tmp_path):
    cfg = build_config(tmp_path)
    extractor = VerifiableClaimExtractor(cfg)
    extractor._call_json = FakeLLM(paper_claims)
    claims = extractor.extract(ANSWER, QUERY)
    assert [(claim.text, claim.sentence_id, claim.sources) for claim in claims] == [
        ("Frankenstein was written by Mary Shelley.", "A0", ("veriscore",)),
        ("Mary Shelley published Frankenstein in 1818.", "A1", ("veriscore",)),
    ]
    assert ANSWER[claims[1].sentence_start:claims[1].sentence_end] == "She published it in 1818."
    assert claims[1].text != ANSWER[claims[1].sentence_start:claims[1].sentence_end]
    assert len(extractor._call_json.calls) == 3
    messages, name = extractor._call_json.calls[1]
    assert name == "verifiable_claims"
    assert "Never extract claims from the question" in messages[0]["content"]
    assert "<SOS>She published it in 1818.<EOS>" in messages[1]["content"]
    assert VerifiableClaimExtractor(cfg, cache_only=True).extract(ANSWER, QUERY) == claims


def test_extraction_protocol_exhaustion_is_not_a_sentence_fallback(tmp_path):
    cfg = build_config(tmp_path)
    extractor = VerifiableClaimExtractor(cfg)
    extractor._call_json = FakeLLM(lambda messages, name: {"claims": "not a list"})
    with pytest.raises(CriticalProtocolError, match="protocol exhausted"):
        extractor.extract("Mary Shelley\nwrote Frankenstein in 1818.")
    assert len(extractor._call_json.calls) == extractor.max_protocol_retries
    with pytest.raises(CacheOnlyMissError):
        VerifiableClaimExtractor(cfg, cache_only=True).extract("Mary Shelley\nwrote Frankenstein in 1818.")


def test_batch_mode_bisects_output_limit_and_falls_back_to_windows(tmp_path):
    cfg = build_config(tmp_path, extractor={"mode": "batch", "max_tokens": 1024, "max_tokens_ceiling": 1024})
    extractor = VerifiableClaimExtractor(cfg)

    def respond(messages, name):
        if name == "verifiable_claims":
            return {"claims": [f"Window claim about {focus_of(messages)}"]}
        ids = messages[1]["content"].split("Focus sentence IDs: ")[-1].split(", ")
        if len(ids) > 1:
            raise CriticalCompletionTruncatedError("output ceiling")
        if ids == ["A2"]:
            return {"sentences": []}
        return {"sentences": [{"id": ids[0], "claims": [f"Batch claim {ids[0]}"]}]}

    extractor._call_json = FakeLLM(respond)
    claims = extractor.extract(ANSWER, QUERY)
    assert [(claim.text, claim.sentence_id) for claim in claims] == [
        ("Batch claim A0", "A0"), ("Batch claim A1", "A1"), ("Window claim about I hope this helps!", "A2"),
    ]
    assert VerifiableClaimExtractor(cfg, cache_only=True).extract(ANSWER, QUERY) == claims


def test_pipeline_always_uses_four_way_verifier_without_entity_filter(tmp_path):
    cfg = build_config(tmp_path)
    pipeline = VeriScorePipeline(cfg)
    assert isinstance(pipeline.verifier, CriticalClaimVerifier)
    pipeline.extractor._call_json = FakeLLM(lambda messages, name: {"claims": ["The rate increased by 5%."]})
    pipeline.verifier._call_json = FakeLLM(lambda messages, name: {"verdict": "unsupported"})
    audits = pipeline.assess("The rate increased by 5%.", "The rate was unchanged.", "How did the rate change?")
    assert [(item["claim"]["text"], item["verdict"]) for item in audits] == [
        ("The rate increased by 5%.", "unsupported")
    ]
    assert audits[0]["claim"]["sentence_id"] == "A0"
    assert audits[0]["claim"]["sentence_start"] == 0
    assert audits[0]["claim"]["sentence_end"] == len("The rate increased by 5%.")
    with pytest.raises(ValueError, match="four-way"):
        VeriScorePipeline(build_config(tmp_path / "binary", labels="binary"))


@pytest.mark.parametrize("verdict", ["entailed", "contradicted", "unsupported", "unknown"])
def test_critical_verifier_accepts_each_pinned_verdict(tmp_path, verdict):
    verifier = CriticalClaimVerifier(build_config(tmp_path))
    verifier._call_json = FakeLLM(lambda messages, name: {"verdict": verdict})
    result = verifier.verify_claim("The rate increased by 5%.", "The rate was unchanged.", None)
    assert result.verdict == verdict
    assert result.protocol_fallback is False


def test_verifier_protocol_exhaustion_is_not_an_unknown_verdict(tmp_path):
    cfg = build_config(tmp_path)
    verifier = CriticalClaimVerifier(cfg)
    verifier._call_json = FakeLLM(lambda messages, name: {"verdict": "maybe"})
    with pytest.raises(CriticalProtocolError, match="protocol exhausted"):
        verifier.verify_claim("Alice lives in Paris.", "Alice lives in Paris.", None)
    with pytest.raises(CacheOnlyMissError):
        CriticalClaimVerifier(cfg, cache_only=True).verify_claim("Alice lives in Paris.", "Alice lives in Paris.", None)
    verifier._load = lambda key: {"verdict": "unknown", "_hallu_protocol_fallback": True}
    with pytest.raises(CriticalProtocolError, match="fallback verdict"):
        verifier.verify_claim("Alice lives in Paris.", "Alice lives in Paris.", None)


def test_batch_transport_exhaustion_does_not_route_to_a_window(tmp_path):
    extractor = VerifiableClaimExtractor(build_config(tmp_path, extractor={"mode": "batch"}))

    def outage(*args, **kwargs):
        raise TimeoutError("gateway unavailable")

    extractor._retry_validated_json = outage
    with pytest.raises(CriticalProtocolError, match="transport exhausted"):
        extractor.extract("Alice lives in Paris. Bob lives in Rome.")


def test_examples_file_changes_prompt_and_protocol_identity(tmp_path):
    examples = {
        "qa": [{"question": "where is paris", "window": "<SOS>Paris is in France.<EOS>", "claims": ["Paris is in France."]}],
        "text": [{"window": "<SOS>Rome is in Italy.<EOS>", "claims": ["Rome is in Italy."]}],
    }
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(examples))
    cfg = build_config(tmp_path / "custom", extractor={"examples_path": str(path)})
    extractor = VerifiableClaimExtractor(cfg)
    system = extraction_messages(extractor.examples["qa"], "<SOS>Berlin is in Germany.<EOS>", "where is berlin")[0]["content"]
    assert 'Output: {"claims": ["Paris is in France."]}' in system
    assert veriscore_protocol(cfg) != veriscore_protocol(build_config(tmp_path / "default"))
