import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from det_rep.contracts import Example
from det_rep.core.config import Config
from det_rep.core.critical import CriticalClaimVerifier, CriticalCompletionTruncatedError
from det_rep.core.extract import Graph
from det_rep.core.matching import DictEmbedder, RefGraph
from det_rep.core.metrics import ScoreResult, score_response, veriscore, veriscore_f1_at_k, veriscore_k
from det_rep.core.veriscore import (
    VeriScoreClaimVerifier,
    VeriScorePipeline,
    VerifiableClaimExtractor,
    answer_sentences,
    extraction_messages,
    render_window,
    veriscore_protocol,
)
from det_rep.core.verifier import EvidenceSpan
from det_rep.evidence import build_evidence
from det_rep.gemini import GeminiFeedbackProducer, runtime_config


MANIFEST = {
    "protocol": "hallu-vertex-openai-gateway-v1", "api_path": "/v1",
    "logical_model": "openai/gemini-3.5-flash", "vertex_model": "gemini-3.5-flash",
    "vertex_location": "eu", "gateway_release": "fixture", "cloud_run_revision": "fixture",
}
ANSWER = "Frankenstein was written by Mary Shelley. She published it in 1818. I hope this helps!"
CONTEXT = "Frankenstein is a novel by Mary Shelley. It was first published in London in 1818."
QUERY = "who wrote frankenstein"


def build_config(root, *, claim_method="veriscore", extractor=None, verifier=None, **veriscore_settings):
    snapshot = root / "sbert"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}")
    base = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())
    base["llm"]["request_min_interval_s"] = 0
    base["claim_method"] = claim_method
    base["veriscore"].update(veriscore_settings)
    base["veriscore"]["claim_extractor"].update(extractor or {})
    base["veriscore"]["claim_verifier"].update(verifier or {})
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


def test_answer_sentences_keep_offsets_and_merge_markers_and_abbreviations():
    text = (
        "Paris is the capital of France. It is known as the City of Light.\n\n"
        "1. The Louvre is in Paris.\n2. Dr. Smith lives in the U.S. and works there."
    )
    sentences = answer_sentences(text)
    assert [sentence.text for sentence in sentences] == [
        "Paris is the capital of France.", "It is known as the City of Light.",
        "1. The Louvre is in Paris.", "2. Dr. Smith lives in the U.S. and works there.",
    ]
    assert [sentence.id for sentence in sentences] == ["A0", "A1", "A2", "A3"]
    assert [sentence.paragraph for sentence in sentences] == [0, 0, 1, 1]
    assert all(text[sentence.start:sentence.end] == sentence.text for sentence in sentences)


def test_window_uses_three_previous_one_next_and_paragraph_lead_for_text_only():
    text = " ".join(f"Sentence number {index} is here." for index in range(7))
    sentences = answer_sentences(text)
    window = render_window(text, sentences, 5, before=3, after=1)
    assert window == (
        "Sentence number 2 is here. Sentence number 3 is here. Sentence number 4 is here. "
        "<SOS>Sentence number 5 is here.<EOS> Sentence number 6 is here."
    )
    assert render_window(text, sentences, 5, before=3, after=1, lead_threshold=5) == (
        "Sentence number 0 is here. [...] " + window
    )
    assert render_window(text, sentences, 3, before=3, after=1, lead_threshold=5).startswith("Sentence number 0 is here. Sentence")


def test_window_extraction_is_self_contained_sentence_anchored_and_replayable(tmp_path):
    cfg = build_config(tmp_path)
    extractor = VerifiableClaimExtractor(cfg)
    extractor._call_json = FakeLLM(paper_claims)
    claims = extractor.extract(ANSWER, QUERY)
    assert [(claim.text, claim.sentence_id, claim.sources) for claim in claims] == [
        ("Frankenstein was written by Mary Shelley.", "A0", ("veriscore",)),
        ("Mary Shelley published Frankenstein in 1818.", "A1", ("veriscore",)),
    ]
    assert ANSWER[claims[1].start:claims[1].end] == "She published it in 1818."
    assert len(extractor._call_json.calls) == 3
    second_messages, name = extractor._call_json.calls[1]
    assert name == "verifiable_claims"
    assert "Never extract claims from the question" in second_messages[0]["content"]
    assert second_messages[1]["content"] == (
        "Question: who wrote frankenstein\n"
        "Response: Frankenstein was written by Mary Shelley. <SOS>She published it in 1818.<EOS> I hope this helps!\n"
        "Sentence to be focused on: She published it in 1818."
    )
    assert VerifiableClaimExtractor(cfg, cache_only=True).extract(ANSWER, QUERY) == claims


def test_extraction_protocol_failure_keeps_the_sentence_as_a_marked_candidate(tmp_path):
    cfg = build_config(tmp_path)
    extractor = VerifiableClaimExtractor(cfg)
    extractor._call_json = FakeLLM(lambda messages, name: {"claims": "not a list"})
    claims = extractor.extract("Mary Shelley\nwrote Frankenstein in 1818.")
    assert [(claim.text, claim.sources) for claim in claims] == [
        ("Mary Shelley", ("veriscore_fallback_sentence",)),
        ("wrote Frankenstein in 1818.", ("veriscore_fallback_sentence",)),
    ]
    assert len(extractor._call_json.calls) == 2 * extractor.max_protocol_retries
    assert VerifiableClaimExtractor(cfg, cache_only=True).extract("Mary Shelley\nwrote Frankenstein in 1818.") == claims


def test_batch_mode_bisects_on_output_limit_and_falls_back_to_windows(tmp_path):
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
    assert "[A1] She published it in 1818." in extractor._call_json.calls[0][0][1]["content"]
    assert VerifiableClaimExtractor(cfg, cache_only=True).extract(ANSWER, QUERY) == claims


def test_verifier_label_sets_map_to_critical_verdicts_and_keep_retrieved_links(tmp_path):
    context, query, claim = "Alice lives in Paris. Bob lives in Rome. The weather is mild.", "Where does Alice live?", "Alice lives in Paris."
    binary = VeriScoreClaimVerifier(build_config(tmp_path / "binary", verifier={"max_evidence_sentences": 2}))
    binary._call_json = FakeLLM(lambda messages, name: {"verdict": "supported"})
    decision = binary.verify_claim(claim, context, query)
    assert (decision.label, decision.verdict, decision.protocol_fallback) == ("supported", "entailed", False)
    messages = binary._call_json.calls[0][0]
    assert messages[1]["content"] == "Claim: Alice lives in Paris.\nEvidence:\n[C0] Alice lives in Paris.\n[C1] Bob lives in Rome."
    assert "- unsupported: the claim is not supported." in messages[0]["content"]
    assert '{"verdict": "contradicted"}' not in messages[0]["content"]
    ternary = VeriScoreClaimVerifier(build_config(
        tmp_path / "ternary", verifier={"labels": "ternary", "evidence_scope": "full_context", "max_evidence_sentences": 2},
    ))
    ternary._call_json = FakeLLM(lambda messages, name: {"verdict": "inconclusive"})
    decision = ternary.verify_claim(claim, context, query)
    assert (decision.label, decision.verdict) == ("inconclusive", "unsupported")
    assert [(span.source, span.index) for span in decision.evidence] == [("context", 0), ("context", 1)]
    assert "[C2] The weather is mild.\n[Q0] Where does Alice live?" in ternary._call_json.calls[0][0][1]["content"]
    assert '{"verdict": "contradicted"}' in ternary._call_json.calls[0][0][0]["content"]


def test_verifier_schema_exhaustion_is_an_explicit_replayable_unknown(tmp_path):
    cfg = build_config(tmp_path)
    verifier = VeriScoreClaimVerifier(cfg)
    verifier._call_json = FakeLLM(lambda messages, name: {"verdict": "maybe"})
    decision = verifier.verify_claim("Alice lives in Paris.", "Alice lives in Paris.", None)
    assert (decision.label, decision.verdict, decision.fallback_reason) == (None, "unknown", "structured_output_exhausted")
    replayed = VeriScoreClaimVerifier(cfg, cache_only=True).verify_claim("Alice lives in Paris.", "Alice lives in Paris.", None)
    assert (replayed.verdict, replayed.cache_hit, replayed.fallback_reason) == ("unknown", True, "structured_output_exhausted")


def test_pipeline_can_add_coverage_review_and_use_the_four_way_verifier(tmp_path):
    cfg = build_config(tmp_path, verifier={"labels": "critical"}, coverage_review=True)
    pipeline = VeriScorePipeline(cfg)
    assert isinstance(pipeline.verifier, CriticalClaimVerifier)
    start = ANSWER.index("in 1818")
    pipeline.extractor._call_json = FakeLLM(lambda messages, name: {"claims": []})
    pipeline.reviewer._call_json = FakeLLM(lambda messages, name: {"claims": [{"text": "in 1818", "start": start, "end": start + 7}]})
    pipeline.verifier._call_json = FakeLLM(lambda messages, name: {"verdict": "contradicted"})
    audits = pipeline.assess(ANSWER, "Frankenstein was published in 1823.", QUERY)
    assert [(audit["claim"]["sentence_id"], audit["claim"]["sources"], audit["label"], audit["verdict"]) for audit in audits] == [
        ("A1", ["global_review"], "contradicted", "contradicted"),
    ]


class FakeExtractor:
    def extract_reference(self, context, query):
        return Graph({"Mary Shelley", "Frankenstein"}, {("Mary Shelley", "wrote", "Frankenstein")}), Graph.empty()

    def extract(self, answer, kind="response"):
        return Graph({"Mary Shelley", "Frankenstein"}, {("Mary Shelley", "wrote", "Frankenstein")})


class FakeRelationVerifier:
    def verify(self, triple, context, query, matching_params=None):
        return SimpleNamespace(verdict="entailed", evidence=(EvidenceSpan("context", 0, 0, 40, "Frankenstein is a novel by Mary Shelley.", 5),), cache_hit=False, protocol_fallback=False, fallback_reason=None)


def test_veriscore_feedback_carries_sentence_anchors_and_paper_labels(tmp_path):
    cfg = build_config(tmp_path)
    producer = GeminiFeedbackProducer(cfg, MANIFEST, tmp_path / "cache" / "feedback")
    producer.extractor = FakeExtractor()
    producer.embedder = DictEmbedder(dim=16)
    producer.relation_verifier = FakeRelationVerifier()
    producer.claim_pipeline.verifier.embedder = None
    producer.claim_pipeline.extractor._call_json = FakeLLM(paper_claims)
    producer.claim_pipeline.verifier._call_json = FakeLLM(
        lambda messages, name: {"verdict": "unsupported" if "1818" in messages[1]["content"].split("\n")[0] else "supported"}
    )
    example = Example("1", "llama31_8b_1", CONTEXT, QUERY, ANSWER, 1, "fixture", "train")
    record = producer.produce(example, ANSWER, build_evidence(example))
    assert [(claim["text"], claim["sentence_id"], claim["label"], claim["verdict"]) for claim in record.claims] == [
        ("Frankenstein was written by Mary Shelley.", "A0", "supported", "entailed"),
        ("Mary Shelley published Frankenstein in 1818.", "A1", "unsupported", "unsupported"),
    ]
    assert record.claims[1]["start"] == ANSWER.index("She published") and record.links[0]["sentence_ids"][0] == "C0"
    critical = GeminiFeedbackProducer(build_config(tmp_path, claim_method="support-critical"), MANIFEST, tmp_path / "cache" / "feedback")
    assert producer.fingerprint != critical.fingerprint
    with pytest.raises(ValueError, match="unknown claim method"):
        GeminiFeedbackProducer(build_config(tmp_path / "bad", claim_method="factscore"), MANIFEST, tmp_path / "bad" / "feedback")


def test_examples_file_replaces_prompt_examples_and_feedback_identity(tmp_path):
    examples = {
        "qa": [{"question": "where is paris", "window": "<SOS>Paris is in France.<EOS>", "claims": ["Paris is in France."]}],
        "text": [{"window": "<SOS>Rome is in Italy.<EOS>", "claims": ["Rome is in Italy."]}],
    }
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(examples))
    cfg = build_config(tmp_path / "custom", extractor={"examples_path": str(path)})
    extractor = VerifiableClaimExtractor(cfg)
    system = extraction_messages(extractor.examples["qa"], "<SOS>Berlin is in Germany.<EOS>", "where is berlin")[0]["content"]
    assert "Output: {\"claims\": [\"Paris is in France.\"]}" in system and "guacamole" not in system
    assert veriscore_protocol(cfg) != veriscore_protocol(build_config(tmp_path / "default"))
    path.write_text(json.dumps({**examples, "text": [{"window": "Rome is in Italy.", "claims": []}]}))
    with pytest.raises(ValueError, match="exactly one focus sentence"):
        VerifiableClaimExtractor(cfg)


def test_veriscore_precision_f1_at_k_and_median_k():
    assert veriscore_f1_at_k(3, 4, 6) == pytest.approx(0.6)
    assert veriscore_f1_at_k(0, 5, 6) == 0.0 and veriscore_f1_at_k(8, 8, 6) == 1.0
    assert veriscore_k([8, 1, 4, 2]) == 2 and veriscore_k([0, 0, 1]) == 1
    assert veriscore([(3, 4), (0, 2)], 6) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        veriscore_f1_at_k(5, 4, 6)
    audits = [{"verdict": "entailed"}, {"verdict": "unsupported"}, {"verdict": "unknown"}, {"verdict": "entailed"}]
    result = ScoreResult(critical={"protocol": "veriscore-v1", "claim_audits": audits})
    assert result.veriscore_precision() == 0.5 and result.veriscore_h() == 0.5
    assert result.veriscore_f1(2) == pytest.approx(2 * 0.5 * 1.0 / 1.5)
    assert ScoreResult().veriscore_h(impute=0.25) == 0.25


def test_score_response_records_the_claim_protocol():
    settings = yaml.safe_load((Path(__file__).parents[1] / "config.example.yaml").read_text())["matching"]
    ref = RefGraph({"Alice"}, set(), Config(settings), DictEmbedder(dim=16))

    class Pipeline:
        protocol = "veriscore-v1"

        def assess(self, answer, context, query, progress_hook=None):
            return [{"claim": {"text": answer}, "verdict": "entailed"}]

    result = score_response(Graph.empty(), ref, answer_text="Alice.", critical_pipeline=Pipeline())
    assert result.critical["protocol"] == "veriscore-v1" and result.veriscore_precision() == 1.0
    del Pipeline.protocol
    assert score_response(Graph.empty(), ref, answer_text="Alice.", critical_pipeline=Pipeline()).critical["protocol"] == "support-critical-v1"
