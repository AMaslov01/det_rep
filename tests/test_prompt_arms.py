"""The correction prompt is the treatment boundary for the E/C study."""

from det_rep.arms import render_prompt
from det_rep.contracts import ARM_CODES, EvidencePack, EvidenceSentence, Example, FeedbackRecord


def test_four_arms_share_evidence_but_only_reveal_assigned_feedback():
    example = Example("42", "llama31_8b_42", "The source says Paris is in France.", "Where is Paris?", "Paris is in Italy.", 1, "test", "train")
    evidence = EvidencePack("42", (EvidenceSentence("C1", "context", 1, 0, 35, "The source says Paris is in France."),), "source-hash")
    feedback = FeedbackRecord("42", "answer-hash", "evidence-hash", "producer-hash", ({"name": "ENTITY_MARKER"},), ({"text": "CLAIM_MARKER", "verdict": "contradicted"},), "ok", "ok")

    assert ARM_CODES == ("B", "E", "C", "EC")
    prompts = {arm: render_prompt(example, example.original_answer, evidence, None if arm == "B" else feedback, arm) for arm in ARM_CODES}
    assert all("[C1] The source says Paris is in France." in prompt for prompt in prompts.values())
    assert "ENTITY_MARKER" not in prompts["B"] and "CLAIM_MARKER" not in prompts["B"]
    assert "ENTITY_MARKER" in prompts["E"] and "CLAIM_MARKER" not in prompts["E"]
    assert "ENTITY_MARKER" not in prompts["C"] and "CLAIM_MARKER" in prompts["C"]
    assert "ENTITY_MARKER" in prompts["EC"] and "CLAIM_MARKER" in prompts["EC"]
