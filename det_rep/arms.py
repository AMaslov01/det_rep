"""The twelve fixed correction treatments and their isolated prompt renderer."""
from __future__ import annotations

import json

from .contracts import ARM_CODES, EvidencePack, Example, FeedbackRecord


PROMPT_VERSION = "correction-prompt-v1"
SYSTEM_PROMPT = (
    "Revise the answer to the question using only the supplied source evidence. "
    "Correct unsupported facts while preserving supported, answer-relevant information. "
    "Return only the revised answer. Do not mention diagnostics."
)


def visible_feedback(record: FeedbackRecord, arm: str) -> dict[str, object]:
    if arm not in ARM_CODES:
        raise ValueError(f"unknown correction arm: {arm}")
    result: dict[str, object] = {}
    for letter, name in (("E", "entities"), ("R", "relations"), ("C", "claims"), ("X", "links")):
        if letter in arm:
            result[name] = getattr(record, name)
    return result


def render_prompt(example: Example, current_answer: str, evidence: EvidencePack, record: FeedbackRecord, arm: str) -> str:
    if record.source_id != example.source_id or evidence.source_id != example.source_id:
        raise ValueError("cross-source feedback or evidence")
    blocks = visible_feedback(record, arm)
    prompt = (
        f"Question:\n{example.query}\n\n"
        f"Source evidence (the same for every revision condition):\n{evidence.render()}\n\n"
        f"Current answer:\n{current_answer}\n\n"
    )
    if blocks:
        prompt += "Factual diagnostics:\n" + json.dumps(blocks, ensure_ascii=False, sort_keys=True) + "\n\n"
    return prompt + "Revised answer:"
