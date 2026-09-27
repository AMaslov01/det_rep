"""The four E/C correction treatments and their isolated prompt renderer."""
from __future__ import annotations

import json

from .contracts import ARM_CODES, EvidencePack, Example, FeedbackRecord


PROMPT_VERSION = "correction-prompt-ec-v1"
SYSTEM_PROMPT = (
    "Revise the answer to the question using only the supplied source evidence. "
    "Correct unsupported facts while preserving supported, answer-relevant information. "
    "Return only the revised answer. Do not mention diagnostics."
)


def visible_feedback(record: FeedbackRecord | None, arm: str) -> dict[str, object]:
    if arm not in ARM_CODES:
        raise ValueError(f"unknown correction arm: {arm}")
    if arm == "B":
        if record is not None:
            raise ValueError("baseline prompt must not depend on feedback")
        return {}
    if record is None:
        raise ValueError(f"{arm} requires feedback")
    if "E" in arm and record.entity_status != "ok":
        raise ValueError("entity feedback is unavailable")
    if "C" in arm and record.claim_status != "ok":
        raise ValueError("claim feedback is unavailable")
    result: dict[str, object] = {}
    if "E" in arm:
        result["entities"] = record.entities
    if "C" in arm:
        result["claims"] = record.claims
    return result


def render_prompt(example: Example, current_answer: str, evidence: EvidencePack, record: FeedbackRecord | None, arm: str) -> str:
    if evidence.source_id != example.source_id or (record is not None and record.source_id != example.source_id):
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
