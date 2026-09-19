"""Arm-independent, complete RAGTruth evidence pack with stable sentence IDs."""
from __future__ import annotations

from .contracts import EvidencePack, EvidenceSentence, Example
from .core.verifier import _sentences
from .util import digest


def build_evidence(example: Example) -> EvidencePack:
    sentences = tuple(
        EvidenceSentence(f"{prefix}{index}", source, index, start, end, text)
        for source, prefix, value in (("context", "C", example.context), ("query", "Q", example.query))
        for index, start, end, text in _sentences(value)
    )
    if not sentences:
        raise ValueError("evidence pack is empty")
    return EvidencePack(example.source_id, sentences, digest({"context": example.context, "query": example.query}))
