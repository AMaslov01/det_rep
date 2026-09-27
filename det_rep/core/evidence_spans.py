"""Stable sentence offsets shared by evidence packing and claim verification."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class EvidenceSpan:
    source: str
    index: int
    start: int
    end: int
    text: str
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "index": self.index, "start": self.start,
            "end": self.end, "text": self.text, "rank": self.rank,
        }


def _sentences(text: str) -> list[tuple[int, int, int, str]]:
    out: list[tuple[int, int, int, str]] = []
    cursor = 0
    for match in _SENTENCE_RE.finditer(text):
        out.extend(_append_sentence(text, cursor, match.start(), len(out)))
        cursor = match.end()
    out.extend(_append_sentence(text, cursor, len(text), len(out)))
    return out


def _append_sentence(text: str, start: int, end: int, index: int) -> list[tuple[int, int, int, str]]:
    piece = text[start:end]
    left = len(piece) - len(piece.lstrip())
    right = len(piece.rstrip())
    if right <= left:
        return []
    actual_start = start + left
    actual_end = start + right
    return [(index, actual_start, actual_end, text[actual_start:actual_end])]
