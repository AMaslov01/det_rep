"""Versioned hand-off records for the six research roles.

Scientific producers implement these protocols. Test fakes belong in tests only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


SCHEMA_VERSION = "det-rep-v1"
ARM_CODES = ("B", "E", "R", "ER", "C", "EC", "RC", "ERC", "CX", "ECX", "RCX", "ERCX")


@dataclass(frozen=True)
class Example:
    source_id: str
    response_id: str
    context: str
    query: str
    original_answer: str
    label: int
    annotation_model: str
    split: str

    def __post_init__(self) -> None:
        if self.split not in {"train", "test"} or self.label not in {0, 1}:
            raise ValueError("example requires a fixed split and binary annotation")
        if not self.source_id or not self.original_answer.strip() or not self.context.strip():
            raise ValueError("example requires source, context, and answer")


@dataclass(frozen=True)
class EvidenceSentence:
    id: str
    source: str
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class EvidencePack:
    source_id: str
    sentences: tuple[EvidenceSentence, ...]
    source_sha256: str
    schema_version: str = SCHEMA_VERSION

    def render(self) -> str:
        return "\n".join(f"[{part.id}] {part.text}" for part in self.sentences)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackRecord:
    source_id: str
    answer_sha256: str
    evidence_sha256: str
    producer_fingerprint: str
    entities: tuple[dict[str, Any], ...]
    relations: tuple[dict[str, Any], ...]
    claims: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    graph_status: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeedbackRecord":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("feedback schema mismatch")
        return cls(**{**value, **{name: tuple(value[name]) for name in ("entities", "relations", "claims", "links")}})


@dataclass(frozen=True)
class Generation:
    answer: str
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    finish_reason: str
    cache_hit: bool = False


@dataclass(frozen=True)
class TrajectoryStep:
    iteration: int
    input_answer: str
    feedback_sha256: str
    prompt_sha256: str
    generation: Generation


@dataclass(frozen=True)
class Trajectory:
    source_id: str
    response_id: str
    arm: str
    split: str
    original_answer: str
    steps: tuple[TrajectoryStep, ...]
    run_fingerprint: str
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRequest:
    """Blind evaluator input: opaque ID, with no source or treatment identity."""
    request_id: str
    context: str
    query: str
    original_answer: str
    revised_answer: str
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class EvaluationResult:
    request_id: str
    original_claims: tuple[dict[str, Any], ...]
    revised_claims: tuple[dict[str, Any], ...]
    alignments: tuple[dict[str, Any], ...]
    verdicts: tuple[dict[str, Any], ...]
    completeness: dict[str, Any]
    schema_version: str = SCHEMA_VERSION


class FeedbackProducer(Protocol):
    def produce(self, example: Example, answer: str, evidence: EvidencePack) -> FeedbackRecord: ...


class Corrector(Protocol):
    def generate(self, prompt: str, *, arm: str, iteration: int) -> Generation: ...


class ClaimAligner(Protocol):  # participant 2
    """Extract original/revised atomic claims and align corresponding claims."""
    def align(self, request: EvaluationRequest) -> tuple[dict[str, Any], ...]: ...


class FrozenEvaluator(Protocol):  # participant 3
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


class SpanAssessor(Protocol):  # participant 4
    def assess(self, request: EvaluationRequest, *, original_spans: tuple[dict[str, Any], ...]) -> dict[str, Any]: ...


class AnswerModelKG(Protocol):  # participant 5; distinct from Gemini cache namespace
    def extract(self, text: str, *, model_fingerprint: str) -> dict[str, Any]: ...


class AnswerSource(Protocol):  # participant 6
    def load(self) -> tuple[Example, ...]: ...
