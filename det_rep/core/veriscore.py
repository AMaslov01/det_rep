"""Verifiable-claim extraction and verification after VeriScore (Song, Kim, and Iyyer, 2024)."""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .cache import CacheOnlyMissError, config_value
from .critical import (
    CriticalClaimVerifier,
    CriticalOutputLimitError,
    CriticalProtocolError,
    _CachedComponent,
)
from .dspy_adapter import (
    StructuredOutputParseError,
    StructuredOutputSchemaError,
    is_retryable_llm_exception,
    validate_json_document,
)
from .matching import Embedder, normalize
from .retry import RequestPacer
from .evidence_spans import _sentences


VERISCORE_PROTOCOL = "det-rep-veriscore-ec-v2"
NO_VERIFIABLE_CLAIM = "No verifiable claim."
CLAIM_LIST_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "string", "minLength": 1}}
CLAIM_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"claims": CLAIM_LIST_SCHEMA},
    "required": ["claims"],
    "additionalProperties": False,
}

_FALLBACK_FIELD = "_hallu_fallback_sentence"
_ROUTE_FIELD = "_hallu_route"
_FOCUS_RE = re.compile(r"<SOS>(.*?)<EOS>", re.DOTALL)
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_LIST_MARKER_RE = re.compile(r"^(?:\(?(?:\d{1,3}|[a-z]|[ivxlc]{1,6})[.)]|[-*•]|#{1,6})$", re.IGNORECASE)
_ABBREVIATIONS = frozenset({
    "a.m.", "approx.", "co.", "col.", "dept.", "dr.", "e.g.", "est.", "fig.", "gen.", "i.e.", "inc.",
    "jr.", "lt.", "ltd.", "mr.", "mrs.", "ms.", "mt.", "no.", "p.m.", "prof.", "sgt.", "sr.", "st.",
    "u.k.", "u.s.", "vs.",
})
_EXTRACTION_RULES = (
    "You are checking how factual a piece of text is. Break the sentence marked between <SOS> and <EOS> into "
    "verifiable claims: fine-grained facts that could be confirmed or refuted against reliable external sources.\n"
    "- A claim describes one event or one state together with every modifier that identifies it in the real "
    "world: time, location, quantity, comparison, condition, negation, and relative clauses.\n"
    "- Extract as many fine-grained verifiable claims as the marked sentence contains. Focus on its named "
    "entities and numbers.\n"
    "- Do not extract stories, personal experiences, subjective opinions, hypotheticals or counterfactuals "
    "(for example \"would have been\"), suggestions, advice, instructions, greetings, or remarks about the text "
    "itself. Biographical, historical, scientific, and other factual statements are not stories or personal "
    "experiences; extract claims from them.\n"
    "- The other sentences are context only. Use them to resolve pronouns, definite descriptions (for example "
    "\"the drug\" or \"the company\"), and omitted subjects, but never extract claims from them.\n"
    "- Every claim must be understandable on its own. Refer to every entity by name instead of a pronoun. If a "
    "definite description is unavoidable, add modifiers that identify the entity.\n"
    "- Situate each claim in time and place whenever the text provides them.\n"
    "- Keep each claim to one sentence with at most one embedded clause. Copy quotations verbatim together with "
    "their speaker or source. Ignore listed references.\n"
    "- If the marked sentence contains no verifiable claim, return an empty claims list."
)
_QUESTION_RULES = (
    "The text is a response to a question. Never extract claims from the question; use it, like the other "
    "sentences, only as context. Relate each claim to the question whenever the marked sentence relies on it."
)
_BATCH_RULES = (
    "Apply the rules to every sentence whose ID is listed as a focus sentence, treating that sentence as the "
    "marked sentence and the rest of the response as context. Return exactly one entry for every focus sentence "
    "ID, with an empty claims list when the sentence has no verifiable claim. Each example shows one marked "
    "sentence."
)
@dataclass(frozen=True)
class ExtractionExample:
    window: str
    claims: tuple[str, ...]
    question: str = ""


DEFAULT_EXTRACTION_EXAMPLES: dict[str, tuple[ExtractionExample, ...]] = {
    "qa": (
        ExtractionExample(
            "Guacamole is a dip that originated in Mexico. <SOS>Its main ingredient is mashed avocado, which is "
            "usually mixed with lime juice and salt.<EOS> Some recipes also add onion and cilantro.",
            (
                "The main ingredient of guacamole is mashed avocado.",
                "In guacamole, mashed avocado is usually mixed with lime juice.",
                "In guacamole, mashed avocado is usually mixed with salt.",
            ),
            "what is guacamole made of",
        ),
        ExtractionExample(
            "To renew a U.S. passport by mail, you will generally need the following documents:\n<SOS>1. Form "
            "DS-82, the application for passport renewal by mail.<EOS>\n2. Your most recent passport.",
            (
                "Form DS-82 is needed to renew a U.S. passport by mail.",
                "Form DS-82 is the application for U.S. passport renewal by mail.",
            ),
            "what do i need to renew my passport by mail",
        ),
        ExtractionExample(
            "Raw cookie dough often contains uncooked eggs and raw flour. <SOS>To be on the safe side, I would "
            "recommend baking the dough before you eat it.<EOS>",
            (),
            "is it safe to eat raw cookie dough",
        ),
        ExtractionExample(
            "A broken wrist is one of the most common fractures. <SOS>It usually takes six to eight weeks to heal "
            "in adults, but recovery may take longer in older patients.<EOS> A cast is typically worn during this "
            "time.",
            (
                "A broken wrist usually takes six to eight weeks to heal in adults.",
                "Recovery from a broken wrist may take longer in older patients.",
            ),
            "how long does it take for a broken wrist to heal",
        ),
        ExtractionExample(
            "<SOS>Based on the given passages, here is the answer to your question.<EOS> Frankenstein was written "
            "by Mary Shelley.",
            (),
            "who wrote frankenstein",
        ),
        ExtractionExample(
            "Based on the given passages, here is the answer to your question. Frankenstein was written by Mary "
            "Shelley. <SOS>She first published it anonymously in London in 1818.<EOS> The novel is often "
            "considered an early work of science fiction.",
            (
                "Mary Shelley first published Frankenstein anonymously.",
                "Mary Shelley first published Frankenstein in London in 1818.",
            ),
            "who wrote frankenstein",
        ),
        ExtractionExample(
            "The Dead Sea lies in a desert basin with no outlet to the ocean. <SOS>Water leaves it only by "
            "evaporation, which leaves the salt behind, and honestly, floating on it would be an amazing "
            "experience.<EOS>",
            (
                "Water leaves the Dead Sea only by evaporation.",
                "Evaporation of water from the Dead Sea leaves the salt behind.",
            ),
            "why is the dead sea so salty",
        ),
    ),
    "text": (
        ExtractionExample(
            "The Amazon River flows through South America. <SOS>It carries more water than any other river in the "
            "world, discharging about 209,000 cubic meters per second into the Atlantic Ocean.<EOS> Its drainage "
            "basin covers parts of several countries.",
            (
                "The Amazon River carries more water than any other river in the world.",
                "The Amazon River discharges about 209,000 cubic meters of water per second.",
                "The Amazon River discharges into the Atlantic Ocean.",
            ),
        ),
        ExtractionExample(
            "Last summer my sister and I drove along the coast for a week. <SOS>We stopped at a small cafe where she "
            "told me she was thinking about changing jobs.<EOS> It was the best trip we have taken together.",
            (),
        ),
        ExtractionExample(
            "John F. Kennedy was inaugurated as president of the United States in January 1961. <SOS>In his "
            "inaugural address he said \"ask not what your country can do for you\", and had he given the speech "
            "a decade later, it might have been received very differently.<EOS>",
            (
                "John F. Kennedy said \"ask not what your country can do for you\" in his inaugural address in "
                "January 1961.",
            ),
        ),
        ExtractionExample(
            "Nokia was founded in 1865 as a paper mill in southwestern Finland. [...] Over the following century it "
            "expanded into rubber and cables. <SOS>In the 1990s the company shifted its focus to mobile phones and "
            "became the world's largest mobile phone maker by 1998.<EOS>",
            (
                "Nokia shifted its focus to mobile phones in the 1990s.",
                "Nokia became the world's largest mobile phone maker by 1998.",
            ),
        ),
        ExtractionExample(
            "Preheat the oven to 180 degrees Celsius. <SOS>Mix the flour and sugar in a large bowl, then add the "
            "eggs one at a time.<EOS> Bake for 25 minutes.",
            (),
        ),
    ),
}

@dataclass(frozen=True)
class AnswerSentence:
    id: str
    index: int
    start: int
    end: int
    text: str
    paragraph: int


@dataclass(frozen=True)
class VerifiableClaim:
    text: str
    sentence_id: str | None
    sentence_start: int
    sentence_end: int
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text, "sentence_id": self.sentence_id,
            "sentence_start": self.sentence_start, "sentence_end": self.sentence_end,
            "sources": list(self.sources),
        }


def answer_sentences(text: str, min_chars: int = 10) -> list[AnswerSentence]:
    pieces = [(start, end) for _, start, end, _ in _sentences(text)]
    spans: list[tuple[int, int]] = []
    begin: int | None = None
    for position, (start, end) in enumerate(pieces):
        begin = start if begin is None else begin
        if position + 1 < len(pieces) and _joins_next(text, begin, end, pieces[position + 1][0], min_chars):
            continue
        spans.append((begin, end))
        begin = None
    breaks = [match.start() for match in _PARAGRAPH_RE.finditer(text)]
    return [
        AnswerSentence(f"A{index}", index, start, end, text[start:end], sum(1 for point in breaks if point < start))
        for index, (start, end) in enumerate(spans)
    ]


def _joins_next(text: str, start: int, end: int, following: int, min_chars: int) -> bool:
    piece = text[start:end]
    if _LIST_MARKER_RE.match(piece):
        return True
    if "\n" in text[end:following]:
        return False
    return len(piece) < min_chars or piece.split()[-1].lower() in _ABBREVIATIONS or text[following].islower()


def render_window(
    text: str,
    sentences: Sequence[AnswerSentence],
    position: int,
    *,
    before: int,
    after: int,
    lead_threshold: int | None = None,
) -> str:
    focus = sentences[position]
    first = sentences[max(0, position - before)]
    last = sentences[min(len(sentences) - 1, position + after)]
    window = f"{text[first.start:focus.start]}<SOS>{focus.text}<EOS>{text[focus.end:last.end]}"
    if lead_threshold is not None:
        paragraph = [sentence for sentence in sentences if sentence.paragraph == focus.paragraph]
        if len(paragraph) > lead_threshold and paragraph[0].index < first.index:
            window = f"{paragraph[0].text} [...] {window}"
    return window


def load_extraction_examples(path: str | Path) -> dict[str, tuple[ExtractionExample, ...]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"qa", "text"}:
        raise ValueError("VeriScore extraction examples need exactly the qa and text lists")
    examples = {
        kind: tuple(
            ExtractionExample(
                str(item["window"]), tuple(str(claim) for claim in item["claims"]), str(item.get("question", ""))
            )
            for item in raw[kind]
        )
        for kind in ("qa", "text")
    }
    for kind, items in examples.items():
        if not items:
            raise ValueError(f"VeriScore {kind} examples must not be empty")
        for example in items:
            _focus_sentence(example.window)
            if (kind == "qa") != bool(example.question.strip()):
                raise ValueError("only VeriScore qa examples carry a question")
    return examples


def _focus_sentence(window: str) -> str:
    found = _FOCUS_RE.findall(window)
    if len(found) != 1 or window.count("<SOS>") != 1 or window.count("<EOS>") != 1:
        raise ValueError("a VeriScore window must mark exactly one focus sentence")
    return found[0]


def _is_qa(question: str | None) -> bool:
    return bool((question or "").strip())


def _extraction_request(window: str, question: str | None) -> str:
    head = f"Question: {question}\nResponse: {window}" if _is_qa(question) else f"Text: {window}"
    return f"{head}\nSentence to be focused on: {_focus_sentence(window)}"


def _render_extraction_example(number: int, example: ExtractionExample) -> str:
    return (
        f"Example {number}\n{_extraction_request(example.window, example.question)}\n"
        f"Output: {json.dumps({'claims': list(example.claims)}, ensure_ascii=False)}"
    )


def _extraction_system(examples: Sequence[ExtractionExample], question: str | None, *extra: str) -> str:
    return "\n\n".join([
        _EXTRACTION_RULES,
        *([_QUESTION_RULES] if _is_qa(question) else []),
        *extra,
        "Examples:",
        *(_render_extraction_example(number, example) for number, example in enumerate(examples, start=1)),
    ])


def extraction_messages(
    examples: Sequence[ExtractionExample], window: str, question: str | None
) -> list[dict[str, str]]:
    system = _extraction_system(examples, question)
    return [
        {"role": "system", "content": system + '\n\nReturn a JSON object {"claims": [...]} with one string per claim.'},
        {"role": "user", "content": _extraction_request(window, question)},
    ]


def batch_messages(
    examples: Sequence[ExtractionExample],
    sentences: Sequence[AnswerSentence],
    focus: Sequence[str],
    question: str | None,
) -> list[dict[str, str]]:
    system = _extraction_system(examples, question, _BATCH_RULES)
    lines: list[str] = []
    for previous, sentence in zip([None, *sentences], sentences):
        if previous is not None and previous.paragraph != sentence.paragraph:
            lines.append("")
        lines.append(f"[{sentence.id}] {sentence.text}")
    head = f"Question: {question}\n" if _is_qa(question) else ""
    return [
        {
            "role": "system",
            "content": system + '\n\nReturn a JSON object {"sentences": [{"id": ..., "claims": [...]}]} with one '
            "entry per focus sentence ID.",
        },
        {
            "role": "user",
            "content": f"{head}Response sentences:\n" + "\n".join(lines) + f"\n\nFocus sentence IDs: {', '.join(focus)}",
        },
    ]


def _batch_schema(ids: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string", "enum": list(ids)}, "claims": CLAIM_LIST_SCHEMA},
                    "required": ["id", "claims"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sentences"],
        "additionalProperties": False,
    }


def _clean_claims(values: Sequence[Any]) -> list[str]:
    texts: list[str] = []
    seen = {normalize(NO_VERIFIABLE_CLAIM)}
    for value in values:
        text = " ".join(str(value).split())
        key = normalize(text)
        if key and key not in seen:
            seen.add(key)
            texts.append(text)
    return texts


def _claim_texts(payload: Any) -> list[str]:
    validate_json_document(payload, CLAIM_TEXT_SCHEMA)
    return _clean_claims(payload["claims"])


def _batch_claim_texts(payload: Any, ids: Sequence[str]) -> dict[str, list[str]]:
    validate_json_document(payload, _batch_schema(ids))
    found: dict[str, list[str]] = {}
    for entry in payload["sentences"]:
        if entry["id"] in found:
            raise StructuredOutputParseError("VeriScore batch repeats a focus sentence")
        found[entry["id"]] = _clean_claims(entry["claims"])
    if set(found) != set(ids):
        raise StructuredOutputParseError("VeriScore batch omits a focus sentence")
    return found


def _batch_outcome(payload: dict[str, Any], ids: Sequence[str]) -> tuple[str, dict[str, list[str]]]:
    route = payload.get(_ROUTE_FIELD)
    if route is None:
        return "claims", _batch_claim_texts(payload, ids)
    if route not in {"bisect", "window"} or len(payload) != 1:
        raise StructuredOutputParseError("VeriScore batch cache has a malformed route")
    return route, {}


def _sentence_claims(sentence: AnswerSentence, texts: Sequence[str]) -> list[VerifiableClaim]:
    return [VerifiableClaim(text, sentence.id, sentence.start, sentence.end, ("veriscore",)) for text in texts]


class VerifiableClaimExtractor(_CachedComponent):
    component = "veriscore_claim_extractor"
    protocol = "det-rep-veriscore-claims-v2"
    config_root = "veriscore"
    repair_message = (
        "The preceding structured answer was rejected. Return one JSON object matching the given schema exactly, "
        "with one self-contained claim per string."
    )

    def __init__(
        self, cfg, usage=None, *, cache_only: bool = False, request_pacer: RequestPacer | None = None
    ):
        super().__init__(cfg, "claim_extractor", usage, cache_only=cache_only, request_pacer=request_pacer)
        self.mode = str(config_value(self.section, "mode", "window"))
        self.context_before = int(config_value(self.section, "context_before", 3))
        self.context_after = int(config_value(self.section, "context_after", 1))
        self.paragraph_lead_threshold = int(config_value(self.section, "paragraph_lead_threshold", 5))
        self.min_sentence_chars = int(config_value(self.section, "min_sentence_chars", 10))
        examples_path = config_value(self.section, "examples_path")
        self.examples = load_extraction_examples(examples_path) if examples_path else DEFAULT_EXTRACTION_EXAMPLES
        if self.mode not in {"window", "batch"}:
            raise ValueError("veriscore.claim_extractor.mode must be window or batch")
        if min(self.context_before, self.context_after, self.paragraph_lead_threshold) < 0:
            raise ValueError("veriscore.claim_extractor window limits must be non-negative")
        if self.min_sentence_chars <= 0:
            raise ValueError("veriscore.claim_extractor.min_sentence_chars must be positive")

    def extract(self, response: str, query: str | None = None) -> list[VerifiableClaim]:
        sentences = answer_sentences(response, self.min_sentence_chars)
        if self.mode == "batch" and sentences:
            found = self._extract_batch(response, sentences, list(range(len(sentences))), query)
            return [claim for sentence in sentences for claim in found[sentence.id]]
        return [
            claim
            for position in range(len(sentences))
            for claim in self._extract_window(response, sentences, position, query)
        ]

    def _examples(self, query: str | None) -> tuple[ExtractionExample, ...]:
        return self.examples["qa" if _is_qa(query) else "text"]

    def _load_texts(self, key: str) -> list[str] | None:
        cached = self._load(key)
        if cached is None:
            return None
        if cached.get(_FALLBACK_FIELD):
            raise CriticalProtocolError("VeriScore claim cache contains a sentence fallback")
        try:
            texts = _claim_texts({"claims": cached.get("claims")})
        except (StructuredOutputParseError, StructuredOutputSchemaError) as exc:
            if self.cache_only:
                raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json") from exc
            return None
        self._record(key, 0.0, cached=True)
        return texts

    def _extract_window(
        self, response: str, sentences: Sequence[AnswerSentence], position: int, query: str | None
    ) -> list[VerifiableClaim]:
        sentence = sentences[position]
        window = render_window(
            response, sentences, position, before=self.context_before, after=self.context_after,
            lead_threshold=None if _is_qa(query) else self.paragraph_lead_threshold,
        )
        messages = extraction_messages(self._examples(query), window, query)
        key = self._cache_key({"mode": "window", "messages": messages})
        cached = self._load_texts(key)
        if cached is not None:
            return _sentence_claims(sentence, cached)
        if self.cache_only:
            raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
        start = time.perf_counter()
        try:
            texts = self._retry_validated_json(messages, CLAIM_TEXT_SCHEMA, "verifiable_claims", _claim_texts)
        except (StructuredOutputParseError, StructuredOutputSchemaError) as exc:
            raise CriticalProtocolError("verifiable claim extraction protocol exhausted") from exc
        except Exception as exc:
            if is_retryable_llm_exception(exc):
                raise CriticalProtocolError("verifiable claim extraction transport exhausted") from exc
            raise CriticalProtocolError("verifiable claim extraction failed") from exc
        self._save(key, {"claims": texts})
        self._record(key, time.perf_counter() - start, cached=False)
        return _sentence_claims(sentence, texts)

    def _load_batch(self, key: str, ids: Sequence[str]) -> tuple[str, dict[str, list[str]]] | None:
        cached = self._load(key)
        if cached is None:
            return None
        try:
            outcome = _batch_outcome(cached, ids)
        except (StructuredOutputParseError, StructuredOutputSchemaError) as exc:
            if self.cache_only:
                raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json") from exc
            return None
        self._record(key, 0.0, cached=True)
        return outcome

    def _request_batch(self, messages: list[dict[str, str]], ids: Sequence[str]) -> dict[str, Any]:
        try:
            found = self._retry_validated_json(
                messages, _batch_schema(ids), "verifiable_claims_batch",
                lambda payload: _batch_claim_texts(payload, ids),
            )
        except CriticalOutputLimitError:
            return {_ROUTE_FIELD: "bisect" if len(ids) > 1 else "window"}
        except (StructuredOutputParseError, StructuredOutputSchemaError):
            return {_ROUTE_FIELD: "window"}
        except Exception as exc:
            if is_retryable_llm_exception(exc):
                raise CriticalProtocolError("verifiable claim extraction transport exhausted") from exc
            raise CriticalProtocolError("verifiable claim extraction failed") from exc
        return {"sentences": [{"id": sentence_id, "claims": found[sentence_id]} for sentence_id in ids]}

    def _extract_batch(
        self, response: str, sentences: Sequence[AnswerSentence], positions: Sequence[int], query: str | None
    ) -> dict[str, list[VerifiableClaim]]:
        focus = [sentences[position] for position in positions]
        ids = [sentence.id for sentence in focus]
        messages = batch_messages(self._examples(query), sentences, ids, query)
        key = self._cache_key({"mode": "batch", "messages": messages})
        outcome = self._load_batch(key, ids)
        if outcome is None:
            if self.cache_only:
                raise CacheOnlyMissError(self.component, key, self.cache_dir / f"{key}.json")
            start = time.perf_counter()
            payload = self._request_batch(messages, ids)
            self._save(key, payload)
            self._record(key, time.perf_counter() - start, cached=False)
            outcome = _batch_outcome(payload, ids)
        route, found = outcome
        if route == "claims":
            return {sentence.id: _sentence_claims(sentence, found[sentence.id]) for sentence in focus}
        if route == "bisect" and len(positions) > 1:
            middle = len(positions) // 2
            return {
                **self._extract_batch(response, sentences, positions[:middle], query),
                **self._extract_batch(response, sentences, positions[middle:], query),
            }
        return {
            sentences[position].id: self._extract_window(response, sentences, position, query)
            for position in positions
        }


class VeriScorePipeline:
    protocol = VERISCORE_PROTOCOL

    def __init__(self, cfg, usage=None, *, cache_only: bool = False, embedder: Embedder | None = None):
        section = getattr(cfg, "veriscore", None)
        if section is None:
            raise ValueError("veriscore config is required")
        if str(config_value(section.claim_verifier, "labels", "")) != "critical":
            raise ValueError("VeriScore correction requires the four-way critical claim verifier")
        pacer = RequestPacer(float(getattr(cfg.llm, "request_min_interval_s", 0)))
        self.extractor = VerifiableClaimExtractor(cfg, usage, cache_only=cache_only, request_pacer=pacer)
        self.verifier = CriticalClaimVerifier(
            cfg, usage, cache_only=cache_only, embedder=embedder, request_pacer=pacer
        )

    def assess(
        self,
        response: str,
        context: str,
        query: str | None,
        *,
        progress_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        def emit(phase: str, completed: int = 0, total: int = 0) -> None:
            if progress_hook is not None:
                progress_hook({"phase": phase, "completed": completed, "total": total})

        emit("claim_extraction")
        claims = self.extractor.extract(response, query)
        claim_total = len(claims)
        claim_interval = max(1, claim_total // 10) if claim_total else 1
        emit("claim_verification", 0, claim_total)
        audits: list[dict[str, Any]] = []
        for completed, claim in enumerate(claims, start=1):
            decision = self.verifier.verify_claim(claim.text, context, query)
            audits.append({
                "claim": claim.to_dict(),
                "evidence": [span.to_dict() for span in decision.evidence],
                "verdict": decision.verdict,
                "verifier_cache_hit": decision.cache_hit,
                "verifier_protocol_fallback": decision.protocol_fallback,
                "verifier_fallback_reason": decision.fallback_reason,
            })
            if completed == 1 or completed == claim_total or completed % claim_interval == 0:
                emit("claim_verification", completed, claim_total)
        emit("claim_verification_done", claim_total, claim_total)
        return audits


def _without_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_cache(item) for key, item in value.items() if "cache" not in key}
    return value


def veriscore_protocol(cfg) -> dict[str, Any]:
    section = cfg.veriscore
    examples_path = config_value(section.claim_extractor, "examples_path")
    examples = load_extraction_examples(examples_path) if examples_path else DEFAULT_EXTRACTION_EXAMPLES
    return {
        "protocol": VERISCORE_PROTOCOL,
        "settings": _without_cache(section.to_dict()),
        "examples": {kind: [asdict(example) for example in items] for kind, items in examples.items()},
    }
