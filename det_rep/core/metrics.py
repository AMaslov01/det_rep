"""HalluGraph metrics, including strict and text-verified relation support.

Strict RP is intentionally retained byte-for-byte in meaning:

  RP_strict = |{e in E_a: exists e' in E_ref, align(e,e')}| / |E_a|

The support path adds endpoint grounding and an evidence-constrained relation
verdict.  It never substitutes ``RP_grounded`` for factual support:

  RP_grounded      = |{e: subject and object match V_ref}| / |E_a|
  RP_entailed_cond = |{grounded e: verifier(e)=entailed}| / |E_grounded|
  RP_support       = |{grounded e: verifier(e)=entailed}| / |E_a|
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .matching import RefGraph, normalize


@dataclass
class ScoreResult:
    # graph sizes
    Vc: int = 0
    Ec: int = 0
    Vq: int = 0
    Eq: int = 0
    Va: int = 0
    Ea: int = 0
    # alpha-independent entity and legacy strict metrics
    EG: float | None = None
    RP: float | None = None  # legacy alias for RP_strict
    RP_defined: bool = False
    RP_strict: float | None = None
    RP_strict_defined: bool = False
    # New relation-support decomposition
    RP_grounded: float | None = None
    RP_grounded_defined: bool = False
    RP_entailed_cond: float | None = None
    RP_entailed_cond_defined: bool = False
    RP_support: float | None = None
    RP_support_defined: bool = False
    support_verified: bool = False
    # legacy audit detail
    matched_entities: list[tuple[str, str, str]] = field(default_factory=list)
    ungrounded_entities: list[str] = field(default_factory=list)
    supported_relations: list[list[str]] = field(default_factory=list)
    unsupported_relations: list[list[str]] = field(default_factory=list)
    # one explanatory record for every answer edge
    relation_audits: list[dict[str, Any]] = field(default_factory=list)
    # ``support-critical`` is deliberately optional.  Omitting it from
    # serialization preserves the historical strict/support artifact bytes.
    critical: dict[str, Any] | None = None
    # flags
    unscorable: bool = False
    ref_empty: bool = False

    def rp_for_mode(self, mode: str = "strict") -> float | None:
        if mode == "strict":
            return self.RP_strict if self.RP_strict is not None else self.RP
        if mode == "support":
            return self.RP_support
        raise ValueError(f"unknown relation mode {mode!r}")

    def rp_defined_for_mode(self, mode: str = "strict") -> bool:
        if mode == "strict":
            return bool(self.RP_strict_defined or self.RP_defined)
        if mode == "support":
            return self.RP_support_defined
        raise ValueError(f"unknown relation mode {mode!r}")

    def cfi_for_mode(self, alpha: float, mode: str = "strict") -> float | None:
        """Composite fidelity for strict or verified relation support."""
        if self.unscorable or self.EG is None:
            return None
        rp = self.rp_for_mode(mode)
        if not self.rp_defined_for_mode(mode) or rp is None:
            # Edge-aware reduction is valid only when the answer graph truly
            # has no edges.  A strict-only run with Ea>0 has not measured
            # support RP and must not publish a fabricated H_support=1-EG.
            return self.EG if self.Ea == 0 else None
        return alpha * self.EG + (1.0 - alpha) * rp

    def h_for_mode(
        self, alpha: float, mode: str = "strict", impute: float | None = None
    ) -> float | None:
        c = self.cfi_for_mode(alpha, mode)
        return impute if c is None else 1.0 - c

    # Backward-compatible strict aliases used by existing code/tests.
    def cfi(self, alpha: float) -> float | None:
        return self.cfi_for_mode(alpha, "strict")

    def h(self, alpha: float, impute: float | None = None) -> float | None:
        return self.h_for_mode(alpha, "strict", impute)

    def eg_only_h(self) -> float | None:
        return None if self.EG is None else 1.0 - self.EG

    def rp_only_h(self) -> float | None:
        rp = self.rp_for_mode("strict")
        return None if rp is None else 1.0 - rp

    def support_rp_only_h(self) -> float | None:
        return None if self.RP_support is None else 1.0 - self.RP_support

    def critical_relation_rp(self, unknown_risk: float) -> float | None:
        """Relation fidelity after hard unsupported/contradicted penalties."""
        if self.Ea == 0:
            return None
        if self.critical is None:
            return None
        from .critical import claim_risk

        risks: list[float] = []
        for audit in self.relation_audits:
            verdict = audit.get("verdict")
            # Graph endpoints that fail strict grounding are candidates too:
            # they retain a hard relation risk rather than disappearing from
            # the response-level detector.
            risks.append(claim_risk(verdict, unknown_risk))
        return 1.0 - (sum(risks) / len(risks)) if risks else None

    def critical_claim_topk(self, k: int, unknown_risk: float) -> float | None:
        """Mean risk among the k worst independently verified atomic claims."""
        if self.critical is None:
            return None
        from .critical import claim_risk

        audits = list(self.critical.get("claim_audits", []))
        if not audits:
            return None
        if k <= 0:
            raise ValueError("critical top-k must be positive")
        risks = sorted(
            (claim_risk(audit.get("verdict"), unknown_risk) for audit in audits), reverse=True
        )
        return sum(risks[: min(k, len(risks))]) / min(k, len(risks))

    def critical_h(
        self, alpha: float, beta: float, top_k: int, unknown_risk: float,
        impute: float | None = None,
    ) -> float | None:
        """Hybrid graph mean + worst-claim risk for ``support-critical``.

        Empty answer graphs can still be diagnostically scored if the atomic
        claim layer has claims.  They remain marked unscorable so paired
        headline metrics can retain the strict/support denominator.
        """
        claim_h = self.critical_claim_topk(top_k, unknown_risk)
        if self.EG is None:
            return claim_h if claim_h is not None else impute
        rp = self.critical_relation_rp(unknown_risk)
        graph_h = 1.0 - (self.EG if rp is None else alpha * self.EG + (1.0 - alpha) * rp)
        if claim_h is None:
            return graph_h
        return (1.0 - beta) * graph_h + beta * claim_h

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "Vc": self.Vc, "Ec": self.Ec, "Vq": self.Vq, "Eq": self.Eq,
            "Va": self.Va, "Ea": self.Ea,
            "EG": self.EG,
            "RP": self.RP, "RP_defined": self.RP_defined,
            "RP_strict": self.RP_strict, "RP_strict_defined": self.RP_strict_defined,
            "RP_grounded": self.RP_grounded, "RP_grounded_defined": self.RP_grounded_defined,
            "RP_entailed_cond": self.RP_entailed_cond,
            "RP_entailed_cond_defined": self.RP_entailed_cond_defined,
            "RP_support": self.RP_support, "RP_support_defined": self.RP_support_defined,
            "support_verified": self.support_verified,
            "matched_entities": [list(m) for m in self.matched_entities],
            "ungrounded_entities": self.ungrounded_entities,
            "supported_relations": self.supported_relations,
            "unsupported_relations": self.unsupported_relations,
            "relation_audits": self.relation_audits,
            "unscorable": self.unscorable, "ref_empty": self.ref_empty,
        }
        if self.critical is not None:
            payload["critical"] = self.critical
        return payload

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreResult":
        # Existing scored.jsonl artifacts only contain the legacy strict fields.
        legacy_rp = d.get("RP")
        legacy_defined = bool(d.get("RP_defined", legacy_rp is not None))
        return cls(
            Vc=d.get("Vc", 0), Ec=d.get("Ec", 0), Vq=d.get("Vq", 0), Eq=d.get("Eq", 0),
            Va=d.get("Va", 0), Ea=d.get("Ea", 0), EG=d.get("EG"),
            RP=legacy_rp, RP_defined=legacy_defined,
            RP_strict=d.get("RP_strict", legacy_rp),
            RP_strict_defined=bool(d.get("RP_strict_defined", legacy_defined)),
            RP_grounded=d.get("RP_grounded"),
            RP_grounded_defined=bool(d.get("RP_grounded_defined", False)),
            RP_entailed_cond=d.get("RP_entailed_cond"),
            RP_entailed_cond_defined=bool(d.get("RP_entailed_cond_defined", False)),
            RP_support=d.get("RP_support"),
            RP_support_defined=bool(d.get("RP_support_defined", False)),
            support_verified=bool(d.get("support_verified", False)),
            matched_entities=[tuple(m) for m in d.get("matched_entities", [])],
            ungrounded_entities=list(d.get("ungrounded_entities", [])),
            supported_relations=[list(r) for r in d.get("supported_relations", [])],
            unsupported_relations=[list(r) for r in d.get("unsupported_relations", [])],
            relation_audits=list(d.get("relation_audits", [])),
            critical=d.get("critical"),
            unscorable=bool(d.get("unscorable", False)), ref_empty=bool(d.get("ref_empty", False)),
        )


def score_response(
    g_a,
    refgraph: RefGraph,
    g_c=None,
    g_q=None,
    *,
    context: str = "",
    query: str | None = None,
    verifier=None,
    verifier_matching_params: dict[str, Any] | None = None,
    answer_text: str | None = None,
    critical_pipeline=None,
    progress_hook: Callable[[dict[str, Any]], None] | None = None,
) -> ScoreResult:
    """Score one answer graph; verifier is optional so strict scoring stays LLM-free."""
    Va, Ea = len(g_a.entities), len(g_a.relations)
    res = ScoreResult(
        Va=Va, Ea=Ea,
        Vc=len(g_c.entities) if g_c is not None else 0,
        Ec=len(g_c.relations) if g_c is not None else 0,
        Vq=len(g_q.entities) if g_q is not None else 0,
        Eq=len(g_q.relations) if g_q is not None else 0,
        support_verified=verifier is not None,
    )
    res.ref_empty = len(refgraph.ent_norm) == 0

    # Entity Grounding remains independent of all relation metrics.
    if Va == 0:
        res.unscorable = True
        res.EG = None
    else:
        grounded_entities = 0
        for entity in sorted(g_a.entities):
            match = refgraph.match_entity(entity)
            if match.matched:
                grounded_entities += 1
                res.matched_entities.append((_norm(entity), match.ref or "", match.method or ""))
            else:
                res.ungrounded_entities.append(_norm(entity))
        res.EG = grounded_entities / Va

    if Ea == 0:
        # Every relation metric is undefined when the response has no relation edges.
        if critical_pipeline is not None:
            if answer_text is None:
                raise ValueError("support-critical scoring requires answer_text")
            res.critical = {
                "protocol": "support-critical-v1",
                "claim_audits": critical_pipeline.assess(
                    answer_text, context, query, progress_hook=progress_hook
                ),
            }
        return res

    strict_supported = 0
    grounded_edges = 0
    entailed_edges = 0
    edges = sorted(g_a.relations)
    edge_total = len(edges)
    edge_interval = max(1, edge_total // 10) if edge_total else 1
    if progress_hook is not None and verifier is not None:
        progress_hook({"phase": "relation_audit", "completed": 0, "total": edge_total})
    for edge_index, edge in enumerate(edges, start=1):
        subject, predicate, obj = (_norm(x) for x in edge)
        subj_match = refgraph.match_entity(edge[0])
        obj_match = refgraph.match_entity(edge[2])
        alignment = refgraph.align_relation(edge)
        triple = [subject, predicate, obj]
        strict_ref = list(alignment.ref) if alignment.ref is not None else None

        if alignment.matched:
            strict_supported += 1
            res.supported_relations.append(triple)
        else:
            res.unsupported_relations.append(triple)

        audit: dict[str, Any] = {
            "answer_edge": triple,
            "canonical_edge": None,
            "subject": {
                "grounded": subj_match.matched,
                "canonical": subj_match.ref,
                "method": subj_match.method,
            },
            "object": {
                "grounded": obj_match.matched,
                "canonical": obj_match.ref,
                "method": obj_match.method,
            },
            "strict_alignment": {
                "matched": alignment.matched,
                "reference_edge": strict_ref,
                "method": alignment.method,
            },
            "evidence": [],
            "verdict": None,
            "verifier_cache_hit": None,
        }
        if critical_pipeline is not None:
            audit["candidate_sources"] = [
                "strict_unmatched_edge" if not alignment.matched else "strict_matched_edge"
            ]

        if not subj_match.matched and not obj_match.matched:
            audit["status"] = "ungrounded_both"
        elif not subj_match.matched:
            audit["status"] = "ungrounded_subject"
        elif not obj_match.matched:
            audit["status"] = "ungrounded_object"
        else:
            grounded_edges += 1
            canonical = (subj_match.ref or subject, predicate, obj_match.ref or obj)
            audit["canonical_edge"] = list(canonical)
            if verifier is None:
                # A strict-only run deliberately does not claim textual entailment.
                audit["status"] = "aligned" if alignment.matched else "grounded_unverified"
            else:
                decision = verifier.verify(
                    canonical, context, query, matching_params=verifier_matching_params
                )
                audit["evidence"] = [span.to_dict() for span in decision.evidence]
                audit["verdict"] = decision.verdict
                audit["verifier_cache_hit"] = decision.cache_hit
                audit["verifier_protocol_fallback"] = bool(
                    getattr(decision, "protocol_fallback", False)
                )
                audit["verifier_fallback_reason"] = getattr(decision, "fallback_reason", None)
                if decision.verdict == "entailed":
                    entailed_edges += 1
                    audit["status"] = "aligned" if alignment.matched else "entailed_from_text"
                elif decision.verdict == "contradicted":
                    audit["status"] = "contradicted"
                elif decision.verdict == "unsupported":
                    audit["status"] = "unsupported"
                else:
                    audit["status"] = "grounded_unknown"
        res.relation_audits.append(audit)
        if (
            progress_hook is not None
            and verifier is not None
            and (edge_index == 1 or edge_index == edge_total or edge_index % edge_interval == 0)
        ):
            progress_hook(
                {"phase": "relation_audit", "completed": edge_index, "total": edge_total}
            )

    # Strict fields and legacy aliases always preserve the historical formula.
    res.RP_strict = strict_supported / Ea
    res.RP_strict_defined = True
    res.RP = res.RP_strict
    res.RP_defined = True
    res.RP_grounded = grounded_edges / Ea
    res.RP_grounded_defined = True

    if verifier is not None:
        res.RP_support = entailed_edges / Ea
        res.RP_support_defined = True
        if grounded_edges:
            res.RP_entailed_cond = entailed_edges / grounded_edges
            res.RP_entailed_cond_defined = True
    if critical_pipeline is not None:
        if answer_text is None:
            raise ValueError("support-critical scoring requires answer_text")
        res.critical = {
            "protocol": "support-critical-v1",
            "claim_audits": critical_pipeline.assess(
                answer_text, context, query, progress_hook=progress_hook
            ),
        }
    return res


def _norm(value: str) -> str:
    return normalize(value)


def cfi(eg: float, rp: float | None, alpha: float) -> float:
    """Pure CFI given values (rp=None => edge-aware reduction to EG)."""
    return eg if rp is None else alpha * eg + (1.0 - alpha) * rp


def hallucination(cfi_value: float) -> float:
    return 1.0 - cfi_value
