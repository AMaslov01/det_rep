"""Entity/relation matching (HalluGraph's match()/align() adapted to KGGen's untyped strings).

Entity match(v, w) is TRUE if ANY of:
  1. exact normalized string equality
  2. (if allow_substring_match) token-boundary substring in either direction, where the
     shorter side has >= min_substring_chars and is not a stopword
  3. cosine(S-BERT(v), S-BERT(w)) >= entity_sim_threshold (tau_e)

Relation align(e=(s,r,o), e'=(s',r',o')) is TRUE iff:
  - match(s,s') AND match(o,o')   (direction-sensitive; inverse-edge ablation optional), AND
  - relation labels compatible: exact normalized equality OR cosine(r,r') >= relation_sim_threshold

Embeddings go through an injectable ``Embedder`` so tests / offline runs avoid torch.
"""
from __future__ import annotations

import hashlib
import re
import string
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

_WS = re.compile(r"\s+")
_ARTICLES = {"a", "an", "the"}
_PUNCT = string.punctuation


def normalize(s: str) -> str:
    """lowercase, strip surrounding punctuation, drop a single leading article, collapse ws."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = _WS.sub(" ", s)
    # strip surrounding punctuation/quotes/whitespace
    s = s.strip(_PUNCT + " \t\n\r\"'`")
    s = _WS.sub(" ", s).strip()
    toks = s.split(" ")
    if len(toks) > 1 and toks[0] in _ARTICLES:
        toks = toks[1:]
    return " ".join(toks).strip()


def _token_boundary_substring(short: str, long: str) -> bool:
    """True if `short` occurs in `long` at word boundaries (both already normalized)."""
    if not short:
        return False
    return re.search(r"(?<!\w)" + re.escape(short) + r"(?!\w)", long) is not None


# --------------------------------------------------------------------------------------
# Embedders (injectable)
# --------------------------------------------------------------------------------------
class Embedder:
    """Interface: encode(list[str]) -> L2-normalized float32 array of shape (n, d)."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SBERTEmbedder(Embedder):
    """Lazy sentence-transformers backend with an in-memory cache."""

    def __init__(
        self,
        model_name: str,
        *,
        model_revision: str | None = None,
        model_path: str | None = None,
        device: str = "cpu",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.model_path = model_path
        self.device = device
        self.local_files_only = bool(local_files_only)
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            target = self.model_path or self.model_name
            kwargs = {
                "device": self.device,
                "local_files_only": self.local_files_only,
            }
            # A local snapshot path is already revision-specific.  Passing a
            # Hub revision alongside it is at best ignored and at worst causes
            # older sentence-transformers releases to consult the network.
            if self.model_revision and not self.model_path:
                kwargs["revision"] = self.model_revision
            self._model = SentenceTransformer(target, **kwargs)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        missing = [t for t in texts if t not in self._cache]
        if missing:
            model = self._ensure()
            vecs = model.encode(missing, convert_to_numpy=True, normalize_embeddings=True)
            for t, v in zip(missing, vecs):
                self._cache[t] = v.astype(np.float32)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack([self._cache[t] for t in texts])


class DictEmbedder(Embedder):
    """Deterministic offline embedder for tests: exact vectors for known strings,
    a fixed pseudo-random unit vector (seeded by the string) for everything else."""

    def __init__(self, mapping: dict[str, Sequence[float]] | None = None, dim: int = 16):
        self.dim = dim
        self._map: dict[str, np.ndarray] = {}
        for k, v in (mapping or {}).items():
            arr = np.asarray(v, dtype=np.float32)
            n = np.linalg.norm(arr)
            self._map[k] = arr / n if n else arr

    def _vec(self, t: str) -> np.ndarray:
        if t in self._map:
            return self._map[t]
        # Do not use Python's process-randomised ``hash`` here: fake/offline
        # runs must reproduce vectors even when PYTHONHASHSEED was not exported.
        digest = hashlib.sha256(f"emb\x00{t}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        return v

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])


# --------------------------------------------------------------------------------------
# Match results
# --------------------------------------------------------------------------------------
@dataclass
class EntityMatch:
    matched: bool
    ref: str | None = None
    method: str | None = None  # exact | substring | embedding


@dataclass
class RelationAlign:
    matched: bool
    ref: tuple[str, str, str] | None = None
    method: str | None = None  # e.g. "exact+forward", "embedding+inverse"


# --------------------------------------------------------------------------------------
# RefGraph: precomputes everything needed to match a response graph against G_ref
# --------------------------------------------------------------------------------------
class RefGraph:
    def __init__(
        self,
        entities: Iterable[str],
        relations: Iterable[tuple[str, str, str]],
        cfg_matching,
        embedder: Embedder | None,
    ):
        self.embedder = embedder
        self.tau_e = float(cfg_matching.entity_sim_threshold)
        self.tau_r = float(cfg_matching.relation_sim_threshold)
        self.allow_substring = bool(cfg_matching.allow_substring_match)
        self.direction_sensitive = bool(cfg_matching.direction_sensitive_edges)
        self.inverse_edge = bool(getattr(cfg_matching, "inverse_edge_match", False))
        self.min_sub = int(getattr(cfg_matching, "min_substring_chars", 2))
        self.stopwords = set(getattr(cfg_matching, "stopwords", []) or [])

        # normalized, de-duplicated reference entities (keep first surface form for audit)
        self.ent_norm: list[str] = []
        seen = set()
        for e in entities:
            n = normalize(e)
            if n and n not in seen:
                seen.add(n)
                self.ent_norm.append(n)
        self.ent_index = {n: i for i, n in enumerate(self.ent_norm)}

        # normalized reference relations
        self.rel_norm: list[tuple[str, str, str]] = []
        rseen = set()
        for (s, r, o) in relations:
            t = (normalize(s), normalize(r), normalize(o))
            if all(t) and t not in rseen:
                rseen.add(t)
                self.rel_norm.append(t)

        # precompute embeddings (once) if an embedder is available
        self._ent_emb: np.ndarray | None = None
        self._rel_pred_emb: dict[str, np.ndarray] = {}
        if self.embedder is not None and self.ent_norm:
            self._ent_emb = self.embedder.encode(self.ent_norm)
        if self.embedder is not None and self.rel_norm:
            preds = sorted({t[1] for t in self.rel_norm})
            embs = self.embedder.encode(preds)
            self._rel_pred_emb = {p: embs[i] for i, p in enumerate(preds)}

        self._match_cache: dict[str, EntityMatch] = {}

    # -- entity matching ------------------------------------------------------
    def match_entity(self, v: str) -> EntityMatch:
        vn = normalize(v)
        if not vn:
            return EntityMatch(False)
        if vn in self._match_cache:
            return self._match_cache[vn]
        res = self._match_entity_norm(vn)
        self._match_cache[vn] = res
        return res

    def _match_entity_norm(self, vn: str) -> EntityMatch:
        # 1) exact
        if vn in self.ent_index:
            return EntityMatch(True, vn, "exact")
        # 2) token-boundary substring (guarded)
        if self.allow_substring:
            for wn in self.ent_norm:
                shorter, longer = (vn, wn) if len(vn) <= len(wn) else (wn, vn)
                if len(shorter) < self.min_sub or shorter in self.stopwords:
                    continue
                if _token_boundary_substring(shorter, longer):
                    return EntityMatch(True, wn, "substring")
        # 3) embedding
        if self.embedder is not None and self._ent_emb is not None and len(self.ent_norm):
            v_emb = self.embedder.encode([vn])[0]
            sims = self._ent_emb @ v_emb  # rows are L2-normalized -> dot = cosine
            j = int(np.argmax(sims))
            if float(sims[j]) >= self.tau_e:
                return EntityMatch(True, self.ent_norm[j], "embedding")
        return EntityMatch(False)

    # -- relation label compatibility ----------------------------------------
    def _rel_compatible(self, r_a: str, r_ref: str) -> bool:
        if r_a == r_ref:
            return True
        if self.embedder is None:
            return False
        # ref predicate embedding is precomputed; encode the response predicate on demand
        ref_emb = self._rel_pred_emb.get(r_ref)
        if ref_emb is None:
            ref_emb = self.embedder.encode([r_ref])[0]
        a_emb = self.embedder.encode([r_a])[0]
        return float(a_emb @ ref_emb) >= self.tau_r

    # -- relation alignment ---------------------------------------------------
    def align_relation(self, e: tuple[str, str, str]) -> RelationAlign:
        s, r, o = normalize(e[0]), normalize(e[1]), normalize(e[2])
        if not (s and o):
            return RelationAlign(False)
        ms = self.match_entity(s)
        mo = self.match_entity(o)
        forward_ok = ms.matched and mo.matched
        inverse_ok = False
        if self.inverse_edge or not self.direction_sensitive:
            # accept swapped orientation: match(s, o') AND match(o, s')
            inverse_ok = ms.matched and mo.matched  # endpoints matched *somewhere* in ref
        for (s2, r2, o2) in self.rel_norm:
            if not self._rel_compatible(r, r2):
                continue
            if forward_ok and self._pair_matches(s, s2) and self._pair_matches(o, o2):
                return RelationAlign(True, (s2, r2, o2), "forward")
            if inverse_ok and self._pair_matches(s, o2) and self._pair_matches(o, s2):
                return RelationAlign(True, (s2, r2, o2), "inverse")
        return RelationAlign(False)

    def _pair_matches(self, v_norm: str, w_norm: str) -> bool:
        """match() between a response endpoint and a specific ref endpoint."""
        if v_norm == w_norm:
            return True
        if self.allow_substring:
            shorter, longer = (v_norm, w_norm) if len(v_norm) <= len(w_norm) else (w_norm, v_norm)
            if len(shorter) >= self.min_sub and shorter not in self.stopwords:
                if _token_boundary_substring(shorter, longer):
                    return True
        if self.embedder is not None:
            a = self.embedder.encode([v_norm])[0]
            b = self.embedder.encode([w_norm])[0]
            if float(a @ b) >= self.tau_e:
                return True
        return False
