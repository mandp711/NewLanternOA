"""Heuristic prior relevance: batched, cache-friendly, no per-prior LLM calls."""

from __future__ import annotations

import re
import logging
from difflib import SequenceMatcher
from functools import lru_cache

log = logging.getLogger(__name__)

# Common radiology modalities at start of study description
_MODALITY_RE = re.compile(
    r"^(MRI|CT|US|ULTRASOUND|PET|XR|X-RAY|XRAY|DEXA|NM|MG|MAMMOGRAM|FLUORO|ANGIO|VUS)\b",
    re.I,
)


def _norm(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return " ".join(s.split())


@lru_cache(maxsize=50_000)
def _modality(text: str) -> str | None:
    t = text.strip()
    m = _MODALITY_RE.match(t)
    if m:
        return m.group(1).upper()
    return None


@lru_cache(maxsize=200_000)
def _description_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@lru_cache(maxsize=200_000)
def _token_jaccard(a: str, b: str) -> float:
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_prior_relevant(current_description: str, prior_description: str) -> bool:
    """
    Baseline: same modality + high description overlap => relevant;
    different unambiguous modalities with low text overlap => not relevant.
    """
    c = _norm(current_description)
    p = _norm(prior_description)
    if c == p:
        return True

    mc, mp = _modality(current_description or ""), _modality(prior_description or "")
    seq = _description_similarity(c, p)
    jac = _token_jaccard(c, p)

    # Strong text match despite minor wording (e.g. CONTRAST vs CNTRST)
    if seq >= 0.82 or (seq >= 0.68 and jac >= 0.45):
        return True

    if mc and mp and mc != mp:
        # Cross-modality priors are usually not "the same" follow-up unless text nearly matches
        if seq >= 0.72 and jac >= 0.2:
            return True
        return False

    if mc and mp and mc == mp:
        # Same modality: lean on overlap
        if jac >= 0.25 or seq >= 0.55:
            return True
        return False

    # Ambiguous modality: use overlap only
    return jac >= 0.3 or seq >= 0.5
