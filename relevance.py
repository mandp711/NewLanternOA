"""Stronger heuristic prior relevance: modality families, anatomy tags, fused score."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

STOP_TOKENS = frozenset(
    """THE A AN AND OR OF FOR ROUTINE STUDY STANDARD LIMITED COMPLETE SINGLE BILATERAL
    UNLISTED FOLLOW UP FOLLOWUP FOLLOW-UP SECONDARY PORTABLE STAT ACUTE QUALITY
    WITHOUT WITH CONTRAST W WO NON UNLISTED SAME DATE PORTABLE SECONDARY SAME""".split()
)

_SYN_REPL = (
    ("CNTRST", "CONTRAST"),
    (" WO W ", " WO WITH "),
)


@lru_cache(maxsize=30_000)
def _clinical_norm(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = " ".join(s.split())
    for a, b in _SYN_REPL:
        s = s.replace(a, b)
    return s


_LEADING_MOD_RE = re.compile(
    r"^(MRI|MRA|MRV|MRS|PET|NM|CT|XR|US|MAM\b|ECHO|DXA)\b",
    re.I,
)


def _leading_mod_class(text: str) -> str | None:
    """First-token modality for radiology ordering (approximate)."""
    t = text.strip().upper().replace(":", " ")
    sp = re.split(r"[\s/]+", t, maxsplit=1)
    w0 = sp[0] if sp else ""
    if not w0:
        return None
    if w0 == "ECHO" or w0.startswith("ECHO"):
        return "ECHO"
    if w0.startswith("MAM"):
        return "MAMMO"
    if w0 in ("MRI", "MR", "MRA", "MRV", "MRS") or w0.startswith("MRI"):
        return "MRI"
    if w0.startswith("CT"):
        return "CT"
    if w0.startswith("PET"):
        return "PET"
    if w0 == "NM":
        return "NM"
    if w0 in ("US", "ULTRASOUND"):
        return "US"
    if w0 in ("XR", "XG"):
        return "XR"
    if w0 in ("DXA", "DEXA"):
        return "DEXA"
    m = _LEADING_MOD_RE.match(t)
    if not m:
        return None
    g = m.group(1).upper()
    if g in ("MRA", "MRV", "MRS") or g.startswith("MRI"):
        return "MRI"
    if g.startswith("CT"):
        return "CT"
    if g.startswith("MAM"):
        return "MAMMO"
    return g


def _mod_bucket(mod: str | None) -> str | None:
    """Coarse bucket: MRI and CT stay distinct (do not merge)."""
    if not mod:
        return None
    if mod == "MRI":
        return "MRI"
    if mod == "CT":
        return "CT"
    if mod in ("NM", "PET"):
        return "NM_PET"
    if mod == "ECHO":
        return "ECHO"
    if mod == "MAMMO":
        return "MAMMO"
    if mod in ("US", "XR", "DEXA"):
        return mod
    return mod


_CACHE = 220_000


@lru_cache(maxsize=_CACHE)
def _seq(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


@lru_cache(maxsize=_CACHE)
def _jac(a: str, b: str) -> float:

    def tok(x: str) -> set[str]:
        return {t for t in x.split() if len(t) > 1 and t not in STOP_TOKENS}

    sa, sb = tok(a), tok(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@lru_cache(maxsize=_CACHE)
def _anatomy_profile(blob: str) -> tuple[frozenset[str], frozenset[str]]:
    """
    Returns (coarse_region_tags, spine_segment_tags segments in {C,T,L,S}).
    Spine segments approximate — used to separate cervical vs lumbar etc.
    """
    s = " " + blob + " "
    tags: set[str] = set()
    spine_seg: set[str] = set()

    if re.search(r"(\bbrain\b|cerebral|intracran|stroke|cerebral|subar)", blob, re.I):
        tags.add("neuro")
    if re.search(r"(\bc head\b|^ct head|^ct\b head| head wo | head\b w | head wo)", blob, re.I):
        tags.add("neuro_head")
    if any(x in s for x in (" BRAIN ", " STROKE ")):
        tags.add("neuro")

    if " C-SPINE " in s or " C SPINE " in s or re.search(r"\bCERVICAL SPINE\b", blob):
        spine_seg.add("C")
    if " T-SPINE " in s or " T SPINE " in s or " THORACIC SPINE " in s:
        spine_seg.add("T")
    if (
        " L-SPINE " in s
        or " L SPINE " in s
        or " LUMBOSACRAL " in s
        or (" LUMBAR " in s and " SPINE " in s)
    ):
        spine_seg.add("L")
    if " SACRAL " in s or " SACRUM " in s or " SI JOINT " in s:
        spine_seg.add("S")

    any_spinal_word = (" SPINE " in s) or spine_seg or re.search(r"\bDISC\b", blob)

    # Thoracic MRI without explicit "thoracic spine" often lists THOR SPINE or segment
    if re.search(r"\bTHORACIC\b", blob) and any_spinal_word:
        spine_seg.add("T")
    if re.search(r"\bCERVICAL\b", blob) and any_spinal_word:
        spine_seg.add("C")
    if any_spinal_word:
        tags.add("spine_any")

    if (
        (" CHEST " in s or " CHST " in s or " CHST." in blob.upper())
        or re.search(r"\bthorax\b", blob, re.I)
        or re.search(r"\bCT\s+CHEST\b|\bMRI\s+CHEST\b", blob, re.I)
    ):
        tags.add("chest_thoracic")
    if re.search(r"(pulmonary|\blung\b|pleur|pneumo)", blob, re.I):
        tags.add("chest_parench")

    if (
        (" CORONARY " in s)
        or (" CARDIAC " in s)
        or re.search(r"\bCAC\b|chest calcium|cor calcif", blob, re.I)
        or re.search(r"angio.?coron|\bcoronary.?ang\b", blob, re.I)
    ):
        tags.add("cardiac_cor")

    # NM myocard perfusion vs CT coronary are often compared for same vascular question
    if re.search(r"(perfusion|spect|myocard|cardiolite|rubidium|thalli)", blob, re.I):
        tags.add("cardiac_perf_nm")
    if re.search(
        r"mamm|breast|MAM|TOMO|tomosynthesis|papill|axilla| nipple\b", blob, re.I
    ):
        tags.add("breast")

    if re.search(
        r"\b(abd\b|abdomen|hepatic|hep\b|splen\b|bilary|bilir|kidney\b|renal|pelvic\b|GU\b|colon|append|pancreas|intestinal)",
        blob,
        re.I,
    ):
        tags.add("abd_pelvic")

    if re.search(r"\bknee\b|\bshoulder\b|ankle\b|MSK\b|\bHIP\b\b", blob, re.I):
        tags.add("msk")

    if re.search(r"\bneck\b\b", blob, re.I) and " SPINE " in s:
        spine_seg.add("C")

    return frozenset(tags), frozenset(spine_seg)


def _spine_conflicts(sg1: frozenset[str], sg2: frozenset[str]) -> float:
    """Return penalty magnitude in ~[0,1]; 0=no conflict."""
    if not sg1 or not sg2:
        return 0.0
    if sg1.intersection(sg2):
        return 0.0
    # Both specify different spine levels — weaker association
    if len(sg1) >= 1 and len(sg2) >= 1 and sg1.isdisjoint(sg2):
        return 0.35
    return 0.0


def _region_synergy(tags1: frozenset[str], tags2: frozenset[str]) -> float:
    if not tags1 or not tags2:
        return 0.0
    inter = len(tags1.intersection(tags2))
    uni = len(tags1.union(tags2))
    if uni == 0:
        return 0.0
    return min(1.0, (2.5 * inter) / max(uni, inter + 2))


def is_prior_relevant(current_description: str, prior_description: str) -> bool:
    cc = _clinical_norm(current_description)
    pp = _clinical_norm(prior_description)
    if cc == pp:
        return True

    seq = _seq(cc, pp)
    jac = _jac(cc, pp)
    mc = _leading_mod_class(current_description.strip())
    mp = _leading_mod_class(prior_description.strip())
    fc = _mod_bucket(mc)
    fp = _mod_bucket(mp)

    tags_c, sc = _anatomy_profile(cc)
    tags_p, sp = _anatomy_profile(pp)

    spine_pen = _spine_conflicts(sc, sp)

    synergy = _region_synergy(tags_c, tags_p)

    score = (
        0.46 * seq
        + 0.42 * jac
        + synergy * (0.16 if synergy > 0.15 else 0.06)
        - spine_pen
    )

    # Near-duplicate wording
    if seq >= 0.84 or (seq >= 0.72 and jac >= 0.38):
        return True

    same_bucket = fc is not None and fc == fp
    cross_mr_ct = (mc == "MRI" and mp == "CT") or (mc == "CT" and mp == "MRI")

    # Same modality family (both MRI / both PET, etc.)
    if same_bucket:
        score += 0.12
        if synergy >= 0.12 or jac >= 0.18:
            threshold = 0.38
            return score >= threshold
        threshold = 0.46
        return score >= threshold

    # PET/NM overlapping anatomy
    if fc == "NM_PET" and fp == "NM_PET" and synergy >= 0.2:
        return score >= 0.42

    # MRI ←→ CT crossover (distinct imaging; allow only with shared story)
    if cross_mr_ct:
        if tags_c.intersection(tags_p) & {"breast"} and len(tags_c & tags_p) >= 2:
            return score >= 0.52
        if {"cardiac_cor", "cardiac_perf_nm"}.issubset(tags_c | tags_p) and synergy >= 0.07:
            if seq >= 0.62 and jac >= 0.08:
                return True
            if synergy >= 0.15:
                return score >= 0.55
        if seq >= 0.76 or (seq >= 0.62 and synergy >= 0.18):
            return True

    diff_bucket = fc and fp and fc != fp
    if diff_bucket:
        # Cross-modality: high bar unless same clinical story
        penalty = -0.12
        merged = synergy + jac * 0.6 + seq * 0.25 + penalty + (0.1 if synergy > 0.18 else 0)
        merged += 0.08 if synergy >= 0.15 else 0
        merged -= spine_pen if spine_pen > 0 else 0

        neuro_overlap = tags_c.intersection(tags_p).intersection(
            frozenset({"neuro_head", "neuro"})
        )
        if neuro_overlap:
            merged += 0.05

        if merged >= 0.58:
            return True
        # MR brain vs CT head — rarely both true in gold; keep harsh
        if {"MRI", "MR"}.intersection({mc or "", mp or ""}) and (
            seq < 0.68 and synergy < 0.12 and jac < 0.08
        ):
            return merged >= 0.62
        threshold = 0.56
        if merged >= threshold and (seq >= 0.71 or synergy >= 0.12):
            return True
        if merged >= 0.64:
            return True
        # Default different modalities not relevant
        if merged < threshold and seq < 0.71:
            return False
        return merged >= threshold

    # Fallback: unknown modality
    fallback = score + synergy * 0.1 + jac * 0.05 - spine_pen
    return fallback >= 0.44


def _checks() -> None:
    assert (
        is_prior_relevant(
            "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
            "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        )
        is True
    )
    assert (
        is_prior_relevant(
            "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
            "CT HEAD WITHOUT CNTRST",
        )
        is False
    )


_checks()
