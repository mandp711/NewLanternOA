# Experiments

## Baseline

Initial approach: **deterministic string heuristics only** (no external models, no per-prior LLM calls) so the service stays within evaluation time limits and stays cheap to host on Render.

- **Stack:** FastAPI + Uvicorn, `POST /predict` on `main.py`.
- **Core logic:** `relevance.py` — normalize text, infer a coarse modality label, tag rough anatomy (spine segments, chest, cardiac/breast/GI, etc.), combine sequence similarity (`difflib`) and token Jaccard overlap, then modality-specific thresholds.
- MRI and CT are kept **distinct modality buckets** (merging them as “XR-style” falsely treated many neuro vs cross-sectional pairs as comparable).

## What worked

- **Contract:** One prediction per prior; shape matches the expected response schema under local checks and hosted smoke tests.
- **Heuristic layers:** Anatomy overlap (“synergy”) + spine-level disagreement penalty reduced wrong “same modality, unrelated body region” positives compared with raw string similarity alone.
- **Clinical bridges (lightweight rules):** e.g., echo wording vs coronary/calcium/perfusion phrasing tagged separately so echocardiography and coronary pathway studies can associate when narratives align, without collapsing all cross-modality pairs.
- **Local validation:** scripting against the downloadable public bundle (`truth` vs predicted) for aggregate accuracy/regression tracking while iterating on thresholds.


## What failed or under-delivered

- **No supervised training:** Heuristics do not optimize the private objective; remaining errors cluster where text alone is ambiguous or where gold relevance follows institutional nuance beyond rule lists.
- **Modality/description noise:** Leading-token modality detection is brittle for atypical prefixes; abbreviations and vendor-specific strings still slip through inconsistently.

## How I would improve it

1. **Batch LLM judging** (single prompt listing *all* priors for a case alongside the current study) behind an API key — required by the organizer hints when going beyond rules; reuse caching by `(normalized current, normalized prior)` to survive retries.

2. **Embedding similarity** between normalized descriptions (`sentence-transformers` or hosted embeddings) calibrated on the public labeled JSON, fused with modality/anatomy masks.

3. **Train a small classifier** on `(current text, prior text)` sparse/dense features plus structured tags; validate with cross-validation on `truth`.

4. **Evaluation discipline:** Held-out folds from the **public labeled file**, track precision/reclevance by modality and body region, and tune thresholds accordingly before submitting.
