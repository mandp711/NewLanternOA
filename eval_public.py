#!/usr/bin/env python3
"""
Local evaluation against `relevant_priors_public.json` (truth + cases).

Usage:
  python eval_public.py --json /path/to/relevant_priors_public.json
  python eval_public.py --json ./relevant_priors_public.json --max-cases 5

Env:
  RELEVANT_PRIORS_PUBLIC  default path if --json is omitted
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pydantic import TypeAdapter

from main import CaseIn, PredictRequest, _build_predictions


def _default_json_path() -> Path:
    env = os.environ.get("RELEVANT_PRIORS_PUBLIC", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "relevant_priors_public.json"


def _truth_key(case_id, study_id) -> tuple[str, str]:
    return (str(case_id), str(study_id))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score predictions vs public gold truth")
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Path to relevant_priors_public.json (or set RELEVANT_PRIORS_PUBLIC)",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Only use the first N cases (for a quick smoke test)",
    )
    args = ap.parse_args()
    path = args.json or _default_json_path()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}\n  Download the public eval JSON and pass --json or set RELEVANT_PRIORS_PUBLIC")

    t0 = time.perf_counter()
    with path.open() as f:
        data = json.load(f)
    load_s = time.perf_counter() - t0
    if data.get("challenge_id") != "relevant-priors-v1":
        raise SystemExit("unexpected challenge_id in file")
    if args.max_cases is None:
        if data.get("truth_count") != len(data["truth"]):
            raise SystemExit("truth_count does not match len(truth)")
        if data.get("case_count") is not None and data["case_count"] != len(data["cases"]):
            raise SystemExit("case_count does not match len(cases)")

    truth_list = data["truth"]
    cases_raw = data["cases"]
    if args.max_cases is not None:
        cases_raw = cases_raw[: args.max_cases]
        # Restrict truth to the priors that still exist in the truncated case set
        allowed = set()
        for c in cases_raw:
            cid = c["case_id"]
            for p in c["prior_studies"]:
                allowed.add(_truth_key(cid, p["study_id"]))
        truth_list = [t for t in truth_list if _truth_key(t["case_id"], t["study_id"]) in allowed]

    truth_map: dict[tuple[str, str], bool] = {}
    for t in truth_list:
        k = _truth_key(t["case_id"], t["study_id"])
        truth_map[k] = t["is_relevant_to_current"]

    cases = TypeAdapter(list[CaseIn]).validate_python(cases_raw)
    body = PredictRequest(
        challenge_id=data["challenge_id"],
        schema_version=data.get("schema_version", 1),
        generated_at=data.get("generated_at", "2026-01-01T00:00:00.000Z"),
        cases=cases,
    )
    n_priors = sum(len(c.prior_studies) for c in body.cases)
    if len(truth_list) != len(truth_map):
        raise SystemExit("duplicate (case_id, study_id) entries in truth slice")

    if n_priors != len(truth_list):
        raise SystemExit(
            f"prior count mismatch: cases have {n_priors} priors, truth has {len(truth_list)}"
        )

    t1 = time.perf_counter()
    predictions = _build_predictions(body)
    infer_s = time.perf_counter() - t1

    if len(predictions) != n_priors:
        raise SystemExit(f"not one prediction per prior: got {len(predictions)} expected {n_priors}")

    correct = 0
    missing = 0
    for p in predictions:
        k = (p.case_id, p.study_id)
        gold = truth_map.get(k)
        if gold is None:
            missing += 1
            continue
        if p.predicted_is_relevant is gold:
            correct += 1
    acc = correct / n_priors if n_priors else 0.0
    total_s = time.perf_counter() - t0
    print(f"file: {path}")
    print(f"cases: {len(body.cases)}  priors: {n_priors}  (json load: {load_s:.2f}s, predict: {infer_s:.2f}s, total: {total_s:.2f}s)")
    print(f"accuracy: {acc:.4f}  ({correct}/{n_priors})")
    if missing:
        print(f"ERROR: {missing} predictions had no gold label (key mismatch)")


if __name__ == "__main__":
    main()
