"""
New Lantern: relevant-priors-v1 — POST /predict
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from relevance import is_prior_relevant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("predict")

app = FastAPI(title="relevant-priors-v1", version="1.0.0")


class StudyIn(BaseModel):
    study_id: str
    study_description: str
    study_date: str


class CaseIn(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: StudyIn
    prior_studies: list[StudyIn]


_EXAMPLE_REQUEST = {
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "generated_at": "2026-04-16T12:00:00.000Z",
    "cases": [
        {
            "case_id": "1001016",
            "patient_id": "606707",
            "patient_name": "Andrews, Micheal",
            "current_study": {
                "study_id": "3100042",
                "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                "study_date": "2026-03-08",
            },
            "prior_studies": [
                {
                    "study_id": "2453245",
                    "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
                    "study_date": "2020-03-08",
                },
                {
                    "study_id": "992654",
                    "study_description": "CT HEAD WITHOUT CNTRST",
                    "study_date": "2021-03-08",
                },
            ],
        }
    ],
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": _EXAMPLE_REQUEST},
    )

    challenge_id: str = Field(..., description="Challenge name, e.g. relevant-priors-v1")
    schema_version: int
    generated_at: str = Field(
        default="2026-01-01T00:00:00.000Z",
        description="ISO timestamp; public eval file may omit this key",
    )
    cases: list[CaseIn] = Field(
        ...,
        description="Patient/case objects with current_study and prior_studies",
    )


class PredictionOut(BaseModel):
    case_id: str
    study_id: str
    predicted_is_relevant: bool


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]


@app.get("/")
def root() -> dict[str, str | list[str]]:
    return {
        "service": "relevant-priors-v1",
        "endpoints": ["POST /predict", "GET /health", "GET /docs"],
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _build_predictions(body: PredictRequest) -> list[PredictionOut]:
    out: list[PredictionOut] = []
    for case in body.cases:
        current_desc = case.current_study.study_description
        # One batched pass over all priors (single process; no N LLM round-trips)
        for prior in case.prior_studies:
            rel = is_prior_relevant(current_desc, prior.study_description)
            out.append(
                PredictionOut(
                    case_id=case.case_id,
                    study_id=prior.study_id,
                    predicted_is_relevant=rel,
                )
            )
    return out


@app.get("/predict", include_in_schema=False)
def predict_must_post() -> dict[str, str]:
    """Explain why the score URL looks 'broken' when opened in a browser (GET vs POST)."""
    return {
        "message": (
            "/predict expects HTTP POST with a JSON body (challenge request schema). "
            "A normal browser tab uses GET—as in /docs—not POST."
        ),
        "docs": "/docs",
        "hint": "New Lantern POSTs JSON to this URL; curl: curl -X POST $HOST/predict -H 'Content-Type: application/json' -d '{...}'.",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest) -> JSONResponse:
    t0 = time.perf_counter()
    n_cases = len(body.cases)
    n_priors = sum(len(c.prior_studies) for c in body.cases)
    log.info(
        "request challenge_id=%s generated_at=%s cases=%d priors=%d",
        body.challenge_id,
        body.generated_at,
        n_cases,
        n_priors,
    )
    try:
        predictions = _build_predictions(body)
    except Exception as e:
        log.exception("predict failed: %s", e)
        raise
    elapsed = time.perf_counter() - t0
    log.info("response predictions=%d elapsed_ms=%.1f", len(predictions), elapsed * 1000)
    if len(predictions) != n_priors:
        log.warning("prior count mismatch: expected %d got %d", n_priors, len(predictions))
    return JSONResponse(
        content=PredictResponse(predictions=predictions).model_dump(),
        status_code=200,
    )
