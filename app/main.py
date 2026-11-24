
from functools import lru_cache

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.api.schemas import AnalysisResponse, AnalyzeTextRequest
from app.core.parser import MAX_FILE_SIZE, parse_cv
from app.ml.hybrid_matcher import HybridMatcher
from app.services.analysis import AnalysisService


app = FastAPI(
    title="CVision API",
    version="0.2.0",
    description="Reusable CV analysis and ML-powered skill matching service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)

service = AnalysisService()


# --------------------------------------------------
# ML request and response schemas
# --------------------------------------------------

class MLMatchRequest(BaseModel):
    requirement: str = Field(min_length=1)
    cv_evidence: str = Field(min_length=1)


class MLMatchResponse(BaseModel):
    similarity: float
    evidence_status: str
    matched: bool
    reason: str


# --------------------------------------------------
# ML model dependency
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_hybrid_matcher() -> HybridMatcher:
    """
    Load the ML model once and reuse it across requests.
    """
    return HybridMatcher(threshold=0.20)


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "app": "CVision",
    }


# --------------------------------------------------
# Existing text analysis endpoint
# --------------------------------------------------

@app.post(
    "/api/v1/analyze",
    response_model=AnalysisResponse,
)
def analyze_text(payload: AnalyzeTextRequest):
    try:
        return service.analyze(
            payload.cv_text,
            payload.job_description,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# Existing CV file analysis endpoint
# --------------------------------------------------

@app.post(
    "/api/v1/analyze-file",
    response_model=AnalysisResponse,
)
async def analyze_file(
        cv: UploadFile = File(...),
        job_description: str = Form(...),
):
    if not cv.filename:
        raise HTTPException(
            status_code=422,
            detail="A named CV file is required.",
        )

    try:
        content = await cv.read(MAX_FILE_SIZE + 1)
        text = parse_cv(cv.filename, content)

        return service.analyze(
            text,
            job_description,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    finally:
        await cv.close()


# --------------------------------------------------
# NEW: Hybrid ML matching endpoint
# --------------------------------------------------

@app.post(
    "/api/v1/ml/match",
    response_model=MLMatchResponse,
)
def match_cv_evidence(
        request: MLMatchRequest,
        matcher: HybridMatcher = Depends(get_hybrid_matcher),
):
    """
    Match a job requirement against CV evidence
    using semantic similarity and evidence validation.
    """

    try:
        result = matcher.match(
            request.requirement,
            request.cv_evidence,
        )

        return MLMatchResponse(
            similarity=round(result.similarity, 3),
            evidence_status=result.evidence.status.value,
            matched=bool(result.predicted_match),
            reason=result.reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
