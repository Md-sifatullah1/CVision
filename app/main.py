
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.schemas import AnalysisResponse, AnalyzeTextRequest
from app.core.parser import MAX_FILE_SIZE, parse_cv
from app.services.analysis import AnalysisService
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="CVision API",
    version="0.1.0",
    description="Reusable CV and job-description analysis service",
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


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "app": "CVision"}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
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


@app.post("/api/v1/analyze-file", response_model=AnalysisResponse)
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

        return service.analyze(text, job_description)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    finally:
        await cv.close()
