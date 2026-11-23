from pydantic import BaseModel, Field

class AnalyzeTextRequest(BaseModel):
    cv_text: str = Field(min_length=1, max_length=100000)
    job_description: str = Field(min_length=1, max_length=50000)

class MatchResponse(BaseModel):
    skill: str
    requirement: str
    cv_evidence: str

class GapResponse(BaseModel):
    skill: str
    requirement: str
    status: str

class QuestionResponse(BaseModel):
    question: str
    category: str
    related_skill: str

class StudyResponse(BaseModel):
    topic: str
    reason: str

class AnalysisResponse(BaseModel):
    matching_skills: list[MatchResponse]
    missing_requirements: list[GapResponse]
    interview_questions: list[QuestionResponse]
    study_topics: list[StudyResponse]
    method: str
    disclaimer: str
