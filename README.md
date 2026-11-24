# CVision

**CV Analysis and Interview Preparation Platform**

CVision is a Python-based application that compares a candidate's CV against a job description, identifies matching skills and requirements not evidenced in the CV, and generates interview preparation questions and suggested study topics.

It provides a reusable FastAPI backend and a lightweight web interface built with HTML, CSS, and JavaScript.

**Version:** 1.0.0 (initial release candidate)

**Matching approach:** Dictionary-based keyword matching. This is an explainable rule-based baseline, not a trained machine learning model.

## Features

- **CV parsing:** Extract selectable text from PDF and DOCX documents.
- **Skill extraction:** Detect technical skills using a configurable dictionary of skill names and aliases.
- **Job matching:** Compare recognized job requirements against skills mentioned in the CV.
- **Evidence extraction:** Show supporting text from the CV for matched skills.
- **Gap identification:** Highlight job requirements that were not evidenced in the CV.
- **Interview preparation:** Generate template-based interview questions from matching skills and identified gaps.
- **Study recommendations:** Suggest topics based on requirements not found in the CV.
- **REST API:** Expose analysis functionality for integration with other applications.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Data validation | Pydantic |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Matching engine | Python, regular expressions |
| Frontend | HTML, CSS, JavaScript |
| Testing | pytest, HTTPX |
| API documentation | OpenAPI / Swagger UI |

No database, paid API, or external AI service is required.

## Project Architecture

```text
CVision/
├── app/
│   ├── api/
│   │   └── schemas.py
│   ├── core/
│   │   ├── models.py
│   │   ├── parser.py
│   │   └── skills.py
│   ├── ml/
│   │   └── matcher.py
│   ├── services/
│   │   └── analysis.py
│   └── main.py
├── data/
│   └── skills.json
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── tests/
│   ├── test_parser.py
│   ├── test_analysis.py
│   └── test_api.py
├── requirements.txt
└── README.md
```

### How CVision Works

1. A user uploads a PDF/DOCX CV and enters a job description.
2. The frontend sends the information to FastAPI.
3. The parser extracts text from the CV.
4. The analysis service calls the matching engine.
5. The matcher identifies supported skills and requirements not evidenced in the CV.
6. The service generates interview questions and study recommendations.
7. FastAPI returns structured JSON results to the frontend.

The matching engine is designed to be replaceable. Future implementations can use semantic similarity without requiring a complete redesign of the analysis service.

## Installation

### Requirements

- Python 3.10 or newer
- Git
- A modern web browser

### 1. Clone the repository

```bash
git clone https://github.com/Md-sifatullah1/CVision.git
cd CVision
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Running CVision Locally

CVision uses separate local servers for its backend and frontend.

### Terminal 1: Start FastAPI

From the project root:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Backend: http://localhost:8000

Interactive API documentation: http://localhost:8000/docs

### Terminal 2: Start the frontend

Open another terminal from the project root:

```bash
python -m http.server 5500 --directory static
```

Open the application:

**http://localhost:5500**

Both servers must remain running while using the web interface.

The frontend is configured to call the backend at `http://localhost:8000`. The backend permits the configured local development frontend origins through CORS.

## REST API

### Health check

`GET /api/v1/health`

Example response:

```json
{
  "status": "ok",
  "app": "CVision"
}
```

### Analyze CV text

`POST /api/v1/analyze`

Example request:

```json
{
  "cv_text": "Developed REST APIs using Python and Git.",
  "job_description": "Experience with Python, Docker and Git required."
}
```

Example curl command:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"cv_text":"Developed REST APIs using Python and Git.","job_description":"Experience with Python, Docker and Git required."}'
```

The response includes:

- `matching_skills`
- `missing_requirements`
- `interview_questions`
- `study_topics`
- `method`
- `disclaimer`

### Analyze an uploaded CV

`POST /api/v1/analyze-file`

Accepts multipart form data:

| Field | Type | Description |
|---|---|---|
| `cv` | File | PDF or DOCX CV, maximum 5 MB |
| `job_description` | String | Job description text |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/analyze-file \
  -F "cv=@resume.pdf" \
  -F "job_description=Experience with Python and Docker required."
```

Replace `resume.pdf` with the path to an existing PDF file.

## Testing

Run all automated tests:

```bash
python -m pytest -v
```

The initial release candidate has 30 passing automated tests covering CV parsing, analysis logic, API validation, and file uploads.

## Reusable Backend

CVision's analysis logic is independent of its web interface.

Other Python applications can use the service directly:

```python
from app.services.analysis import AnalysisService

service = AnalysisService()

result = service.analyze(
    cv_text="Developed Python applications.",
    job_description="Python and Docker required."
)

print(result["matching_skills"])
print(result["missing_requirements"])
```

External applications can integrate through the versioned REST API.

## Known Limitations

- Matching relies on predefined skills and aliases rather than semantic understanding.
- Keyword matching may produce false positives, including negated experience.
- A requirement not found in the CV does not necessarily mean the candidate lacks that skill.
- Scanned or image-only PDFs are not supported because OCR is not implemented.
- Interview questions are generated from templates rather than a language model.
- Skill recognition is limited to the configured dictionary.
- CVs are processed in memory and are not intentionally stored by the application.
- The application is a local development prototype. Public deployment requires additional security, privacy, and operational safeguards, including appropriate authentication, rate limiting, HTTPS, and upload validation.

## Future Development: Machine Learning

The next development phase will focus on implementing and evaluating actual ML-based matching.

Planned improvements:

1. Create a labeled dataset of CV evidence and job requirement pairs.
2. Establish a TF-IDF baseline and evaluate precision, recall, and F1.
3. Introduce Sentence Transformers for semantic similarity.
4. Compare semantic matching against the keyword baseline.
5. Tune similarity thresholds and analyze false positives.
6. Improve interview question generation using grounded evidence.

The goal is to develop an explainable, measurable matching system rather than relying on an unverified AI score.

## License

No license has been selected for this repository yet.


## CVision v2: Hybrid ML Skill Matching

CVision analyzes CVs against job descriptions using a
hybrid approach combining keyword extraction, semantic
similarity, and rule-based evidence validation.

### Features

- PDF and DOCX CV parsing
- Job description skill extraction
- Hybrid ML skill matching
- Missing requirement identification
- CV-based interview questions
- Suggested study topics
- REST API with FastAPI
- Browser-based frontend

### ML Architecture

1. KeywordMatcher identifies candidate skills.
2. Sentence Transformers calculates semantic similarity.
3. EvidenceValidator checks for explicit limitations.
4. HybridMatcher combines similarity and validation.
5. AnalysisService generates the final analysis.

Embedding model: `all-MiniLM-L6-v2`

The hybrid matcher uses a similarity threshold of `0.20`.

### Evaluation

On a 24-example synthetic challenge dataset:

- Accuracy: 70.8%
- Precision: 66.7%
- Recall: 83.3%
- F1 score: 0.741

These metrics describe the evaluated hybrid pair matcher,
not the complete CV analysis pipeline. The dataset is small
and was designed to probe known weaknesses.

### Limitations

- Skill discovery depends on a predefined dictionary.
- Semantic similarity does not establish proficiency.
- Evidence validation can misinterpret complex statements.
- Results should be reviewed manually.
- No external database is required.

### Running Locally

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app.main:app --reload
```

Start the frontend in another terminal:

```bash
python -m http.server 5500 --directory static
```

Open http://localhost:5500.

### Testing

```bash
python -m pytest -q
```
