
from io import BytesIO

import fitz
import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)

    content = document.tobytes()
    document.close()

    return content


def create_docx(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)

    buffer = BytesIO()
    document.save(buffer)

    return buffer.getvalue()


def test_health():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "CVision",
    }


def test_text_endpoint():
    response = client.post(
        "/api/v1/analyze",
        json={
            "cv_text": "Python developer",
            "job_description": "Python and Docker",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["method"] == "hybrid_semantic_evidence"
    assert "Python" in {
        item["skill"] for item in data["matching_skills"]
    }
    assert "Docker" in {
        item["skill"] for item in data["missing_requirements"]
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"cv_text": "", "job_description": "Python"},
        {"cv_text": "Python", "job_description": ""},
        {"cv_text": "Python"},
        {"job_description": "Python"},
    ],
)
def test_invalid_text_input(payload):
    response = client.post(
        "/api/v1/analyze",
        json=payload,
    )

    assert response.status_code == 422


def test_pdf_upload():
    response = client.post(
        "/api/v1/analyze-file",
        data={"job_description": "Python and Docker"},
        files={
            "cv": (
                "resume.pdf",
                create_pdf("Experienced Python developer"),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["matching_skills"]


def test_docx_upload():
    response = client.post(
        "/api/v1/analyze-file",
        data={"job_description": "Python required"},
        files={
            "cv": (
                "resume.docx",
                create_docx("Python developer"),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["matching_skills"]


def test_invalid_file():
    response = client.post(
        "/api/v1/analyze-file",
        data={"job_description": "Python"},
        files={
            "cv": ("cv.txt", b"hello", "text/plain")
        },
    )

    assert response.status_code == 422


def test_missing_file():
    response = client.post(
        "/api/v1/analyze-file",
        data={"job_description": "Python"},
    )

    assert response.status_code == 422


def test_oversized_file():
    response = client.post(
        "/api/v1/analyze-file",
        data={"job_description": "Python"},
        files={
            "cv": (
                "resume.pdf",
                b"x" * (5 * 1024 * 1024 + 1),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 422
