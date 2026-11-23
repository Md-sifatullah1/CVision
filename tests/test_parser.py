
from io import BytesIO

import fitz
import pytest
from docx import Document

from app.core.parser import MAX_FILE_SIZE, parse_cv


def create_pdf(text: str) -> bytes:
    """Generate a PDF entirely in memory."""
    document = fitz.open()
    page = document.new_page()

    if text:
        page.insert_text((72, 72), text)

    content = document.tobytes()
    document.close()
    return content


def create_docx(text: str) -> bytes:
    """Generate a DOCX entirely in memory."""
    document = Document()
    document.add_paragraph(text)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def test_parse_valid_pdf():
    content = create_pdf("Python developer with REST API experience")

    result = parse_cv("resume.pdf", content)

    assert "Python developer" in result
    assert "REST API" in result


def test_parse_valid_docx():
    content = create_docx("Experienced Java developer")

    result = parse_cv("resume.docx", content)

    assert "Java developer" in result


def test_parse_docx_tables():
    document = Document()
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Skill"
    table.cell(0, 1).text = "Python"

    buffer = BytesIO()
    document.save(buffer)

    result = parse_cv("resume.docx", buffer.getvalue())

    assert "Skill" in result
    assert "Python" in result


def test_reject_empty_file():
    with pytest.raises(ValueError, match="nonempty"):
        parse_cv("resume.pdf", b"")


def test_reject_oversized_file():
    content = b"x" * (MAX_FILE_SIZE + 1)

    with pytest.raises(ValueError, match="5 MB"):
        parse_cv("resume.pdf", content)


def test_reject_unsupported_format():
    with pytest.raises(ValueError, match="Only PDF and DOCX"):
        parse_cv("resume.txt", b"Some resume text")


def test_reject_corrupted_pdf():
    with pytest.raises(ValueError, match="Could not read PDF"):
        parse_cv("resume.pdf", b"invalid pdf content")


def test_reject_corrupted_docx():
    with pytest.raises(ValueError, match="Could not read DOCX"):
        parse_cv("resume.docx", b"invalid docx content")


def test_reject_pdf_without_text():
    content = create_pdf("")

    with pytest.raises(ValueError, match="No selectable text"):
        parse_cv("resume.pdf", content)
