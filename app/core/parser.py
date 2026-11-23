from io import BytesIO

MAX_FILE_SIZE = 5 * 1024 * 1024


def parse_cv(filename: str, content: bytes) -> str:
    if not content or len(content) > MAX_FILE_SIZE:
        raise ValueError('CV must be nonempty and no larger than 5 MB.')
    suffix = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if suffix == 'pdf':
        import fitz
        try:
            with fitz.open(stream=content, filetype='pdf') as document:
                text = '\n'.join(page.get_text() for page in document)
        except Exception as exc:
            raise ValueError('Could not read PDF. Check that it is valid and not password-protected.') from exc
    elif suffix == 'docx':
        from docx import Document
        try:
            document = Document(BytesIO(content))
            paragraphs = [p.text for p in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    paragraphs.extend(cell.text for cell in row.cells)
            text = '\n'.join(paragraphs)
        except Exception as exc:
            raise ValueError('Could not read DOCX file.') from exc
    else:
        raise ValueError('Only PDF and DOCX CVs are supported.')
    if not text.strip():
        raise ValueError('No selectable text found. Scanned/image-only CVs are not supported yet.')
    return text
