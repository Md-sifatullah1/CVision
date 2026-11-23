import json
import re
from pathlib import Path

SKILL_FILE = Path(__file__).resolve().parents[2] / 'data' / 'skills.json'


def load_skills() -> dict[str, list[str]]:
    return json.loads(SKILL_FILE.read_text(encoding='utf-8'))


def mentions(text: str, aliases: list[str]) -> bool:
    return any(re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text, re.I) for alias in aliases)


def extract_skills(text: str) -> list[str]:
    return [name for name, aliases in load_skills().items() if mentions(text, aliases)]


def evidence_for(text: str, aliases: list[str]) -> str:
    # Sentence-like chunks preserve a short source quote for the UI.
    chunks = [chunk.strip() for chunk in re.split(r'(?<=[.!?])\s+|\n+', text) if chunk.strip()]
    return next((chunk[:350] for chunk in chunks if mentions(chunk, aliases)), '')
