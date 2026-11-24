
import csv
from collections import Counter
from pathlib import Path


DATASET_PATH = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "evaluation"
        / "skill_pairs.csv"
)


def load_rows():
    with DATASET_PATH.open(
            newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def test_dataset_is_not_empty():
    rows = load_rows()
    assert len(rows) >= 12


def test_required_columns():
    rows = load_rows()

    required = {
        "id",
        "requirement",
        "cv_evidence",
        "label",
        "category",
    }

    assert required.issubset(rows[0].keys())


def test_labels_are_binary():
    rows = load_rows()

    assert all(
        row["label"] in {"0", "1"}
        for row in rows
    )


def test_no_empty_text():
    rows = load_rows()

    assert all(
        row["requirement"].strip()
        and row["cv_evidence"].strip()
        for row in rows
    )


def test_unique_ids():
    rows = load_rows()
    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids))


def test_both_classes_present():
    rows = load_rows()

    counts = Counter(row["label"] for row in rows)

    assert counts["0"] > 0
    assert counts["1"] > 0


def test_categories_present():
    rows = load_rows()

    assert all(row["category"].strip() for row in rows)
