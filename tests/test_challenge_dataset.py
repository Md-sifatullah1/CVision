
import csv
from collections import Counter
from pathlib import Path


DATASET = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "evaluation"
        / "challenge_v1.csv"
)


def load_rows():
    with DATASET.open(
            newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def test_dataset_size():
    assert len(load_rows()) == 24


def test_unique_ids():
    rows = load_rows()
    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids))


def test_valid_labels():
    assert all(
        row["label"] in {"0", "1"}
        for row in load_rows()
    )


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


def test_no_empty_text():
    assert all(
        row["requirement"].strip()
        and row["cv_evidence"].strip()
        for row in load_rows()
    )


def test_category_distribution():
    counts = Counter(
        row["category"] for row in load_rows()
    )

    assert len(counts) == 6
    assert all(count == 4 for count in counts.values())


def test_both_classes_present():
    labels = Counter(
        row["label"] for row in load_rows()
    )

    assert labels["0"] == 12
    assert labels["1"] == 12
