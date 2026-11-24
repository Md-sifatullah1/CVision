
import csv
from collections import Counter
from pathlib import Path


DATASET_PATH = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "ml"
        / "skill_pairs_v2.csv"
)


def load_dataset():
    with DATASET_PATH.open(
            newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def test_dataset_size():
    rows = load_dataset()
    assert len(rows) == 50


def test_required_columns():
    rows = load_dataset()

    required = {
        "id",
        "requirement",
        "cv_evidence",
        "label",
        "category",
        "split",
    }

    assert required.issubset(rows[0].keys())


def test_unique_ids():
    rows = load_dataset()
    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids))


def test_valid_labels():
    rows = load_dataset()

    assert all(
        row["label"] in {"0", "1"}
        for row in rows
    )


def test_valid_splits():
    rows = load_dataset()

    assert all(
        row["split"] in {"train", "val", "test"}
        for row in rows
    )


def test_split_sizes():
    rows = load_dataset()

    counts = Counter(row["split"] for row in rows)

    assert counts == {
        "train": 30,
        "val": 10,
        "test": 10,
    }


def test_both_classes_in_each_split():
    rows = load_dataset()

    for split in ("train", "val", "test"):
        labels = {
            row["label"]
            for row in rows
            if row["split"] == split
        }

        assert labels == {"0", "1"}


def test_no_empty_text():
    rows = load_dataset()

    assert all(
        row["requirement"].strip()
        and row["cv_evidence"].strip()
        for row in rows
    )
