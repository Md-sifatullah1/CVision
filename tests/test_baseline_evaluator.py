
import csv

from app.ml.baseline_evaluator import (
    evaluate_keyword_baseline,
    load_evaluation_dataset,
    predict_keyword_match,
)
from app.ml.matcher import KeywordMatcher


def test_load_dataset():
    rows = load_evaluation_dataset()

    assert len(rows) == 12
    assert rows[0]["id"] == "1"


def test_predict_exact_keyword_match():
    matcher = KeywordMatcher()

    result = predict_keyword_match(
        requirement="Experience with Python",
        cv_evidence="Developed applications using Python",
        matcher=matcher,
    )

    assert result == 1


def test_predict_unrelated_skills():
    matcher = KeywordMatcher()

    result = predict_keyword_match(
        requirement="Experience with Kubernetes",
        cv_evidence="Developed React applications",
        matcher=matcher,
    )

    assert result == 0


def test_evaluation_runner(tmp_path):
    dataset = tmp_path / "evaluation.csv"

    with dataset.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "requirement",
                "cv_evidence",
                "label",
                "category",
            ],
        )

        writer.writeheader()

        writer.writerows(
            [
                {
                    "id": "1",
                    "requirement": "Python required",
                    "cv_evidence": "Built Python applications",
                    "label": "1",
                    "category": "exact_match",
                },
                {
                    "id": "2",
                    "requirement": "Kubernetes required",
                    "cv_evidence": "Built React applications",
                    "label": "0",
                    "category": "unrelated",
                },
            ]
        )

    metrics, predictions = evaluate_keyword_baseline(dataset)

    assert metrics.accuracy == 1.0
    assert metrics.true_positives == 1
    assert metrics.true_negatives == 1
    assert len(predictions) == 2
