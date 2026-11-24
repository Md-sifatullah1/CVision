
import csv

import pytest

from app.ml.tfidf_evaluator import evaluate_tfidf


def test_evaluator_returns_all_predictions():
    metrics, predictions = evaluate_tfidf()

    assert len(predictions) == 12
    assert 0.0 <= metrics.f1 <= 1.0


def test_evaluator_uses_threshold(tmp_path):
    path = tmp_path / "pairs.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
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
        writer.writerow({
            "id": "1",
            "requirement": "Python backend development",
            "cv_evidence": "Python backend development",
            "label": "1",
            "category": "exact_match",
        })

    metrics, predictions = evaluate_tfidf(
        path=path,
        threshold=0.5,
    )

    assert predictions[0]["predicted"] == 1
    assert predictions[0]["score"] == pytest.approx(1.0)
    assert metrics.true_positives == 1


def test_evaluator_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        evaluate_tfidf(threshold=1.5)
