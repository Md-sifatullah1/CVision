
import csv
from pathlib import Path

from app.ml.evaluation import evaluate, EvaluationMetrics
from app.ml.matcher import KeywordMatcher


DATASET_PATH = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "evaluation"
        / "skill_pairs.csv"
)


def load_evaluation_dataset(
        path: Path = DATASET_PATH,
) -> list[dict]:
    """Load labeled requirement/evidence pairs."""

    with path.open(
            newline="",
            encoding="utf-8",
    ) as file:
        return list(csv.DictReader(file))


def predict_keyword_match(
        requirement: str,
        cv_evidence: str,
        matcher: KeywordMatcher,
) -> int:
    """Return 1 if the matcher finds supporting skill evidence."""

    matches, _ = matcher.match(
        cv_text=cv_evidence,
        job_description=requirement,
    )

    return int(bool(matches))


def evaluate_keyword_baseline(
        path: Path = DATASET_PATH,
) -> tuple[EvaluationMetrics, list[dict]]:
    """Evaluate the v1 keyword matcher on labeled pairs."""

    rows = load_evaluation_dataset(path)
    matcher = KeywordMatcher()

    y_true = []
    y_pred = []
    predictions = []

    for row in rows:
        actual = int(row["label"])

        predicted = predict_keyword_match(
            requirement=row["requirement"],
            cv_evidence=row["cv_evidence"],
            matcher=matcher,
        )

        y_true.append(actual)
        y_pred.append(predicted)

        predictions.append(
            {
                "id": row["id"],
                "category": row["category"],
                "actual": actual,
                "predicted": predicted,
                "correct": actual == predicted,
            }
        )

    metrics = evaluate(y_true, y_pred)

    return metrics, predictions


if __name__ == "__main__":
    metrics, predictions = evaluate_keyword_baseline()

    print("\nCVision v1 Keyword Baseline Evaluation")
    print("=" * 45)

    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1 score:  {metrics.f1:.3f}")

    print("\nConfusion Matrix")
    print(f"TP: {metrics.true_positives}")
    print(f"FP: {metrics.false_positives}")
    print(f"TN: {metrics.true_negatives}")
    print(f"FN: {metrics.false_negatives}")

    print("\nIndividual Predictions")
    print("-" * 45)

    for prediction in predictions:
        status = "PASS" if prediction["correct"] else "FAIL"

        print(
            f"ID {prediction['id']:>2} | "
            f"{prediction['category']:<22} | "
            f"Actual: {prediction['actual']} | "
            f"Predicted: {prediction['predicted']} | "
            f"{status}"
        )
