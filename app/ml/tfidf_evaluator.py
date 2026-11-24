
from pathlib import Path

from app.ml.baseline_evaluator import (
    DATASET_PATH,
    load_evaluation_dataset,
)
from app.ml.evaluation import EvaluationMetrics, evaluate
from app.ml.tfidf_matcher import TfidfMatcher


def evaluate_tfidf(
        path: Path = DATASET_PATH,
        threshold: float = 0.3,
) -> tuple[EvaluationMetrics, list[dict]]:
    """Evaluate TF-IDF on labeled requirement/evidence pairs."""

    rows = load_evaluation_dataset(path)
    matcher = TfidfMatcher(threshold=threshold)

    y_true = []
    y_pred = []
    predictions = []

    for row in rows:
        actual = int(row["label"])
        score = matcher.similarity(
            requirement=row["requirement"],
            cv_evidence=row["cv_evidence"],
        )
        predicted = int(score >= matcher.threshold)

        y_true.append(actual)
        y_pred.append(predicted)

        predictions.append({
            "id": row["id"],
            "category": row["category"],
            "actual": actual,
            "predicted": predicted,
            "score": score,
            "correct": actual == predicted,
        })

    return evaluate(y_true, y_pred), predictions


if __name__ == "__main__":
    metrics, predictions = evaluate_tfidf()

    print("\nCVision TF-IDF Baseline Evaluation")
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
    print("-" * 75)

    for prediction in predictions:
        status = "PASS" if prediction["correct"] else "FAIL"
        print(
            f"ID {prediction['id']:>2} | "
            f"{prediction['category']:<22} | "
            f"Score: {prediction['score']:.3f} | "
            f"Actual: {prediction['actual']} | "
            f"Predicted: {prediction['predicted']} | "
            f"{status}"
        )
