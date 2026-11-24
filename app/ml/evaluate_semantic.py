
from app.ml.evaluation import evaluate
from app.ml.semantic_matcher import SemanticMatcher
from app.ml.train_tfidf import load_dataset, get_split


def score_examples(matcher, rows):
    """Calculate semantic similarity once per example."""
    results = []

    for row in rows:
        score = matcher.similarity(
            row["requirement"],
            row["cv_evidence"],
        )

        results.append({
            "id": row["id"],
            "category": row["category"],
            "actual": int(row["label"]),
            "score": score,
        })

    return results


def evaluate_threshold(results, threshold):
    actual = [row["actual"] for row in results]
    predicted = [
        int(row["score"] >= threshold)
        for row in results
    ]

    return evaluate(actual, predicted)


def main():
    rows = load_dataset()

    validation_rows = get_split(rows, "val")
    test_rows = get_split(rows, "test")

    matcher = SemanticMatcher()

    print("\nCVision Semantic Matching Evaluation")
    print("=" * 45)

    validation_scores = score_examples(
        matcher, validation_rows
    )

    print("\nValidation Similarity Scores")

    for row in validation_scores:
        print(
            f"ID {row['id']:>2} | "
            f"{row['category']:<22} | "
            f"Actual: {row['actual']} | "
            f"Score: {row['score']:.3f}"
        )

    thresholds = [
        0.20, 0.25, 0.30, 0.35,
        0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75,
        0.80,
    ]

    best_threshold = None
    best_f1 = -1.0

    print("\nValidation Threshold Results")

    for threshold in thresholds:
        metrics = evaluate_threshold(
            validation_scores, threshold
        )

        print(
            f"Threshold: {threshold:.2f} | "
            f"F1: {metrics.f1:.3f} | "
            f"Accuracy: {metrics.accuracy:.3f}"
        )

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = threshold

    print(
        f"\nSelected threshold: {best_threshold:.2f}"
    )

    test_scores = score_examples(matcher, test_rows)

    test_metrics = evaluate_threshold(
        test_scores, best_threshold
    )

    print("\nIndividual Test Predictions")
    print("-" * 85)

    for row in test_scores:
        predicted = int(
            row["score"] >= best_threshold
        )

        status = (
            "PASS"
            if predicted == row["actual"]
            else "FAIL"
        )

        print(
            f"ID {row['id']:>2} | "
            f"{row['category']:<22} | "
            f"Score: {row['score']:.3f} | "
            f"Actual: {row['actual']} | "
            f"Predicted: {predicted} | "
            f"{status}"
        )

    print("\nHeld-out Test Results")
    print(f"Accuracy:  {test_metrics.accuracy:.3f}")
    print(f"Precision: {test_metrics.precision:.3f}")
    print(f"Recall:    {test_metrics.recall:.3f}")
    print(f"F1 score:  {test_metrics.f1:.3f}")

    print("\nConfusion Matrix")
    print(f"TP: {test_metrics.true_positives}")
    print(f"FP: {test_metrics.false_positives}")
    print(f"TN: {test_metrics.true_negatives}")
    print(f"FN: {test_metrics.false_negatives}")


if __name__ == "__main__":
    main()
