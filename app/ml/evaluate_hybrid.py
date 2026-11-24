
from app.ml.evaluation import evaluate
from app.ml.hybrid_matcher import HybridMatcher
from app.ml.train_tfidf import load_dataset, get_split


def main():
    rows = load_dataset()
    test_rows = get_split(rows, "test")

    matcher = HybridMatcher(threshold=0.20)

    actual = []
    predicted = []

    print("\nCVision Hybrid Matching Evaluation")
    print("=" * 50)

    for row in test_rows:
        result = matcher.match(
            row["requirement"],
            row["cv_evidence"],
        )

        label = int(row["label"])

        actual.append(label)
        predicted.append(result.predicted_match)

        status = (
            "PASS"
            if label == result.predicted_match
            else "FAIL"
        )

        print(
            f"ID {row['id']:>2} | "
            f"Similarity: {result.similarity:.3f} | "
            f"Evidence: {result.evidence.status.value:<14} | "
            f"Actual: {label} | "
            f"Predicted: {result.predicted_match} | "
            f"{status}"
        )

    metrics = evaluate(actual, predicted)

    print("\nHybrid Evaluation Results")
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1 score:  {metrics.f1:.3f}")

    print("\nConfusion Matrix")
    print(f"TP: {metrics.true_positives}")
    print(f"FP: {metrics.false_positives}")
    print(f"TN: {metrics.true_negatives}")
    print(f"FN: {metrics.false_negatives}")


if __name__ == "__main__":
    main()
