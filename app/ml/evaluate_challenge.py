
import csv
from collections import defaultdict
from pathlib import Path

from app.ml.evaluation import evaluate
from app.ml.hybrid_matcher import HybridMatcher


DATASET = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "evaluation"
        / "challenge_v1.csv"
)


def load_challenge_data():
    with DATASET.open(
            newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def main():
    rows = load_challenge_data()

    matcher = HybridMatcher(threshold=0.20)

    actual = []
    predicted = []
    category_results = defaultdict(
        lambda: {"actual": [], "predicted": []}
    )

    print("\nCVision Challenge Evaluation")
    print("=" * 60)

    for row in rows:
        result = matcher.match(
            row["requirement"],
            row["cv_evidence"],
        )

        label = int(row["label"])
        prediction = result.predicted_match

        actual.append(label)
        predicted.append(prediction)

        category = row["category"]

        category_results[category]["actual"].append(label)
        category_results[category]["predicted"].append(prediction)

        status = (
            "PASS"
            if prediction == label
            else "FAIL"
        )

        print(
            f"ID {int(row['id']):>2} | "
            f"{category:<23} | "
            f"Score: {result.similarity:.3f} | "
            f"Evidence: {result.evidence.status.value:<14} | "
            f"Actual: {label} | "
            f"Predicted: {prediction} | "
            f"{status}"
        )

    metrics = evaluate(actual, predicted)

    print("\nOverall Results")
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1 score:  {metrics.f1:.3f}")

    print("\nConfusion Matrix")
    print(f"TP: {metrics.true_positives}")
    print(f"FP: {metrics.false_positives}")
    print(f"TN: {metrics.true_negatives}")
    print(f"FN: {metrics.false_negatives}")

    print("\nAccuracy by Category")

    for category, values in category_results.items():
        correct = sum(
            a == p
            for a, p in zip(
                values["actual"],
                values["predicted"],
            )
        )

        total = len(values["actual"])

        print(
            f"{category:<23} | "
            f"{correct}/{total} correct"
        )


if __name__ == "__main__":
    main()
