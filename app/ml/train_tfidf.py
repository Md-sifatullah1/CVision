
import csv
from pathlib import Path

from app.ml.evaluation import evaluate
from app.ml.trained_tfidf import TrainedTfidfMatcher


DATASET_PATH = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "ml"
        / "skill_pairs_v2.csv"
)


def load_dataset():
    with DATASET_PATH.open(
            newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def get_split(rows, split):
    return [
        row for row in rows
        if row["split"] == split
    ]


def evaluate_split(matcher, rows):
    y_true = []
    y_pred = []

    for row in rows:
        y_true.append(int(row["label"]))
        y_pred.append(
            matcher.predict(
                row["requirement"],
                row["cv_evidence"],
            )
        )

    return evaluate(y_true, y_pred)


def main():
    rows = load_dataset()

    train_rows = get_split(rows, "train")
    val_rows = get_split(rows, "val")
    test_rows = get_split(rows, "test")

    matcher = TrainedTfidfMatcher()

    training_texts = []

    for row in train_rows:
        training_texts.extend([
            row["requirement"],
            row["cv_evidence"],
        ])

    matcher.fit(training_texts)

    print("\nCVision Trained TF-IDF Evaluation")
    print("=" * 45)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    best_threshold = None
    best_f1 = -1.0

    print("\nValidation Results")

    print("\nValidation Similarity Scores")
    for row in val_rows:
        score = matcher.similarity(
            row["requirement"],
            row["cv_evidence"],
        )
        print(
            f"ID {row['id']:>2} | "
            f"Label: {row['label']} | "
            f"Score: {score:.3f}"
        )

    for threshold in thresholds:
        matcher.threshold = threshold
        metrics = evaluate_split(matcher, val_rows)

        print(
            f"Threshold: {threshold:.2f} | "
            f"F1: {metrics.f1:.3f} | "
            f"Accuracy: {metrics.accuracy:.3f}"
        )

        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = threshold

    matcher.threshold = best_threshold

    print(f"\nSelected threshold: {best_threshold:.2f}")

    test_metrics = evaluate_split(matcher, test_rows)


    print("\nIndividual Test Predictions")
    print("-" * 85)

    for row in test_rows:
        score = matcher.similarity(
            row["requirement"],
            row["cv_evidence"],
        )

        actual = int(row["label"])
        predicted = int(score >= matcher.threshold)

        status = "PASS" if actual == predicted else "FAIL"

        print(
            f"ID {row['id']:>2} | "
            f"{row['category']:<22} | "
            f"Score: {score:.3f} | "
            f"Actual: {actual} | "
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
