
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float


def evaluate(
        y_true: list[int],
        y_pred: list[int],
) -> EvaluationMetrics:
    """Evaluate binary matching predictions."""

    if not y_true or len(y_true) != len(y_pred):
        raise ValueError(
            "Labels must be nonempty and have equal lengths."
        )

    if any(value not in (0, 1) for value in y_true + y_pred):
        raise ValueError("Labels must be binary: 0 or 1.")

    tp = sum(
        actual == 1 and predicted == 1
        for actual, predicted in zip(y_true, y_pred)
    )

    fp = sum(
        actual == 0 and predicted == 1
        for actual, predicted in zip(y_true, y_pred)
    )

    tn = sum(
        actual == 0 and predicted == 0
        for actual, predicted in zip(y_true, y_pred)
    )

    fn = sum(
        actual == 1 and predicted == 0
        for actual, predicted in zip(y_true, y_pred)
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    accuracy = (tp + tn) / len(y_true)

    return EvaluationMetrics(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )
