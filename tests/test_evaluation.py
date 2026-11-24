
import pytest

from app.ml.evaluation import evaluate


def test_perfect_predictions():
    result = evaluate(
        [1, 1, 0, 0],
        [1, 1, 0, 0],
    )

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.accuracy == 1.0


def test_confusion_matrix_counts():
    result = evaluate(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    )

    assert result.true_positives == 5
    assert result.false_positives == 2
    assert result.true_negatives == 4
    assert result.false_negatives == 1


def test_precision_recall_f1():
    result = evaluate(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0],
    )

    assert result.precision == pytest.approx(5 / 7)
    assert result.recall == pytest.approx(5 / 6)
    assert result.f1 == pytest.approx(10 / 13)


def test_no_positive_predictions():
    result = evaluate(
        [1, 0, 1, 0],
        [0, 0, 0, 0],
    )

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_reject_empty_labels():
    with pytest.raises(ValueError):
        evaluate([], [])


def test_reject_different_lengths():
    with pytest.raises(ValueError):
        evaluate([1, 0], [1])


def test_reject_nonbinary_labels():
    with pytest.raises(ValueError):
        evaluate([1, 2], [1, 0])
