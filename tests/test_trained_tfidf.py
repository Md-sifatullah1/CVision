
import pytest

from sklearn.exceptions import NotFittedError

from app.ml.trained_tfidf import TrainedTfidfMatcher


TRAINING_TEXTS = [
    "Python backend development",
    "Developed Python backend applications",
    "React frontend development",
    "Created React components",
    "Docker container deployment",
    "Built Docker images",
]


def test_model_requires_training():
    matcher = TrainedTfidfMatcher()

    with pytest.raises(NotFittedError):
        matcher.similarity("Python", "Python")


def test_identical_text_similarity():
    matcher = TrainedTfidfMatcher()
    matcher.fit(TRAINING_TEXTS)

    score = matcher.similarity(
        "Python backend development",
        "Python backend development",
    )

    assert score == pytest.approx(1.0)



def test_unrelated_text_similarity():
    matcher = TrainedTfidfMatcher()
    matcher.fit(TRAINING_TEXTS)

    score = matcher.similarity(
        "Python backend development",
        "React frontend development",
    )

    assert 0.0 < score < 0.3



def test_prediction():
    matcher = TrainedTfidfMatcher(threshold=0.3)
    matcher.fit(TRAINING_TEXTS)

    assert matcher.predict(
        "Python backend development",
        "Developed Python backend applications",
    ) == 1


def test_invalid_threshold():
    with pytest.raises(ValueError):
        TrainedTfidfMatcher(threshold=1.5)


def test_empty_training_data():
    matcher = TrainedTfidfMatcher()

    with pytest.raises(ValueError):
        matcher.fit([])
