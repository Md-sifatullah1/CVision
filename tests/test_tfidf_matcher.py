
import pytest

from app.ml.tfidf_matcher import TfidfMatcher


def test_identical_text_similarity():
    matcher = TfidfMatcher()

    score = matcher.similarity(
        "Python backend development",
        "Python backend development",
    )

    assert score == pytest.approx(1.0)


def test_unrelated_text_similarity():
    matcher = TfidfMatcher()

    score = matcher.similarity(
        "Kubernetes deployment",
        "React frontend development",
    )

    assert score == pytest.approx(0.0)


def test_identical_text_predicts_match():
    matcher = TfidfMatcher(threshold=0.3)

    assert matcher.predict(
        "Python programming",
        "Python programming",
    ) == 1


def test_unrelated_text_predicts_no_match():
    matcher = TfidfMatcher(threshold=0.3)

    assert matcher.predict(
        "Docker containers",
        "React interfaces",
    ) == 0


def test_reject_invalid_threshold():
    with pytest.raises(ValueError):
        TfidfMatcher(threshold=-0.1)

    with pytest.raises(ValueError):
        TfidfMatcher(threshold=1.1)


def test_reject_empty_text():
    matcher = TfidfMatcher()

    with pytest.raises(ValueError):
        matcher.similarity("", "Python")

    with pytest.raises(ValueError):
        matcher.similarity("Python", " ")
