
from unittest.mock import patch

import numpy as np
import pytest

from app.ml.semantic_matcher import SemanticMatcher


@patch("app.ml.semantic_matcher.SentenceTransformer")
def test_identical_embeddings(mock_model):
    mock_model.return_value.encode.return_value = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    matcher = SemanticMatcher()

    score = matcher.similarity(
        "Python development",
        "Python development",
    )

    assert score == pytest.approx(1.0)


@patch("app.ml.semantic_matcher.SentenceTransformer")
def test_orthogonal_embeddings(mock_model):
    mock_model.return_value.encode.return_value = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    matcher = SemanticMatcher()

    score = matcher.similarity(
        "Python development",
        "React development",
    )

    assert score == pytest.approx(0.0)


@patch("app.ml.semantic_matcher.SentenceTransformer")
def test_prediction(mock_model):
    mock_model.return_value.encode.return_value = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])

    matcher = SemanticMatcher(threshold=0.5)

    assert matcher.predict(
        "Python programming",
        "Python development",
    ) == 1


def test_invalid_threshold():
    with pytest.raises(ValueError):
        SemanticMatcher(threshold=1.5)


@patch("app.ml.semantic_matcher.SentenceTransformer")
def test_empty_text(mock_model):
    matcher = SemanticMatcher()

    with pytest.raises(ValueError):
        matcher.similarity("", "Python")

    with pytest.raises(ValueError):
        matcher.similarity("Python", " ")
