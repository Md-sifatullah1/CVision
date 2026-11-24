
from unittest.mock import Mock

import pytest

from app.ml.evidence_validator import (
    EvidenceResult,
    EvidenceStatus,
)
from app.ml.hybrid_matcher import HybridMatcher


def make_matcher(score, evidence_status, threshold=0.20):
    semantic = Mock()
    semantic.similarity.return_value = score

    validator = Mock()
    validator.validate.return_value = EvidenceResult(
        status=evidence_status,
        reason="Test evidence classification",
    )

    return HybridMatcher(
        threshold=threshold,
        semantic_matcher=semantic,
        evidence_validator=validator,
    )


def test_supported_semantic_match():
    matcher = make_matcher(
        0.75,
        EvidenceStatus.SUPPORTED,
    )

    result = matcher.match(
        "Experience with REST APIs",
        "Developed endpoints using FastAPI",
    )

    assert result.predicted_match == 1
    assert result.similarity == 0.75


def test_negated_evidence_is_rejected():
    matcher = make_matcher(
        0.90,
        EvidenceStatus.NOT_SUPPORTED,
    )

    result = matcher.match(
        "Experience with Docker",
        "Have never worked with Docker",
    )

    assert result.predicted_match == 0


def test_insufficient_evidence_is_rejected():
    matcher = make_matcher(
        0.80,
        EvidenceStatus.INSUFFICIENT,
    )

    result = matcher.match(
        "Production ML experience",
        "Completed an introductory ML course",
    )

    assert result.predicted_match == 0


def test_low_similarity_is_rejected():
    matcher = make_matcher(
        0.10,
        EvidenceStatus.SUPPORTED,
    )

    result = matcher.match(
        "Experience with Kubernetes",
        "Developed CSS layouts",
    )

    assert result.predicted_match == 0


def test_threshold_boundary():
    matcher = make_matcher(
        0.20,
        EvidenceStatus.SUPPORTED,
    )

    result = matcher.match("Requirement", "Evidence")

    assert result.predicted_match == 1


def test_invalid_threshold():
    with pytest.raises(ValueError):
        HybridMatcher(threshold=1.5)


def test_empty_input():
    matcher = make_matcher(
        0.75,
        EvidenceStatus.SUPPORTED,
    )

    with pytest.raises(ValueError):
        matcher.match("", "Some evidence")
