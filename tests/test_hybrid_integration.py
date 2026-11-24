
from unittest.mock import Mock

from app.services.analysis import AnalysisService


def make_service(predicted_match):
    hybrid = Mock()
    hybrid.match.return_value.predicted_match = predicted_match

    return AnalysisService(
        hybrid_matcher=hybrid,
        use_ml=True,
    ), hybrid


def test_supported_skill_appears_in_matches():
    service, hybrid = make_service(1)

    result = service.analyze(
        "Developed SQL queries for reporting.",
        "Experience with SQL.",
    )

    assert any(
        match["skill"] == "SQL"
        for match in result["matching_skills"]
    )

    assert result["method"] == "hybrid_semantic_evidence"
    hybrid.match.assert_called_once()


def test_rejected_skill_appears_in_gaps():
    service, hybrid = make_service(0)

    result = service.analyze(
        "Developed SQL queries for reporting.",
        "Experience with SQL.",
    )

    assert not any(
        match["skill"] == "SQL"
        for match in result["matching_skills"]
    )

    assert any(
        gap["skill"] == "SQL"
        for gap in result["missing_requirements"]
    )


def test_missing_skill_remains_a_gap():
    service, hybrid = make_service(1)

    result = service.analyze(
        "Developed SQL queries for reporting.",
        "Experience with Kubernetes.",
    )

    assert any(
        gap["skill"] == "Kubernetes"
        for gap in result["missing_requirements"]
    )

    hybrid.match.assert_not_called()