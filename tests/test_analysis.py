
import pytest

from app.ml.matcher import Matcher
from app.services.analysis import AnalysisService
from unittest.mock import Mock


def test_matching_and_missing():
    result = AnalysisService().analyze(
        "I built services with Python and Git.",
        "Requires Python, Git and Docker."
    )

    matched = {m["skill"] for m in result["matching_skills"]}
    missing = {g["skill"] for g in result["missing_requirements"]}

    assert {"Python", "Git"}.issubset(matched)
    assert "Docker" in missing


@pytest.mark.parametrize(
    "cv_text, job_description",
    [
        ("", "Requires Python"),
        ("Python developer", ""),
        ("   ", "Requires Python"),
        ("Python developer", "   "),
    ],
)
def test_reject_empty_input(cv_text, job_description):
    with pytest.raises(ValueError, match="required"):
        AnalysisService().analyze(cv_text, job_description)


def test_generates_cv_based_questions():
    result = AnalysisService().analyze(
        "Developed applications using Python.",
        "Experience with Python required."
    )

    questions = result["interview_questions"]

    assert any(
        q["category"] == "cv_based"
        and q["related_skill"] == "Python"
        for q in questions
    )


def test_generates_knowledge_gap_questions():
    result = AnalysisService().analyze(
        "Experienced Java developer.",
        "Experience with Docker required."
    )

    questions = result["interview_questions"]

    assert any(
        q["category"] == "knowledge_gap"
        and q["related_skill"] == "Docker"
        for q in questions
    )


def test_study_recommendations():
    result = AnalysisService().analyze(
        "Experienced Python developer.",
        "Requires Python and Kubernetes."
    )

    topics = {item["topic"] for item in result["study_topics"]}

    assert "Kubernetes" in topics
    assert "Python" not in topics


class FakeMatcher(Matcher):
    def match(self, cv_text, job_description):
        return (
            [
                {
                    "skill": "CustomSkill",
                    "requirement": "CustomSkill required",
                    "cv_evidence": "Used CustomSkill",
                }
            ],
            [],
        )


def test_supports_custom_matcher():
    service = AnalysisService(matcher=FakeMatcher())

    result = service.analyze(
        "Some CV content",
        "Some job description"
    )

    assert result["matching_skills"][0]["skill"] == "CustomSkill"


def test_analysis_metadata():
    result = AnalysisService().analyze(
        "Python developer",
        "Python required"
    )

    assert result["method"] == "keyword_baseline"
    assert "disclaimer" in result


def test_hybrid_analysis_preserves_keyword_evidence():
    original_evidence = (
        "Designed SQL queries for a reporting application."
    )

    unrelated_sentence = (
        "Supported incoming students and answered questions."
    )

    matcher = Mock()
    matcher.match.return_value = (
        [
            {
                "skill": "SQL",
                "requirement": "Experience with SQL",
                "cv_evidence": original_evidence,
            }
        ],
        [],
    )

    hybrid_matcher = Mock()
    hybrid_matcher.match.return_value.predicted_match = 1

    service = AnalysisService(
        matcher=matcher,
        hybrid_matcher=hybrid_matcher,
        use_ml=True,
    )

    result = service.analyze(
        original_evidence + "\n" + unrelated_sentence,
        "Experience with SQL",
        )

    assert len(result["matching_skills"]) == 1
    assert (
            result["matching_skills"][0]["cv_evidence"]
            == original_evidence
    )

    hybrid_matcher.match.assert_called_once_with(
        "Experience with SQL",
        original_evidence,
    )

