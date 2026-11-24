
import pytest

from app.ml.evidence_validator import (
    EvidenceStatus,
    EvidenceValidator,
)


@pytest.fixture
def validator():
    return EvidenceValidator()


@pytest.mark.parametrize(
    "requirement,evidence",
    [
        (
                "Experience with Docker",
                "Have never worked with Docker containers",
        ),
        (
                "Experience with Python",
                "Have not used Python professionally",
        ),
        (
                "Experience with Docker",
                "Have no practical Docker experience",
        ),
        (
                "Experience with Python",
                "Have never programmed in Python",
        ),
    ],
)
def test_negated_experience(
        validator, requirement, evidence
):
    result = validator.validate(requirement, evidence)

    assert result.status == EvidenceStatus.NOT_SUPPORTED


@pytest.mark.parametrize(
    "requirement,evidence",
    [
        (
                "Production machine learning experience",
                "Completed a beginner course in machine learning",
        ),
        (
                "Experience with CI/CD",
                "Only learned CI/CD concepts in lectures",
        ),
        (
                "Experience with Docker",
                "Read documentation about Docker",
        ),
        (
                "Experience with production APIs",
                "Completed a tutorial on REST API development",
        ),
        (
                "Experience with machine learning deployment",
                "Studied theoretical machine learning concepts",
        ),
        (
                "Experience with production ML",
                "Attended an introductory AI workshop",
        ),
        (
                "Experience with production ML",
                "Completed an introductory machine learning course",
        ),
    ],
)
def test_insufficient_experience(
        validator, requirement, evidence
):
    result = validator.validate(requirement, evidence)

    assert result.status == EvidenceStatus.INSUFFICIENT


def test_manual_testing_not_automated(validator):
    result = validator.validate(
        "Experience with automated testing",
        "Performed manual testing without automation",
    )

    assert result.status == EvidenceStatus.INSUFFICIENT


def test_valid_experience(validator):
    result = validator.validate(
        "Experience with Docker",
        "Built Docker images for deployment",
    )

    assert result.status == EvidenceStatus.SUPPORTED


def test_empty_input(validator):
    with pytest.raises(ValueError):
        validator.validate("", "Built Docker images")



def test_manual_testing_without_automation(validator):
    result = validator.validate(
        "Experience with automated testing",
        "Performed manual testing without automation",
    )

    assert result.status == EvidenceStatus.INSUFFICIENT


def test_manual_and_automated_testing(validator):
    result = validator.validate(
        "Experience with automated testing",
        "Performed both manual and automated testing",
    )

    assert result.status == EvidenceStatus.SUPPORTED
