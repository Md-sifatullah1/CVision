
import re
from dataclasses import dataclass
from enum import Enum


class EvidenceStatus(str, Enum):
    SUPPORTED = "supported"
    INSUFFICIENT = "insufficient"
    NOT_SUPPORTED = "not_supported"


@dataclass(frozen=True)
class EvidenceResult:
    status: EvidenceStatus
    reason: str


class EvidenceValidator:
    """A conservative, rule-based evidence classifier."""

    NEGATION_PATTERNS = (
        r"\b(?:have|has|had)\s+(?:never|not)\b",
        r"\b(?:have|has|had)\s+no\b",
        r"\bnever\s+(?:used|worked|programmed|configured)\b",
        r"\bno\s+(?:practical|hands-on)\s+experience\b",
    )

    LEARNING_PATTERNS = (
        r"\b(?:introductory|beginner|basic)\s+(?:course|workshop)\b",
        r"\b(?:completed|attended|studied)\s+(?:an?\s+)?"
        r"(?:introductory|beginner|basic)\b",
        r"\bonly\s+learned\b",
        r"\bread\s+documentation\b",
        r"\bcompleted\s+(?:an?\s+)?tutorial\b",
        r"\bstudied\s+theoretical\b",
    )

    MANUAL_TESTING_PATTERN = (
        r"\bmanual(?:ly)?\s+test(?:ed|ing)?\b"
    )

    AUTOMATION_REQUIREMENT_PATTERN = (
        r"\bautomat(?:ed|ion|e)\b"
    )

    def validate(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> EvidenceResult:
        if not requirement.strip() or not cv_evidence.strip():
            raise ValueError(
                "Requirement and CV evidence must be nonempty."
            )

        requirement = requirement.lower()
        evidence = cv_evidence.lower()

        if any(
                re.search(pattern, evidence)
                for pattern in self.NEGATION_PATTERNS
        ):
            return EvidenceResult(
                EvidenceStatus.NOT_SUPPORTED,
                "Evidence contains an explicit negation.",
            )

        if (
            re.search(
                self.AUTOMATION_REQUIREMENT_PATTERN,
                requirement,
            )
            and re.search(
                self.MANUAL_TESTING_PATTERN,
                evidence,
            )
            and (
                re.search(
                    r"\bwithout\s+automation\b",
                    evidence,
                )
                or not re.search(
            r"\bautomat(?:ed|ion|e)\b",
                    evidence,
                )
            )
        ):
            return EvidenceResult(
                EvidenceStatus.INSUFFICIENT,
                "Manual testing does not demonstrate automation.",
            )


        if any(
                re.search(pattern, evidence)
                for pattern in self.LEARNING_PATTERNS
        ):
            return EvidenceResult(
                EvidenceStatus.INSUFFICIENT,
                "Learning activity alone may not demonstrate practical experience.",
            )

        return EvidenceResult(
            EvidenceStatus.SUPPORTED,
            "No explicit evidence limitation detected.",
        )

