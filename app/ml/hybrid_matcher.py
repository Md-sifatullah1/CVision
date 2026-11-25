
from dataclasses import dataclass

from app.ml.evidence_validator import (
    EvidenceResult,
    EvidenceStatus,
    EvidenceValidator,
)
from app.ml.onnx_matcher import ONNXSemanticMatcher


@dataclass(frozen=True)
class HybridMatchResult:
    similarity: float
    evidence: EvidenceResult
    predicted_match: int
    reason: str


class HybridMatcher:
    """Combine semantic similarity with evidence validation."""

    def __init__(
            self,
            threshold: float = 0.20,
            semantic_matcher: ONNXSemanticMatcher | None = None,
            evidence_validator: EvidenceValidator | None = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold

        self.semantic_matcher = (
            semantic_matcher
            if semantic_matcher is not None
            else ONNXSemanticMatcher()
        )

        self.evidence_validator = (
            evidence_validator
            if evidence_validator is not None
            else EvidenceValidator()
        )

    def match(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> HybridMatchResult:
        if not requirement.strip() or not cv_evidence.strip():
            raise ValueError(
                "Requirement and CV evidence must be nonempty."
            )

        similarity = self.semantic_matcher.similarity(
            requirement,
            cv_evidence,
        )

        evidence = self.evidence_validator.validate(
            requirement,
            cv_evidence,
        )

        if evidence.status != EvidenceStatus.SUPPORTED:
            return HybridMatchResult(
                similarity=similarity,
                evidence=evidence,
                predicted_match=0,
                reason=evidence.reason,
            )

        if similarity < self.threshold:
            return HybridMatchResult(
                similarity=similarity,
                evidence=evidence,
                predicted_match=0,
                reason="Semantic similarity is below the threshold.",
            )

        return HybridMatchResult(
            similarity=similarity,
            evidence=evidence,
            predicted_match=1,
            reason="Semantic similarity meets the threshold and no blocking evidence was detected.",
        )
