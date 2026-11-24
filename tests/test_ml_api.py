
from fastapi.testclient import TestClient

from app.main import app, get_hybrid_matcher
from app.ml.evidence_validator import (
    EvidenceResult,
    EvidenceStatus,
)
from app.ml.hybrid_matcher import HybridMatchResult


class FakeMatcher:
    def match(self, requirement, cv_evidence):
        return HybridMatchResult(
            similarity=0.75,
            evidence=EvidenceResult(
                status=EvidenceStatus.SUPPORTED,
                reason="No blocking evidence detected.",
            ),
            predicted_match=1,
            reason="Matched.",
        )


def test_ml_match_endpoint():
    app.dependency_overrides[get_hybrid_matcher] = (
        lambda: FakeMatcher()
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/api/v1/ml/match",
            json={
                "requirement": "Experience with Python",
                "cv_evidence": "Built Python APIs",
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert data["matched"] is True
        assert data["similarity"] == 0.75
        assert data["evidence_status"] == "supported"
    finally:
        app.dependency_overrides.clear()
