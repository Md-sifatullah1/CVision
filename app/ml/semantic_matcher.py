
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticMatcher:
    """Semantic similarity using pretrained sentence embeddings."""

    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            threshold: float = 0.5,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def similarity(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> float:
        """Compute cosine similarity between two sentences."""

        if not requirement.strip() or not cv_evidence.strip():
            raise ValueError("Both texts must be nonempty.")

        embeddings = self.model.encode(
            [requirement, cv_evidence],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        score = np.dot(embeddings[0], embeddings[1])

        return float(score)

    def predict(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> int:
        score = self.similarity(
            requirement,
            cv_evidence,
        )

        return int(score >= self.threshold)
