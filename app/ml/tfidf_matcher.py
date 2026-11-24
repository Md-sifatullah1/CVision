
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfMatcher:
    """TF-IDF similarity baseline for requirement/evidence pairs."""

    def __init__(self, threshold: float = 0.3):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold

    def similarity(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> float:
        """Calculate cosine similarity between two texts."""

        if not requirement.strip() or not cv_evidence.strip():
            raise ValueError("Both texts must be nonempty.")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        vectors = vectorizer.fit_transform(
            [requirement, cv_evidence]
        )

        score = cosine_similarity(
            vectors[0:1],
            vectors[1:2],
        )[0][0]

        return float(score)

    def predict(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> int:
        """Return 1 when similarity reaches the threshold."""

        score = self.similarity(
            requirement,
            cv_evidence,
        )

        return int(score >= self.threshold)
