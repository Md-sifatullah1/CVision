
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class TrainedTfidfMatcher:
    """TF-IDF matcher fitted on a fixed training corpus."""

    def __init__(self, threshold: float = 0.3):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

    def fit(self, texts: list[str]) -> None:
        """Learn vocabulary and IDF from training texts."""

        if not texts or not all(
                isinstance(text, str) and text.strip()
                for text in texts
        ):
            raise ValueError(
                "Training texts must be nonempty strings."
            )

        self.vectorizer.fit(texts)

    def similarity(
            self,
            requirement: str,
            cv_evidence: str,
    ) -> float:
        """Compare two texts using the fitted vectorizer."""

        check_is_fitted(self.vectorizer)

        if not requirement.strip() or not cv_evidence.strip():
            raise ValueError("Both texts must be nonempty.")

        vectors = self.vectorizer.transform(
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
        return int(
            self.similarity(requirement, cv_evidence)
            >= self.threshold
        )
