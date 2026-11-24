
import re

from app.ml.matcher import KeywordMatcher, Matcher
from app.ml.hybrid_matcher import HybridMatcher


class AnalysisService:
    def __init__(
            self,
            matcher: Matcher | None = None,
            hybrid_matcher: HybridMatcher | None = None,
            use_ml: bool = False,
    ):
        self.matcher = matcher if matcher is not None else KeywordMatcher()
        self.hybrid_matcher = hybrid_matcher
        self.use_ml = use_ml

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]


    def _verify_matches(
            self,
            matches: list[dict],
            gaps: list[dict],
            cv_text: str,
    ) -> tuple[list[dict], list[dict]]:
        if self.hybrid_matcher is None:
            raise RuntimeError("Hybrid matcher is not configured.")

        verified_matches = []
        verified_gaps = list(gaps)

        for match in matches:
            requirement = match["requirement"]
            evidence = match["cv_evidence"]

            # Verify the evidence identified by KeywordMatcher.
            # Do not search for unrelated CV sentences.
            result = self.hybrid_matcher.match(
                requirement,
                evidence,
            )

            if result.predicted_match == 1:
                verified_matches.append(match)
            else:
                verified_gaps.append({
                    "skill": match["skill"],
                    "requirement": requirement,
                    "status": "not_evidenced",
                })

        return verified_matches, verified_gaps


    def analyze(
            self,
            cv_text: str,
            job_description: str,
    ) -> dict:
        if not cv_text.strip() or not job_description.strip():
            raise ValueError(
                "CV text and job description are required."
            )

        matches, gaps = self.matcher.match(
            cv_text,
            job_description,
        )

        if self.use_ml:
            matches, gaps = self._verify_matches(
                matches,
                gaps,
                cv_text,
            )

        questions = []

        for match in matches[:5]:
            questions.append({
                "question": (
                    f"Can you describe a project where you used "
                    f"{match['skill']}? What challenges did you face?"
                ),
                "category": "cv_based",
                "related_skill": match["skill"],
            })

        for gap in gaps[:5]:
            questions.append({
                "question": (
                    f"How would you approach a task requiring "
                    f"{gap['skill']}?"
                ),
                "category": "knowledge_gap",
                "related_skill": gap["skill"],
            })

        study_topics = [
            {
                "topic": gap["skill"],
                "reason": (
                    "Mentioned in the job description but "
                    "not evidenced in the CV."
                ),
            }
            for gap in gaps
        ]

        return {
            "matching_skills": matches,
            "missing_requirements": gaps,
            "interview_questions": questions,
            "study_topics": study_topics,
            "method": (
                "hybrid_semantic_evidence"
                if self.use_ml
                else "keyword_baseline"
            ),
            "disclaimer": (
                "Results are automated estimates. "
                "Missing evidence does not prove a candidate "
                "lacks a skill. Review results manually."
            ),
        }
