"""Replaceable matching interface and a transparent rule-based baseline.

This starter does NOT pretend to use trained ML. See README for the next ML milestone.
"""
from abc import ABC, abstractmethod
from app.core.skills import evidence_for, load_skills, mentions


class Matcher(ABC):
    @abstractmethod
    def match(self, cv_text: str, job_description: str) -> tuple[list[dict], list[dict]]:
        ...


class KeywordMatcher(Matcher):
    def match(self, cv_text: str, job_description: str) -> tuple[list[dict], list[dict]]:
        matches, gaps = [], []
        for skill, aliases in load_skills().items():
            requirement = evidence_for(job_description, aliases)
            if not requirement:
                continue
            evidence = evidence_for(cv_text, aliases)
            if evidence:
                matches.append({'skill': skill, 'requirement': requirement, 'cv_evidence': evidence})
            else:
                gaps.append({'skill': skill, 'requirement': requirement, 'status': 'not_found_in_cv'})
        return matches, gaps
