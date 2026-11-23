from dataclasses import dataclass

@dataclass(frozen=True)
class Match:
    skill: str
    requirement: str
    cv_evidence: str

@dataclass(frozen=True)
class Gap:
    skill: str
    requirement: str
    status: str = 'not_found_in_cv'
