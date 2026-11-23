from app.ml.matcher import KeywordMatcher, Matcher


class AnalysisService:
    def __init__(self, matcher: Matcher | None = None):
        self.matcher = matcher or KeywordMatcher()

    def analyze(self, cv_text: str, job_description: str) -> dict:
        if not cv_text.strip() or not job_description.strip():
            raise ValueError('CV text and job description are required.')
        matches, gaps = self.matcher.match(cv_text, job_description)
        questions = []
        for match in matches[:5]:
            questions.append({
                'question': f'Can you describe a project where you used {match["skill"]}? What challenges did you face?',
                'category': 'cv_based', 'related_skill': match['skill']
            })
        for gap in gaps[:5]:
            questions.append({
                'question': f'How would you approach a task requiring {gap["skill"]}?',
                'category': 'knowledge_gap', 'related_skill': gap['skill']
            })
        study_topics = [
            {'topic': gap['skill'], 'reason': 'Mentioned in the job description but not evidenced in the CV.'}
            for gap in gaps
        ]
        return {
            'matching_skills': matches,
            'missing_requirements': gaps,
            'interview_questions': questions,
            'study_topics': study_topics,
            'method': 'keyword_baseline',
            'disclaimer': 'Not found in CV does not mean the candidate lacks the skill. Review results manually.'
        }
