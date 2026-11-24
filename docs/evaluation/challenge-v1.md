
# CVision Challenge Evaluation v1

## Configuration

- Model: all-MiniLM-L6-v2
- Matching approach: Semantic similarity + evidence rules
- Threshold: 0.20
- Dataset: data/evaluation/challenge_v1.csv
- Examples: 24 synthetic labeled pairs

## Results

| Metric | Value |
|---|---:|
| Accuracy | 0.708 |
| Precision | 0.667 |
| Recall | 0.833 |
| F1 | 0.741 |

Confusion matrix:

- TP: 10
- FP: 5
- TN: 7
- FN: 2

## Category Accuracy

| Category | Correct |
|---|---:|
| Semantic match | 4/4 |
| Exact match | 4/4 |
| Negation | 3/4 |
| Insufficient evidence | 2/4 |
| Mixed evidence | 2/4 |
| Unrelated | 2/4 |

## Failure Analysis

False positives:
- ID 10: Negated Python experience.
- ID 15: Beginner REST API tutorial.
- ID 16: Theoretical Docker knowledge.
- ID 23: MongoDB incorrectly matched PostgreSQL.
- ID 24: Figma incorrectly matched automated testing.

False negatives:
- ID 17: Negated Kubernetes experience incorrectly
  blocked positive Docker evidence.
- ID 19: Negated Java experience incorrectly
  blocked positive Python evidence.

## Conclusions

The hybrid matcher handles straightforward positive
evidence but struggles with:

1. Negation scope
2. Insufficient experience
3. Closely related but non-equivalent technologies
4. Low-threshold semantic false positives

This benchmark is a targeted synthetic challenge set,
not a representative real-world performance estimate.

The dataset must remain unchanged when developing
subsequent improvements.
