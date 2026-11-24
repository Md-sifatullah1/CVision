
# CVision Keyword Baseline Evaluation

## Experiment

- Model: CVision v1 KeywordMatcher
- Dataset: data/evaluation/skill_pairs.csv
- Examples: 12 synthetic labeled pairs
- Positive examples: 6
- Negative examples: 6
- Evaluation unit: One job requirement and one CV evidence passage
- Prediction rule: Match if at least one recognized skill is found in both texts

## Results

| Metric | Value |
|---|---:|
| Accuracy | 0.750 |
| Precision | 0.714 |
| Recall | 0.833 |
| F1 | 0.769 |
| True positives | 5 |
| False positives | 2 |
| True negatives | 4 |
| False negatives | 1 |

## Error Analysis

### ID 3: False negative
Requirement: Experience with relational databases.

Evidence: Designed PostgreSQL tables and wrote SQL queries.

Cause: The skill dictionary does not establish the broader relationship between PostgreSQL/SQL and relational databases.

### ID 10: False positive
Requirement: Experience with Docker containerization.

Evidence: Have not used Docker in professional or personal projects.

Cause: Keyword matching does not understand negation.

### ID 11: False positive
Requirement: Experience with production machine learning.

Evidence: Completed an introductory machine learning course.

Cause: The matcher recognizes machine learning terminology but does not verify the required experience level.

## Limitations

The evaluation dataset is small, synthetic, and balanced.

The evaluation measures pair-level skill overlap, not full CV-to-job suitability.

The results are preliminary and should not be interpreted as real-world performance estimates.

## Reproduce

Run from the repository root:

```bash
python -m app.ml.baseline_evaluator
```

## Next Steps

1. Expand and review the labeled dataset.
2. Introduce development and held-out test splits.
3. Implement a TF-IDF similarity baseline.
4. Tune thresholds using development data only.
5. Compare models on the same held-out examples.
