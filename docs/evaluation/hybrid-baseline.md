
# Hybrid Skill Matching Evaluation

## Approach

Combines:
1. Sentence Transformer semantic similarity
2. Rule-based evidence validation

A requirement is classified as matched when:
- Semantic similarity meets the threshold.
- No blocking evidence pattern is detected.

## Model Configuration

- Embedding model: all-MiniLM-L6-v2
- Similarity threshold: 0.20
- Evidence validation: negation and insufficient-evidence rules

## Evaluation Dataset

- 50 synthetic examples
- 30 training examples
- 10 validation examples
- 10 test examples

The hybrid evaluation uses the 10 test examples.

## Results

| Metric | Value |
|---|---:|
| Accuracy | 0.900 |
| Precision | 1.000 |
| Recall | 0.800 |
| F1 | 0.889 |

Confusion matrix:
- TP: 4
- FP: 0
- TN: 5
- FN: 1

## Error Analysis

False negative:
- ID 45: GitLab CI pipelines were not recognized
  as sufficient continuous integration experience.

The evidence validator correctly rejected:
- ID 47: Negated Docker experience
- ID 48: Beginner machine learning course
- ID 50: Manual testing without automation

## Limitations

- The dataset contains synthetic examples.
- The test set was inspected before hybrid rules were developed.
- Rules were informed by observed test errors.
- The reported performance is not an unbiased
  estimate of real-world generalization.
- Regex rules may reject valid evidence or miss
  complex negation.
- The current pipeline does not measure proficiency.

## Next Steps

1. Add independent, unseen evaluation examples.
2. Test more complex negation and evidence patterns.
3. Compare models using the same evaluation protocol.
4. Integrate the validated matching pipeline into FastAPI.
