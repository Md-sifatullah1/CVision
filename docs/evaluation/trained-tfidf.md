# Trained TF-IDF Evaluation

## Objective

Evaluate a reusable TF-IDF text matcher for determining whether CV evidence satisfies a job requirement.

## Dataset

- Total examples: 50 synthetic labeled pairs
- Training: 30 examples
- Validation: 10 examples
- Held-out test: 10 examples

## Method

The TF-IDF vectorizer was fitted using training texts only.

Validation examples were used to select a cosine similarity threshold from 0.05 to 0.30, based on F1 score.

All candidate thresholds produced a validation F1 score of 0.000. The first threshold, 0.05, was selected by the tie-breaking behavior of the implementation.

## Held-out Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.400 |
| Precision | 0.400 |
| Recall | 0.400 |
| F1 score | 0.400 |

Confusion matrix:

- True positives: 2
- False positives: 3
- True negatives: 2
- False negatives: 3

## Error Analysis

The model frequently failed to recognize semantically related technologies and concepts when the text contained little or no shared vocabulary.

Negated statements and descriptions of introductory knowledge produced false positives because lexical similarity does not establish whether a requirement is satisfied.

## Limitations

The dataset is small, synthetic, and contains overlapping terminology across splits.

The results are useful for comparing experimental implementations but do not establish real-world generalization performance.

## Conclusion

The trained TF-IDF model did not reliably distinguish genuine skill evidence from lexical similarity.

The next experiment will investigate pretrained sentence embeddings while preserving the same evaluation splits.
