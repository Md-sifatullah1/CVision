
# Sentence Transformer Baseline Evaluation

## Model

- Model: sentence-transformers/all-MiniLM-L6-v2
- Library: sentence-transformers 6.3.0
- Method: normalized sentence embeddings and cosine similarity
- Classification threshold: selected using validation F1

## Dataset

- Total: 50 synthetic labeled examples
- Training: 30
- Validation: 10
- Test: 10

The pretrained model was not fine-tuned on the training split.

## Threshold Selection

Validation F1 was 0.714 for thresholds 0.20,
0.25, 0.30, and 0.35.

The evaluation script selected 0.20 as the
first threshold with the highest F1.

## Held-out Test Results

- Accuracy: 0.600
- Precision: 0.571
- Recall: 0.800
- F1: 0.667

Confusion matrix:

- TP: 4
- FP: 3
- TN: 2
- FN: 1

## Comparison With Trained TF-IDF

| Metric | TF-IDF | Sentence Transformer |
|---|---:|---:|
| Accuracy | 0.400 | 0.600 |
| Precision | 0.400 | 0.571 |
| Recall | 0.400 | 0.800 |
| F1 | 0.400 | 0.667 |

## Error Analysis

False negatives:
- ID 45: GitLab CI experience was not recognized
  as sufficient continuous integration experience.

False positives:
- ID 47: Negated Docker experience.
- ID 48: Beginner ML course versus production ML.
- ID 50: Manual testing versus automated testing.

## Limitations

- Only 10 synthetic test examples.
- No real-world CV evaluation.
- Semantic similarity does not establish skill proficiency.
- Negation and insufficient evidence are not handled.
- Threshold selection is sensitive to the small validation set.

## Conclusion

Sentence Transformers improved recall and F1
compared with the trained TF-IDF baseline on this
dataset.

The next experiment should evaluate an
evidence-aware matching pipeline.
