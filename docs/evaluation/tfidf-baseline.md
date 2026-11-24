# TF-IDF Baseline Evaluation

## Configuration

- Dataset: `data/evaluation/skill_pairs.csv`
- Evaluation examples: 12 synthetic pairs
- Vectorizer: `TfidfVectorizer`
- Stop words: English
- N-grams: Unigrams and bigrams
- Vectorizer fitting: Independently per text pair
- Similarity: Cosine similarity
- Classification threshold: 0.30 (untuned)

## Results

| Metric | Keyword baseline | TF-IDF |
|---|---:|---:|
| Accuracy | 0.750 | 0.500 |
| Precision | 0.714 | 0.000 |
| Recall | 0.833 | 0.000 |
| F1 | 0.769 | 0.000 |

TF-IDF confusion matrix:

- True positives: 0
- False positives: 0
- True negatives: 6
- False negatives: 6

## Observations

All TF-IDF predictions were negative at the initial threshold of 0.30.

Several positive examples received zero similarity because the requirement and evidence used different terminology.

The highest observed similarity, 0.236, belonged to a negative example involving introductory versus production machine learning experience.

## Interpretation

This experiment does not demonstrate that TF-IDF is universally worse than keyword matching. It demonstrates that the current TF-IDF configuration performs worse on this small synthetic dataset.

The dataset is not large enough to support meaningful generalization claims.

## Next Steps

1. Expand the labeled dataset.
2. Establish separate development and held-out evaluation splits.
3. Implement a shared fitted TF-IDF vectorizer.
4. Tune thresholds using development data only.
5. Compare the resulting models on held-out examples.
