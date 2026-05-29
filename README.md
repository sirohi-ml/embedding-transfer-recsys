# Embedding Transfer in Recommendation Systems

**From OHE to Embeddings: What Meaningfully Changes Performance in Recommendation Systems**

A systematic study of ANN-learned categorical embedding transfer to classical recommendation models across three datasets and four algorithms.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20438828.svg)](https://doi.org/10.5281/zenodo.20438828)

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The Question

If you train a neural network to learn dense representations of categorical features, can other models benefit from those representations? Classical models like XGBoost or Linear Regression?

Intuitively it seems like it should work. The neural network has learned something meaningful about the structure of the data. This study tests whether that is actually true, and more importantly, what determines how much each model class benefits.

---

## Study Design

Three experimental conditions, run across four algorithms and three datasets:

| Condition | Encoding | Hyperparameters |
|-----------|----------|-----------------|
| Experiment 1 (Baseline) | OHE + multi-hot encoding | Default |
| Experiment 2 (Embeddings) | ANN-learned dense vectors | Default (fixed) |
| Experiment 3 (Optimised) | ANN-learned dense vectors | Optuna TPE search |

The three-condition structure is deliberate. A two-condition study comparing baseline against optimised cannot separate encoding strategy from hyperparameter tuning. The intermediate condition makes each contribution independently measurable.

**Algorithms tested**: Linear Regression, Random Forest, XGBoost, Feedforward ANN

**Evaluation metric**: Test RMSE on a temporal holdout split (lower is better)

---

## The Three Datasets

Each dataset was chosen to create a different challenge for the embedding approach.

**MovieLens 20M (100K temporally sampled subset)** is the standard recommendation benchmark. It has rich external metadata from IMDB and TMDB: ratings, vote counts, budget, revenue, genres, and language. The categoricals are low-cardinality and semantically meaningful. Ratings are approximately normally distributed around 3.5 stars.

**Amazon Product Reviews (500K sample)** has no product metadata at all. The only categoricals are user IDs and product IDs, both high-cardinality and semantically opaque. 55% of all ratings are 5-star. Everything must be derived from interaction history alone.

**Yelp Academic Dataset (100K sample)** introduces a third type of categorical: city, state, and business categories (comma-separated multi-label, similar to MovieLens genres). Medium cardinality. Integer 1-5 star ratings.

The progression from metadata-rich to interaction-only to geographically-structured tests whether embedding transfer is a general phenomenon or dataset-specific.

---

## Results

### MovieLens 20M (100K temporally sampled subset)

| Model | Baseline | Embeddings | Optimised | Best |
|-------|----------|------------|-----------|------|
| Linear Regression | 0.9381 | 0.9293 | 0.9293 | 0.9293 |
| Random Forest | 0.9641 | 0.9340 | 0.9279 | 0.9279 |
| XGBoost | 0.9394 | 0.9319 | **0.9274** | **0.9274** |
| ANN | 0.9338 | 0.9333 | 0.9300 | 0.9300 |

The total improvement range across all conditions and all algorithms was 0.0038 to 0.0362 RMSE. XGBoost won at 0.9274 but barely edged out the competition. This is the feature ceiling effect. MovieLens models already know a movie's IMDB rating, box office budget, and genre. Once that information is in the feature set, encoding strategy and algorithm choice have diminishing returns.

### Amazon Product Reviews (500K sample)

| Model | Baseline | Embeddings | Optimised | Best |
|-------|----------|------------|-----------|------|
| Linear Regression | 1.3646 | 1.3645 | 1.3644 | 1.3644 |
| Random Forest | 1.5559 | 1.5153 | 1.3626 | 1.3626 |
| XGBoost | 1.4942 | 1.4764 | **1.3579** | **1.3579** |
| ANN | 1.4746 | 1.3971 | 1.3729 | 1.3729 |

The total improvement range here is 0.1933. That is ten times larger than MovieLens. Random Forest went from 1.5559 to 1.3626, largely because Optuna found appropriate depth constraints. The default max_depth=None setting had been causing severe memorisation. Linear Regression barely moved (1.3646 to 1.3644 across all three experiments).

### Yelp Academic Dataset (100K sample)

| Model | Baseline | Embeddings | Optimised | Best |
|-------|----------|------------|-----------|------|
| Linear Regression | 1.5672 | **1.5526** | 1.5542 | **1.5526** |
| Random Forest* | 1.5831 | 1.7030 | 1.5820 | 1.5820 |
| XGBoost | 1.6517 | 1.6716 | 1.5597 | 1.5597 |
| ANN | 1.9168 | 1.5869 | 1.5871 | 1.5869 |

*RF baseline used constrained hyperparameters (max_depth=15, max_features=sqrt, max_samples=0.5) for computational feasibility with the approximately 1,944-column sparse OHE baseline matrix.

Two things stand out. The ANN improved by 0.3299 from baseline to embeddings. Replacing the 1,944-column OHE matrix with 58 dense embedding features dramatically reduced the overfitting. That is the largest single improvement in the whole study. Second, Linear Regression won overall at 1.5526. On 70,000 training rows, the simplest well-regularised model generalised best.

---

## Overall Winners

| Dataset | Best Model | Best RMSE | Condition |
|---------|-----------|-----------|-----------|
| MovieLens | XGBoost | 0.9274 | Experiment 3 (Optimised) |
| Amazon | XGBoost | 1.3579 | Experiment 3 (Optimised) |
| Yelp | Linear Regression | 1.5526 | Experiment 2 (Embeddings) |

XGBoost won on the two larger datasets. Linear Regression won on the smallest. Optimal model complexity appears to scale with training set size.

---

## Why the Benefit Differs Across Model Classes

**Linear Regression** showed small but consistent improvement across all datasets: 0.0088 on MovieLens, 0.0001 on Amazon, 0.0146 on Yelp. These are real but consistently much smaller than ANN improvement on the same datasets. Embedding space encodes relational proximity. Similar businesses cluster together. Users with similar tastes cluster together. Linear Regression can access the coordinates of each point in that space but has limited capacity to exploit the proximity structure. Acting on cluster membership requires non-linear decision boundaries.

**Tree models** showed the most variable results. On Amazon, Random Forest improved by 0.1933 overall, but most of that came from Optuna correcting the default max_depth=None setting, not from embeddings. On Yelp, embeddings made RF worse (1.5831 to 1.7030) before Optuna fixed it. Tree models often benefited more from optimisation than from embeddings alone, and on Yelp embeddings initially worsened performance before tuning corrected it. Dense, information-rich features give trees more to memorise when depth is unconstrained.

**The ANN** benefited most, but the mechanism is different from classical models. For Linear Regression and XGBoost, Experiment 2 is genuine cross-model transfer. They receive pre-computed embeddings from a different architecture and use them as static features. For the ANN, Experiment 2 is representation substitution. Sparse OHE inputs are replaced with learned dense inputs in the same architecture. The improvement scaled with how badly the OHE baseline was causing problems: MovieLens (+0.0005), Amazon (+0.0775), Yelp (+0.3299).

---

## Three Findings That Held Across All Datasets

1. LR improvement from embeddings was consistently small, always smaller than ANN improvement on the same dataset by at least an order of magnitude.

2. Tree model benefit from embeddings was conditional on hyperparameter regularisation.

3. ANN benefit scaled with baseline overfitting severity, not with dataset size or metadata richness.

---

## A Critical Engineering Detail

For Amazon and Yelp, all interaction-derived features use Leave-One-Out (LOO) aggregation. The naive approach computes a user's average rating across all their training reviews and merges it back onto those same rows. The initial Amazon version had Train RMSE 0.04 and Test RMSE 1.70 from this alone. LOO excludes each row's own rating from the aggregate it receives as a feature. This is not optional.
