# Yelp Notebooks

Run in this order:

| # | Notebook | Description |
|---|----------|-------------|
| 00 | DataPrep | Download, join reviews + business, temporal split |
| 01 | FeatureEngineering | 6 LOO features + city/state/categories |
| 02 | Baseline_LR/RF/XGB/ANN | Experiment 1 — OHE baseline (~1,944 cols) |
| 03 | EmbeddingANN | Train all 5 categorical embeddings |
| 04 | LR/RF/XGB_Embeddings | Experiment 2 — ~58 dense features |
| 05 | ANN/LR/RF/XGB_Optimised | Experiment 3 — Optuna tuning |

Note: RF baseline uses constrained params (max_depth=15) for speed.
Run 05_ANN_Optimised before other 05_ notebooks.
