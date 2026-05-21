# Amazon Notebooks

Run in this order:

| # | Notebook | Description |
|---|----------|-------------|
| 00 | DataPrep | Download, clean, temporal split |
| 01 | FeatureEngineering | 6 LOO interaction features |
| 02 | Baseline_LR/RF/XGB/ANN | Experiment 1 — 6 feature baseline |
| 03 | EmbeddingANN | Train userId + productId embeddings |
| 04 | LR/RF/XGB_Embeddings | Experiment 2 — embedding features |
| 05 | ANN/LR/RF/XGB_Optimised | Experiment 3 — Optuna tuning |

Run 05_ANN_Optimised before other 05_ notebooks.
