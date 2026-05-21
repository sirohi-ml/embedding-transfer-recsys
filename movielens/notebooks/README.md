# MovieLens Notebooks

Run in this order:

| # | Notebook | Description |
|---|----------|-------------|
| 00 | DataPrep | Download, clean, temporal split |
| 01 | FeatureEngineering | 11 numerical + genre/language features |
| 02 | Baseline_LR/RF/XGB/ANN | Experiment 1 — OHE baseline |
| 03 | EmbeddingANN | Train genre + language embeddings |
| 04 | LR/RF/XGB_Embeddings | Experiment 2 — embedding features |
| 05 | ANN/LR/RF/XGB_Optimised | Experiment 3 — Optuna tuning |

Run 05_ANN_Optimised before other 05_ notebooks.
