# Setup Guide

Step-by-step instructions to reproduce all experiments from scratch.

---

## 1. Clone the Repository

```bash
git clone https://github.com/sirohi-ml/embedding-transfer-recsys.git
cd embedding-transfer-recsys
```

---

## 2. Install Dependencies

Python 3.10 or higher is required.

```bash
pip install -r requirements.txt
```

To also install nbstripout (recommended — keeps notebooks clean for git):

```bash
nbstripout --install
```

This sets up a pre-commit hook that automatically strips notebook outputs before every commit.

---

## 3. Download Datasets

### MovieLens 20M

```bash
# Option A — Direct download from GroupLens
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip -d movielens/data/raw/

# Option B — Kaggle
pip install kagglehub
python -c "import kagglehub; kagglehub.dataset_download('grouplens/movielens-20m-dataset')"
```

Also download IMDB and TMDB metadata from Kaggle:
- IMDB: `kaggle datasets download -d stefanoleone992/imdb-extensive-dataset`
- TMDB: `kaggle datasets download -d tmdb/tmdb-movie-metadata`

Place all files under `movielens/data/raw/`.

### Amazon Product Reviews

```python
import kagglehub
path = kagglehub.dataset_download("saurav9786/amazon-product-reviews")
print(f"Downloaded to: {path}")
```

Place the CSV file under `amazon/data/raw/`.

### Yelp Academic Dataset

```python
import kagglehub
path = kagglehub.dataset_download("yelp-dataset/yelp-dataset")
print(f"Downloaded to: {path}")
```

Place the JSON files under `yelp/data/raw/`.

---

## 4. Configure Paths

Each notebook has a `BASE_DIR` variable at the top of Step 1. 
Update it to match your local directory structure:

```python
# Example — update this in each notebook
BASE_DIR = r"C:\Your\Path\To\embedding-transfer-recsys\movielens"
```

---

## 5. Run Notebooks

Run notebooks **in order within each dataset**. Across datasets, any order works.

### Recommended order per dataset:

```
00_DataPrep.ipynb               ← download + clean + split
01_FeatureEngineering.ipynb     ← LOO features + categorical cleaning
02_Baseline_LR.ipynb            ← Experiment 1
02_Baseline_RF.ipynb
02_Baseline_XGB.ipynb
02_Baseline_ANN.ipynb
03_EmbeddingANN.ipynb           ← trains embeddings, saves weights
04_LR_Embeddings.ipynb          ← Experiment 2 (run after 03)
04_RF_Embeddings.ipynb
04_XGB_Embeddings.ipynb
05_ANN_Optimised.ipynb          ← Experiment 3 ANN (run first)
05_LR_Optimised.ipynb           ← Experiment 3 classical (run after ANN)
05_RF_Optimised.ipynb
05_XGB_Optimised.ipynb
```

**Critical**: `05_ANN_Optimised.ipynb` must run before the other `05_*` notebooks.
It saves optimised embedding weights to `optimization_ANN/` which the classical
models then load.

---

## 6. GPU Acceleration (Optional)

The EmbeddingANN and ANN notebooks will automatically use GPU if available:

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

No code changes are needed. Training times on CPU are approximately:
- MovieLens EmbeddingANN: ~60s
- Amazon EmbeddingANN: ~60s  
- Yelp EmbeddingANN: ~50s

---

## 7. Expected Runtimes (CPU)

| Notebook | Dataset | Approx. time |
|----------|---------|-------------|
| Data Prep | Any | 2-5 min |
| Feature Engineering | Any | 1-3 min |
| Baseline LR | Any | < 1 min |
| Baseline RF | MovieLens/Amazon | 2-15 min |
| Baseline RF | Yelp | 20 min (constrained params) |
| Baseline XGB | Any | 1-30 min |
| Baseline ANN | Any | 2-5 min |
| EmbeddingANN | Any | 1-3 min |
| Optimised ANN | Any | 15-30 min (50 Optuna trials) |
| Optimised XGB | Any | 10-20 min (50 Optuna trials) |
| Optimised RF | Any | 15-30 min (30 Optuna trials) |

---

## 8. Troubleshooting

**MemoryError on Yelp baseline RF**
The OHE feature matrix is ~1,944 columns. If RF runs out of memory, add:
```python
model = RandomForestRegressor(
    max_depth=15, max_features='sqrt', max_samples=0.5,
    random_state=42, n_jobs=-1
)
```

**CUDA error on EmbeddingANN**
Add index validation before training:
```python
assert user_idx.max() < n_users, f"User index OOB: max={user_idx.max()}"
assert prod_idx.max() < n_products, f"Product index OOB: max={prod_idx.max()}"
```

**user_avg_rating mean is ~1.5 instead of ~4.0**
LOO clip bug. Replace `.clip(lower=1)` with `np.where`:
```python
df['user_avg'] = np.where(
    (count - 1) > 0,
    (sum - rating) / (count - 1),
    np.nan
)
df['user_avg'] = df['user_avg'].fillna(global_avg)
```

---

## 9. Kaggle API Setup

If you are using kagglehub, you need a Kaggle API key:

1. Go to kaggle.com, click your profile, select "Account"
2. Scroll to "API" and click "Create New Token"
3. This downloads `kaggle.json`
4. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<user>\.kaggle\kaggle.json` (Windows)
5. Run `chmod 600 ~/.kaggle/kaggle.json` on Linux/Mac

The `kaggle.json` file is in `.gitignore` and will never be committed.
