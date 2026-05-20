# Spectral Clustering Optimisation Dashboard

End-to-end pipeline that finds the optimal number of clusters using **Spectral Clustering**
with full graph construction parameter search, evaluated by the reduced asymmetric NMI.

## Key results

| Configuration            | k     | NMI (reduced asym.) | Graph                 | Notes            |
| ------------------------ | ----- | ------------------- | --------------------- | ---------------- |
| **PCA(10), std, NN=10**  | **4** | **0.354**           | **nearest_neighbors** | **Best overall** |
| PCA(10), std, NN=5, gw=3 | 3     | 0.336               | nearest_neighbors     | Best for k=3     |
| PCA(10), std, NN=5, gw=3 | 5     | 0.342               | nearest_neighbors     | Best for k=5     |

# Create and activate a virtual environment

```
python -m venv venvSpec
.\venvSpec\Scripts\Activate.ps1   # Windows PowerShell
```

## Quick start

```bash
pip install -r requirements.txt
python run_pipeline.py --data data.csv --output dashboard.html
open dashboard.html
```

## What's searched

1,872 configurations: 6 PCA dims × 2 standardisation × 4 grade weights × 13 graph params × 3 k values.

Graph parameters include:

- **RBF kernel**: γ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5}
- **k-NN graph**: n_neighbors ∈ {5, 7, 10, 15, 20, 30}
- All use `assign_labels='discretize'`

## References

- Jerdee et al. (2025) _Nature Communications_ 16
- Vinh et al. (2010) _JMLR_ 1
