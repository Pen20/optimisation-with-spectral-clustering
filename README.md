# Spectral Clustering Optimisation Dashboard

End-to-end pipeline that finds the optimal number of clusters using **Spectral Clustering**
with full graph construction parameter search, evaluated by the reduced asymmetric NMI.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

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

- Jerdee, M., Kirkley, A. & Newman, M. (2025). _Normalized mutual information is a biased measure for classification and community detection._ Nature Communications, 16, 11268.
- Vinh, N.X., Epps, J. & Bailey, J. (2010). _Information theoretic measures for clusterings comparison: Variants, properties, normalization and correction for chance._ JMLR, 11, 2837–2854.
- Vinh, N.X., Epps, J. & Bailey, J. (2009). _Information theoretic measures for clusterings comparison: Is a correction for chance necessary?_ ICML 2009.

## License

MIT
