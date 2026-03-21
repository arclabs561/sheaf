# cohera

Python bindings for [sheaf](https://docs.rs/sheaf) (Rust).

Forecast reconciliation (OLS, WLS, MinTrace), hierarchical conformal prediction,
and community detection (Leiden, Louvain, Label Propagation).

## Install

```
pip install cohera
```

## Usage

### Reconciliation

```python
import cohera

# 2-level hierarchy: 1 root + 3 leaves
s = cohera.simple_star_matrix(3)

# Incoherent base forecasts
base = [12.0, 3.0, 4.0, 5.0]

# OLS reconciliation
reconciled = cohera.reconcile(s, base, method="ols")
# reconciled[0] == reconciled[1] + reconciled[2] + reconciled[3]
```

### Conformal prediction

```python
hc = cohera.HierarchicalConformal(s, method="ols")
hc.calibrate(y_calib, y_hat_calib, alpha=0.1)
lower, upper = hc.predict(y_hat)
```

### Community detection

```python
edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
communities = cohera.leiden(edges, n_nodes=3, resolution=1.0)
```

### Clustering metrics

```python
nmi = cohera.nmi([0, 0, 1, 1], [0, 0, 1, 1])  # 1.0
ari = cohera.ari([0, 0, 1, 1], [0, 0, 1, 1])  # 1.0
```

## License

MIT OR Apache-2.0
