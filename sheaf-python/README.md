# cohera

Hierarchical forecast reconciliation (OLS, WLS, MinTrace), conformal prediction intervals, community detection (Leiden, Louvain, Label Propagation), and clustering evaluation metrics. Backed by a Rust implementation.

Python bindings for the [sheaf](https://crates.io/crates/sheaf) Rust crate.

## Install

    pip install cohera

## Quick start

```python
import cohera

# Reconcile incoherent forecasts to satisfy a hierarchy
s = cohera.simple_star_matrix(3)  # 1 root + 3 leaves
base = [12.0, 3.0, 4.0, 5.0]
reconciled = cohera.reconcile(s, base, method="ols")
# reconciled[0] == reconciled[1] + reconciled[2] + reconciled[3]

# Community detection on a weighted graph
edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
communities = cohera.leiden(edges, n_nodes=3, resolution=1.0)

# Clustering evaluation
nmi_score = cohera.nmi([0, 0, 1, 1], [0, 0, 1, 1])  # 1.0
ari_score = cohera.ari([0, 0, 1, 1], [0, 0, 1, 1])  # 1.0
```

## API

| Name | Description |
|------|-------------|
| `reconcile` | Reconcile base forecasts via OLS, WLS, or MinTrace |
| `simple_star_matrix` | Build summing matrix for a 2-level hierarchy |
| `HierarchicalConformal` | Conformal predictor with hierarchical reconciliation |
| `HierarchicalConformal.calibrate` | Calibrate intervals from held-out data |
| `HierarchicalConformal.predict` | Produce reconciled (lower, upper) prediction intervals |
| `leiden` | Leiden community detection |
| `louvain` | Louvain community detection |
| `label_propagation` | Label propagation community detection |
| `nmi` | Normalized Mutual Information |
| `ari` | Adjusted Rand Index |
| `v_measure` | V-Measure (harmonic mean of homogeneity and completeness) |
| `purity` | Clustering purity |
| `homogeneity` | Each cluster contains only one class |
| `completeness` | All members of a class are in the same cluster |
| `fowlkes_mallows` | Fowlkes-Mallows Index |

## numpy support

All functions accept numpy arrays or Python lists. Reconciled forecasts and community labels are returned as numpy arrays.

## License

MIT OR Apache-2.0
