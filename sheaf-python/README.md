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

# Dendrogram: build from merge history, cut to k clusters
merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
dendro = cohera.Dendrogram(merges, n_items=4)
labels = dendro.cut_to_k(k=2)   # array([0, 0, 1, 1])

# kNN graph from embeddings -> community detection
import numpy as np
embeddings = np.random.randn(100, 64).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
edges = cohera.knn_graph(embeddings, k=10)
communities = cohera.leiden(edges, n_nodes=100)

# Clustering evaluation
nmi_score = cohera.nmi([0, 0, 1, 1], [0, 0, 1, 1])  # 1.0
ari_score = cohera.ari([0, 0, 1, 1], [0, 0, 1, 1])  # 1.0
```

## API

| Name | Description |
|------|-------------|
| `reconcile` | Reconcile base forecasts via OLS, WLS, or MinTrace |
| `simple_star_matrix` | Build summing matrix for a 2-level hierarchy |
| `summing_matrix_from_tree` | Build summing matrix from dendrogram merge history |
| `Dendrogram` | Hierarchical clustering dendrogram with cut operations |
| `Dendrogram.cut_at_distance` | Cut dendrogram at a distance threshold |
| `Dendrogram.cut_to_k` | Cut dendrogram to exactly k clusters |
| `knn_graph` | Build k-nearest-neighbor graph from embeddings (HNSW) |
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
