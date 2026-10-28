# sheaf

[![crates.io](https://img.shields.io/crates/v/sheaf.svg)](https://crates.io/crates/sheaf)
[![Documentation](https://docs.rs/sheaf/badge.svg)](https://docs.rs/sheaf)

Sheaf and hierarchy algorithms.

`sheaf` includes clustering, community detection, hierarchy reconciliation, and
conformal intervals for tree-structured predictions.

## Problem

Use it for clustering points, detecting graph communities, and making
tree-structured predictions sum to their parents.

## Examples

See [examples/README.md](examples/README.md) for the full runnable example map.

**Embeddings to communities**. Build a kNN graph from 2D points and detect clusters
with connectivity-refined Louvain:

```bash
cargo run --example embedding_clustering --features knn-graph
```

**Forecast reconciliation**. Given a tree of predictions, reconcile the point
forecasts to the hierarchy, then produce marginal conformal intervals:

```bash
cargo run --example forecast_reconciliation
```

The clustering-evaluation metrics check whether two point sets share the same cluster structure: for example, whether a generated set of earthquake locations forms the same geographic clusters as the USGS catalog.

## What it provides

- **Clustering**: k-means, DBSCAN, hierarchical clustering.
- **Community detection**: kNN graph construction (feature-gated), Louvain with or
  without connectivity refinement, and label propagation. On Zachary's karate
  club both Louvain variants reach networkx's modularity range (0.395 to 0.420),
  checked by `tests/karate_modularity.rs`.
- **Hierarchy + conformal**: hierarchical point-forecast reconciliation and split conformal intervals.
- **Metrics**: clustering evaluation helpers.

## Custom distance metrics

K-means and DBSCAN accept a pluggable distance metric via `with_metric`. Built-in metrics
re-exported from `clump`: `Euclidean`, `SquaredEuclidean`, `CosineDistance`, `InnerProductDistance`.
Implement `DistanceMetric` for your own. These need the `cluster` feature:

```toml
[dependencies]
sheaf = { version = "0.1", features = ["cluster"] }
```

```rust
use sheaf::cluster::Dbscan;
use sheaf::{Clustering, CosineDistance, Kmeans};

let data = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
let km = Kmeans::with_metric(2, CosineDistance)
    .with_seed(42)
    .with_seeding_alpha(2.0); // exponent on distance in k-means++ seeding
let labels = km.fit_predict(&data).unwrap();
assert_eq!(labels.len(), 3);

// epsilon is compared against cosine distance (range [0, 2])
let db = Dbscan::with_metric(0.3, 2, CosineDistance);
let labels = db.fit_predict(&data).unwrap();
assert_eq!(labels[0], labels[1]);
```

## Usage

```toml
[dependencies]
sheaf = "0.1"
faer = "0.19"
```

```rust
use faer::Mat;
use sheaf::{HierarchicalConformal, HierarchyTree, ReconciliationMethod};

// Three leaves: node 3 = leaves 0 + 1, node 4 (root) = node 3 + leaf 2.
let h_tree = HierarchyTree::from_merges(&[(0, 1, 1.0, 2), (3, 2, 2.0, 3)], 3);
let s = h_tree.summing_matrix();
let m = s.m();

// Calibrate on 19 held-out columns (truth and base forecasts, m x 19).
let y_calib = Mat::from_fn(m, 19, |i, j| (i + j) as f64);
let y_hat_calib = Mat::from_fn(m, 19, |i, j| (i + j) as f64 + 0.1 * ((j % 3) as f64 - 1.0));
let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);
cp.calibrate(&y_calib, &y_hat_calib, 0.1).unwrap(); // 90% coverage

// Marginal prediction intervals around reconciled forecasts
let y_hat_test = Mat::from_fn(m, 1, |i, _| i as f64);
let (lower, upper) = cp.predict_intervals(&y_hat_test).unwrap();
assert!(lower[(0, 0)] <= upper[(0, 0)]);
```

Python bindings are published on PyPI as `cohera` (`pip install cohera`); their
source is in [sheaf-python/](sheaf-python/).

## References

- Principato et al. (2024). "Conformal Prediction for Hierarchical Data."
- Qiu & Li (2015). "IT-Dendrogram: A New Member of the In-Tree (IT) Clustering Family."
- Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."

## License

MIT OR Apache-2.0
