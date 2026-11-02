# sheaf

[![crates.io](https://img.shields.io/crates/v/sheaf.svg)](https://crates.io/crates/sheaf)
[![Documentation](https://docs.rs/sheaf/badge.svg)](https://docs.rs/sheaf)

Sheaf and hierarchy algorithms.

Use it to make tree-structured predictions sum to their parents, put conformal
intervals around them, cluster points, detect graph communities, and build
cellular sheaf Laplacians.

## What it provides

- **Hierarchy + conformal** (default): hierarchical point-forecast
  reconciliation and split conformal intervals.
- **Sheaf Laplacians** (default): `CellularSheaf` and `LearnableSheaf` on graphs.
- **Clustering** (`cluster` feature): k-means, DBSCAN, GMM, hierarchical
  clustering.
- **Community detection** (`community` feature): Louvain with or without
  connectivity refinement, Leiden, and label propagation; kNN graph construction
  from embeddings with `knn-graph`. On Zachary's karate club both Louvain
  variants reach networkx's modularity range (0.395 to 0.420), checked by
  `tests/karate_modularity.rs`.
- **Metrics** (default): clustering evaluation (ARI, NMI, purity, and others).

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

`cargo run --example forecast_reconciliation` reconciles a product hierarchy
whose base forecasts disagree (excerpt):

```text
Base (incoherent):
         node   forecast
-------------------------
        Total     250.00
  Electronics     120.00
     Clothing      90.00
...
  Base coherence gap (Total vs leaf sum): 2.00e1 [INCOHERENT]

OLS reconciled:
         node   forecast
-------------------------
        Total     235.71
  Electronics     132.86
     Clothing     102.86
...
  OLS coherence gap (Total vs leaf sum): 0.00e0 [coherent]
```

## Custom distance metrics

K-means and DBSCAN accept a pluggable distance metric via `with_metric`. Built-in metrics
re-exported from `clump`: `Euclidean`, `SquaredEuclidean`, `CosineDistance`.
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

## Examples

[examples/README.md](examples/README.md) maps each runnable example to its task
and required features.

## Python

Python bindings are published on PyPI as `cohera` (`pip install cohera`); their
source is in [sheaf-python/](sheaf-python/).

## References

- Bodnar et al. (2022). "Neural Sheaf Diffusion."
- Hansen & Ghrist (2019). "Toward a Spectral Theory of Cellular Sheaves."
- Principato et al. (2024). "Conformal Prediction for Hierarchical Data."
- Qiu & Li (2015). "IT-Dendrogram: A New Member of the In-Tree (IT) Clustering Family."
- Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."

## License

MIT OR Apache-2.0
