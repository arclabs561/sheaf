# sheaf

[![crates.io](https://img.shields.io/crates/v/sheaf.svg)](https://crates.io/crates/sheaf)
[![Documentation](https://docs.rs/sheaf/badge.svg)](https://docs.rs/sheaf)
[![CI](https://github.com/arclabs561/sheaf/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/sheaf/actions/workflows/ci.yml)

Partition, reconcile, predict. Clustering, community detection, and hierarchical conformal prediction in Rust.

## Problem

Given a set of points or a graph, you want to find groups: k-means for embeddings, Leiden for networks, hierarchical clustering for dendrograms. When predictions have hierarchical structure (e.g., forecasts at country/region/city levels), they should be coherent -- the parts should sum to the whole. Conformal prediction can provide calibrated intervals that respect this structure.

This library provides the algorithms. It is domain-agnostic; the stable contract is in [CONTRACT.md](CONTRACT.md).

## Examples

**Embeddings to communities**. Build a kNN graph from 2D points and detect clusters via Leiden:

```bash
cargo run --example embedding_clustering --features knn-graph
```

**Hierarchical conformal prediction**. Given a tree of predictions, reconcile them so they are structurally coherent, then produce calibrated prediction intervals:

```bash
cargo run --example hierarchical_conformal
```

Used by [`flowmatch`](https://github.com/arclabs561/flowmatch) (behind `--features sheaf-evals`) to evaluate whether generated samples preserve the cluster structure of real data -- for example, whether generated earthquake locations form the same geographic clusters as the USGS catalog.

## What it provides

- **Clustering**: k-means, DBSCAN, hierarchical clustering.
- **Community detection**: kNN graph construction (feature-gated), Leiden/Louvain/label propagation.
- **Hierarchy + conformal**: hierarchical reconciliation, split conformal prediction with coherence guarantees.
- **Metrics**: clustering evaluation helpers (used by `flowmatch` sheaf-eval examples).

## Custom distance metrics

K-means and DBSCAN accept a pluggable distance metric via `with_metric`. Built-in metrics
re-exported from `clump`: `Euclidean`, `SquaredEuclidean`, `CosineDistance`, `InnerProductDistance`.
Implement `DistanceMetric` for your own.

```rust
use sheaf::{Kmeans, CosineDistance};

let km = Kmeans::with_metric(8, CosineDistance)
    .with_seed(42)
    .with_seeding_alpha(2.0); // oversampling factor for k-means++
let result = km.fit(&data)?;
```

```rust
use sheaf::cluster::{Dbscan, CosineDistance};

// epsilon is compared against cosine distance (range [0, 2])
let db = Dbscan::with_metric(0.3, 5, CosineDistance);
let labels = db.fit(&data);
```

## Usage

```toml
[dependencies]
sheaf = "0.1.1"
```

```rust
use sheaf::{HierarchicalConformal, HierarchyTree, ReconciliationMethod};

// Build hierarchy, get summing matrix
let h_tree = HierarchyTree::from_raptor(&tree);
let s = h_tree.summing_matrix();

// Calibrate on held-out data
let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);
cp.calibrate(&y_calib, &y_hat_calib, 0.1)?; // 90% coverage

// Coherent prediction intervals
let (lower, upper) = cp.predict_intervals(&y_hat_test)?;
```

## References

- Principato et al. (2024). "Conformal Prediction for Hierarchical Data."
- Qiu & Li (2015). "IT-Dendrogram: A new representation for hierarchical clustering."
- Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."

## License

MIT OR Apache-2.0
