# parti examples

Examples for the `parti` clustering and hierarchical structure crate.

## Running

```sh
# Default features (hierarchy only):
cargo run -p parti --example hierarchical_conformal

# Requires knn-graph feature:
cargo run -p parti --example embedding_clustering --features knn-graph
```

## Examples

| Example | Features | Description |
|---|---|---|
| `embedding_clustering` | `knn-graph` | Generates 250 points in 4 Gaussian clusters (8D) + noise, builds a kNN graph via HNSW, and runs Leiden community detection. Prints cluster sizes and purity. |
| `hierarchical_conformal` | default | Split conformal prediction on a 2-level hierarchy (1 root + 3 leaves). Calibrates on synthetic data with systematic bias, produces reconciled prediction intervals, and verifies coherence (root = sum of leaves) and empirical coverage. |
