# parti examples

Examples for the `parti` clustering and hierarchical structure crate.

## Running

```sh
# Default features (hierarchy only):
cargo run -p parti --example forecast_reconciliation
cargo run -p parti --example hierarchical_conformal

# Requires cluster feature:
cargo run -p parti --example clustering_comparison --features cluster

# Requires knn-graph feature:
cargo run -p parti --example embedding_clustering --features knn-graph
```

## Examples

| Example | Features | Description |
|---|---|---|
| `forecast_reconciliation` | default | Hierarchical forecast reconciliation on a product hierarchy (Total -> Electronics/Clothing -> 4 leaves). Generates incoherent base forecasts, applies OLS and WLS reconciliation, prints before/after tables, and verifies that reconciled forecasts satisfy summing constraints. |
| `hierarchical_conformal` | default | Split conformal prediction on a 2-level hierarchy (1 root + 3 leaves). Calibrates on synthetic data with systematic bias, produces reconciled prediction intervals, and verifies coherence (root = sum of leaves) and empirical coverage. |
| `clustering_comparison` | `cluster` | Generates 300 points in 4 clusters (5D) with noise, runs K-means, GMM, and hierarchical (Ward) clustering, and compares them using ARI, NMI, and purity. Also shows GMM soft assignment entropy. |
| `embedding_clustering` | `knn-graph` | Generates 250 points in 4 Gaussian clusters (8D) + noise, builds a kNN graph via HNSW, and runs Leiden community detection. Prints cluster sizes and purity. |
| `raptor_retrieval` | default | Builds a RAPTOR tree from 20 synthetic documents (4 topics, bag-of-words vectors), clusters by cosine similarity, and searches at leaf vs. root level to show narrow vs. broad retrieval. |
