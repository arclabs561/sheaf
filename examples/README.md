# sheaf examples

Examples for hierarchy-constrained prediction, community detection, and sheaf
spectral methods.

## Running

```sh
# Default features:
cargo run -p sheaf --example forecast_reconciliation
cargo run -p sheaf --example hierarchical_conformal
cargo run -p sheaf --example raptor_retrieval
cargo run -p sheaf --example sheaf_spectral

# Requires cluster feature:
cargo run -p sheaf --example clustering_comparison --features cluster

# Requires knn-graph feature:
cargo run -p sheaf --example embedding_clustering --features knn-graph
```

Use `cargo test --examples` to compile the default example set. Use
`cargo test --examples --features cluster` and
`cargo test --examples --features knn-graph` for the feature-gated examples.

## Task map

| Goal | Example | Features | What to inspect |
|---|---|---|---|
| Fix incoherent forecasts | `forecast_reconciliation` | default | OLS and WLS reconciliation on a product hierarchy where parent forecasts initially disagree with leaf sums. |
| Calibrate coherent intervals | `hierarchical_conformal` | default | Split conformal intervals on a root-plus-leaves hierarchy, including empirical joint coverage. |
| Search a hierarchy of documents | `raptor_retrieval` | default | Leaf-level versus root-level retrieval over a synthetic RAPTOR tree. |
| Compare clustering algorithms | `clustering_comparison` | `cluster` | K-means, GMM, and Ward clustering on the same labeled point cloud, scored with ARI, NMI, and purity. |
| Cluster embeddings through a graph | `embedding_clustering` | `knn-graph` | HNSW kNN graph construction followed by connectivity-refined Louvain and a purity check. |
| Learn sheaf restrictions | `neural_sheaf_diffusion` | default | Finite-difference optimization of diagonal restriction maps to reduce sheaf Laplacian energy. |
| Compare sheaf and graph spectra | `sheaf_spectral` | default | Trivial, constant, and twisted sheaves, including the case where the sheaf Laplacian reproduces the graph Laplacian. |
| Cluster with sheaf eigenvectors | `sheaf_clustering` | default | K-means over sheaf Laplacian eigenvectors when bridge rotation maps encode inconsistency. |
| Inspect random sheaf spectra | `sheaf_rmt` | default | Random sheaf Laplacian eigenvalues against a semicircle-law diagnostic and spacing statistics. |

## Reading path

Start with `forecast_reconciliation` or `hierarchical_conformal` if your data
has parent-child constraints. Use `embedding_clustering` when your input is a
vector collection and you want graph communities. Use `sheaf_spectral` before
the other sheaf Laplacian examples, because it shows the graph-Laplacian
baseline first.

## Check held-out forecasts from two files

Use a total with two children. Each whitespace-separated row contains five
numbers in this order: base total, base left, base right, observed left,
observed right. The observed total is the sum of the two observed children.
Supply past forecasts for calibration and separate rows for evaluation.

```sh
printf '30 10 20 11 20\n30 10 20 10 20\n30 10 20 12 20\n' > calibration.txt
printf '30 10 20 11.2 18.8\n33 10 20 11 21\n' > test.txt
cargo run --example forecast_coverage -- calibration.txt test.txt 0.5
```

The first forecast already adds up. Its observed child errors are +1.2 and
−1.2. Each fits the calibrated radius of 1.4142, but the combined residual
length is 1.6971. The coordinate box accepts this observation and the
Euclidean ball excludes it. The second forecast starts with a total of 33
and children summing to 30. OLS moves it to `32, 11, 21`, exactly matching
the observation.

The last argument is the target miscoverage. If calibration is too small
for the requested level, the radius is infinite. The two hand-picked test
rows demonstrate membership, not population coverage.
