# Changelog

## [Unreleased]

### Added

- `ItDendrogram::with_n_clusters(c)` cuts the longest in-tree edges until
  `c` trees remain (the `c - 1` longest when the tree has one root), so two
  well-separated blobs come back as two clusters. `fit_predict` returns
  `Error::InvalidClusterCount` for `c = 0` or `c > n`. Additive; the default
  behaviour is unchanged.

### Fixed

- `ItDendrogram` docs now describe what `fit_predict` does. They listed a
  step that prunes low-density bridges, but no edges were ever removed:
  without `with_n_clusters` the result has one cluster per density maximum,
  usually one cluster in total.

- `Dendrogram::cut_to_k` now applies exactly n − k merges. It cut at the
  (n − k)th merge height and applied every merge at or below it, so tied
  heights (for example duplicate points) gave fewer than k clusters:
  single-linkage `HierarchicalClustering` on {0, 1, 10, 11} with k = 3
  returned 2.
- `LabelPropagation` with a fixed seed now returns the same partition on
  every run. It drew tie candidates from a `HashMap`, whose iteration order
  changes per process.
- `knn_graph_from_embeddings` with `symmetric: true` now adds one edge per
  mutual neighbour pair instead of two parallel edges, which changed the
  graph passed on to community detection.
- `SpectralClustering::fit` with `AffinityType::Precomputed` now returns
  `Error::InvalidParameter` instead of panicking; precomputed matrices go
  through `fit_affinity`.
- `ItDendrogram` with `k = 0` now returns `Error::InvalidParameter` instead
  of panicking.
- `Gmm` now draws its initial means from distinct data points. Drawing with
  replacement could start two components on one point; they stayed identical
  and every point got one label (8 of 20 seeds on two well-separated pairs).
- `ConcatSummarizer` now truncates at a UTF-8 character boundary instead of
  panicking when `max_len` falls inside a multi-byte character.

## [0.1.12] - 2026-10-30

### Fixed

- `SpectralClustering::fit` and `fit_affinity` above 64 points took their
  eigenvectors from `lapl` 0.2's 50 unchecked orthogonal-iteration steps
  (or, with `spectral-faer` above 512 points, a single-start-vector Krylov
  solver that misses repeated eigenvalues). Two disconnected chains split by
  position instead of by component, and kNN moons and circles at n = 200
  scored ARI 0.007 and 0.015 against scikit-learn's 1.000. Those sizes now
  use a dense `faer` eigendecomposition in sheaf, so the result no longer
  depends on the `spectral-faer` feature; graphs of 64 points or fewer still
  use `lapl`'s exact Jacobi path.
- `KmeansElkan` now compares later centres against a point's current centre
  after reassigning it within a pass (Elkan 2003, step 3). It could move a
  point to a farther centre, so its labels differed from Lloyd's from the same
  initial centres.
- `RaptorTree::build` now builds levels above the first. Level 2 clusters
  contain summary nodes, which were skipped as having no leaf content, so the
  tree stopped after one summary level. Each cluster is now summarized from
  the leaf items under its members, and `leaf_count` counts those leaves.
- Louvain and connected Louvain now reach networkx's modularity range on the
  karate club graph (0.415 to 0.420, 4 communities, was 0.335 with 7) by
  requiring graphops 0.5.2, which stopped counting a super-node's self-loop
  as a link to its own community. The karate reference tests run again.

### Documentation

- State in the README that Louvain reaches modularity 0.335 on the karate club
  graph, below networkx's 0.395 to 0.420, as the ignored reference test shows.
  Superseded by the fix above.
- State the symmetry and finite, nonnegative weight preconditions for
  `SpectralClustering::fit_affinity`. Runtime behavior is unchanged.
- Keep related-crate names readable when optional dependencies are disabled,
  fixing strict documentation builds without the `community` feature.

### Fixed

- Gate the distribution-distance `Array2` import on its `wass` or test uses,
  so spectral builds without `wass` compile without an unused-import warning.

### Maintenance

- Pin the CI semver checker to 0.51.0 so it can parse stable Rust's
  rustdoc JSON format while retaining the API comparison.
- Execute all-feature tests, doctests and strict documentation builds in CI.

## [0.1.11] - 2026-10-09

### Added

- `LouvainConnected`, the accurate name for the community detector formerly
  called `Leiden`. It runs `graphops`'s connected-Louvain routine, whose
  partitions can differ from true Leiden. `Leiden` remains as
  `pub type Leiden = LouvainConnected` and gives the same partitions.

### Changed

- Removed the optional `drawset` dependency, which no code used. The
  `drawset` and `kuji` features remain as empty compatibility aliases. This
  replaces the unreleased change that required drawset 0.1.2.
- The optional `graphops` dependency (enabled by `community`) moves from 0.2
  to 0.5.

### Fixed

- `HierarchicalConformal::calibrate` returns an infinite quantile (unbounded
  intervals) when the conformal rank `ceil((n + 1)(1 - alpha))` exceeds the
  calibration size `n`, instead of clamping to the largest score and
  under-covering. `alpha` outside `(0, 1)` now returns
  `Error::InvalidParameter`, and an empty calibration set returns
  `Error::EmptyInput`; `alpha = 1` and an empty set previously underflowed
  an index.
- The kNN-graph sigma estimate and the GMM component argmax compare with
  `total_cmp`, so NaN values no longer reach an inconsistent comparator.

## [0.1.10] - 2026-07-07

### Changed

- Switched optional stochastic sampling support from `kuji` to `drawset`;
  kept the `kuji` feature as a compatibility alias.

## [0.1.9] - 2026-07-05

### Changed

- Updated the optional `knn-graph` path for current `vicinity` behavior.
- Updated spectral example dependencies to current `lapl` and `rmt` releases.

## [0.1.8] - 2026-06-10

### Changed
- CI: nextest, feature fan-in gate, concurrency cancel. README and CONTRIBUTING polish. No code changes.

Earlier releases predate this changelog; see git history.
