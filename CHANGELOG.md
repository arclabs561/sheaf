# Changelog

## [Unreleased]

### Fixed

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
