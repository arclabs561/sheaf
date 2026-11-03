//! Regression tests for hierarchical clustering input checks, weighted
//! community detection, orthogonal restriction maps, tree validation and
//! sliced Wasserstein.
#![allow(clippy::unwrap_used)]

use sheaf::{LearnableSheaf, RestrictionFamily};

// n = 0: `n * (n - 1) / 2` underflowed (a debug-build panic) before the
// documented `EmptyInput` check ran.
#[cfg(feature = "cluster")]
#[test]
fn from_condensed_empty_is_error_not_panic() {
    use sheaf::cluster::HierarchicalClustering;
    let r = std::panic::catch_unwind(|| {
        HierarchicalClustering::new(1)
            .fit_dendrogram_from_condensed(vec![], 0)
            .is_err()
    });
    assert_eq!(
        r.ok(),
        Some(true),
        "fit_dendrogram_from_condensed(vec![], 0) panicked"
    );
}

/// Two triangles joined by a bridge, given either as 5 parallel unit edges or
/// as one edge of weight 5. Modularity uses A_ij = total weight between i and
/// j, so both graphs must get the same partition. networkx 3.7
/// `louvain_communities` sums multigraph weights the same way and returns
/// {0,1}, {2,3}, {4,5} for both.
#[cfg(feature = "community")]
fn bridged_triangles(parallel: bool) -> petgraph::graph::UnGraph<(), f32> {
    let mut g = petgraph::graph::UnGraph::<(), f32>::new_undirected();
    let v: Vec<_> = (0..6).map(|_| g.add_node(())).collect();
    for &(a, b) in &[(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)] {
        let _ = g.add_edge(v[a], v[b], 1.0);
    }
    if parallel {
        for _ in 0..5 {
            let _ = g.add_edge(v[2], v[3], 1.0);
        }
    } else {
        let _ = g.add_edge(v[2], v[3], 5.0);
    }
    g
}

#[cfg(feature = "community")]
#[test]
fn leiden_weighted_parallel_edges_sum() {
    use sheaf::LouvainConnected;
    let det = LouvainConnected::new();
    let a = det.detect_weighted(&bridged_triangles(true)).unwrap();
    let b = det.detect_weighted(&bridged_triangles(false)).unwrap();
    assert_eq!(a, b, "parallel bridge {a:?} vs summed bridge {b:?}");
}

#[cfg(feature = "community")]
#[test]
fn louvain_weighted_parallel_edges_sum() {
    use sheaf::community::Louvain;
    let det = Louvain::new();
    let a = det.detect_weighted(&bridged_triangles(true)).unwrap();
    let b = det.detect_weighted(&bridged_triangles(false)).unwrap();
    assert_eq!(a, b, "parallel bridge {a:?} vs summed bridge {b:?}");
}

/// Column-major d x d map for the source endpoint of edge 0, with the first
/// skew-symmetric parameter set to `leading` (A[0,1] = leading,
/// A[1,0] = -leading) and all others zero.
fn src_map(d: usize, leading: f64) -> Vec<f64> {
    let mut ls = LearnableSheaf::new(2, d, vec![(0, 1)], RestrictionFamily::Orthogonal).unwrap();
    ls.init_identity();
    ls.params_mut()[0] = leading;
    ls.build_maps()[0].0.clone()
}

/// exp(A) for A[0,1] = θ, A[1,0] = -θ is the plane rotation
/// [[cos θ, sin θ], [-sin θ, cos θ]] in coordinates (0, 1) and the identity
/// elsewhere (scipy.linalg.expm agrees). The d >= 4 path documents a
/// Padé(6,6) approximant with scaling to norm <= 0.5, accurate to f64
/// rounding.
#[test]
fn orthogonal_map_d4_matches_exact_matrix_exponential() {
    let theta = 1.0f64;
    let d = 4;
    let m = src_map(d, theta);
    let at = |r: usize, c: usize| m[c * d + r];
    let (s, c) = theta.sin_cos();
    let mut worst = 0.0f64;
    for r in 0..d {
        for col in 0..d {
            let want = match (r, col) {
                (0, 0) | (1, 1) => c,
                (0, 1) => s,
                (1, 0) => -s,
                _ if r == col => 1.0,
                _ => 0.0,
            };
            worst = worst.max((at(r, col) - want).abs());
        }
    }
    assert!(worst < 1e-12, "max |F - exp(A)| = {worst:e}");
}

/// The same parameter must give the same rotation in the (0, 1) plane whether
/// the stalk is 2-d (closed form) or 3-d (Rodrigues): F = exp(A) with the
/// parameters filling the upper triangle of A in both cases.
#[test]
fn orthogonal_map_parameter_convention_is_dimension_independent() {
    let theta = 0.7f64;
    let m2 = src_map(2, theta);
    let m3 = src_map(3, theta);
    for r in 0..2 {
        for c in 0..2 {
            assert!(
                (m2[c * 2 + r] - m3[c * 3 + r]).abs() < 1e-12,
                "d=2 map {m2:?} differs from the d=3 (0,1) block {m3:?} at ({r}, {c})"
            );
        }
    }
}

// stalk_dim = 0: `d * (d - 1) / 2` underflowed (a debug-build panic).
#[test]
fn learnable_orthogonal_zero_stalk_dim_does_not_panic() {
    let r = std::panic::catch_unwind(|| {
        LearnableSheaf::new(2, 0, vec![(0, 1)], RestrictionFamily::Orthogonal).map(|_| ())
    });
    assert!(
        r.is_ok(),
        "LearnableSheaf::new(stalk_dim = 0, Orthogonal) panicked"
    );
}

/// Node 2 is listed as a child of both 0 and 1, so the structure is not a
/// tree. `parents` can hold only one parent per node, so the check has to come
/// from the children lists.
#[test]
fn validate_reports_node_with_two_parents() {
    use sheaf::hierarchy::validate_tree_structure;
    use std::collections::HashMap;
    let children: HashMap<usize, Vec<usize>> =
        HashMap::from([(3, vec![0, 1]), (0, vec![2]), (1, vec![2])]);
    let parents: HashMap<usize, usize> = HashMap::from([(0, 3), (1, 3), (2, 0)]);
    let r = validate_tree_structure(&parents, &children, 4);
    assert!(
        !r.is_clean(),
        "node 2 has two parents but the report is clean: {r}"
    );
}

/// The module and field docs say sliced W₁. In 1-d every projection is ±1, so
/// SW_p equals W_p: x = {0,0,0,0}, y = {0,0,0,4} gives W₁ = 1 and W₂ = 2
/// (POT 0.9.7 `sliced_wasserstein_distance` returns 1.0 for p = 1, 2.0 for
/// p = 2).
#[cfg(feature = "wass")]
#[test]
fn sliced_wasserstein_is_w1_as_documented() {
    use ndarray::array;
    use sheaf::distribution_distance::{DistributionDistance, DistributionDistanceConfig};
    let x = array![[0.0f32], [0.0], [0.0], [0.0]];
    let y = array![[0.0f32], [0.0], [0.0], [4.0]];
    let d =
        DistributionDistance::compute(x.view(), y.view(), &DistributionDistanceConfig::default())
            .unwrap();
    let sw = d.sliced_wasserstein.unwrap();
    assert!((sw - 1.0).abs() < 1e-4, "sliced_wasserstein = {sw}, W1 = 1");
}
