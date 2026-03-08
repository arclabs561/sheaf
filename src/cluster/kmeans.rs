//! K-means clustering.
//!
//! Partitions data into k clusters by minimizing **within-cluster sum of squares**
//! (WCSS). The foundational clustering algorithm, dating to 1957 (Lloyd).
//!
//! # The Objective
//!
//! K-means minimizes:
//!
//! ```text
//! WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
//! ```
//!
//! Sum of squared distances from each point to its cluster centroid.
//!
//! # Lloyd's Algorithm
//!
//! 1. Initialize k centroids (randomly or via k-means++)
//! 2. **Assign**: Each point → nearest centroid
//! 3. **Update**: Each centroid → mean of assigned points
//! 4. Repeat until convergence
//!
//! **Why it converges**: WCSS decreases monotonically. Each step either
//! decreases WCSS or leaves it unchanged. Bounded below by 0 → must converge.
//!
//! # Failure Modes
//!
//! - **Local optima**: NP-hard problem; Lloyd finds local minimum only
//! - **Wrong k**: Must specify k in advance; use elbow method or silhouette
//! - **Non-spherical clusters**: Assumes roughly spherical, equal-sized clusters
//! - **Initialization sensitivity**: Bad initial centroids → bad results
//!
//! ## K-means++ Initialization
//!
//! Addresses initialization by spreading initial centroids:
//! 1. Choose first centroid uniformly at random
//! 2. Choose next centroid with probability proportional to D(x)²
//!    (squared distance to nearest existing centroid)
//!
//! Provides provable O(log k) approximation to optimal WCSS.
//!
//! # Connection to IVF
//!
//! K-means is the foundation of IVF (Inverted File) indexing for ANN search.
//! Partition vectors into k cells, search only nearby cells at query time.

use super::traits::Clustering;
use crate::error::{Error, Result};
use clump::cluster::Clustering as _;
use clump::cluster::Kmeans as ClumpKmeans;
use clump::DistanceMetric;

/// K-means clustering algorithm, generic over a distance metric.
///
/// The default metric is [`clump::SquaredEuclidean`], preserving backward
/// compatibility: `Kmeans::new(k)` works exactly as before.
///
/// To use a different metric, construct via [`Kmeans::with_metric`]:
///
/// ```rust
/// use sheaf::cluster::{Kmeans, CosineDistance};
///
/// let km = Kmeans::with_metric(8, CosineDistance).with_seed(42);
/// ```
#[derive(Debug, Clone)]
pub struct Kmeans<D: DistanceMetric = clump::SquaredEuclidean> {
    /// Number of clusters.
    k: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f64,
    /// Random seed.
    seed: Option<u64>,
    /// Seeding alpha (exponent for k-means++ probability weighting).
    /// Standard k-means++ uses alpha=2.0 (D-squared weighting).
    seeding_alpha: Option<f32>,
    /// Distance metric.
    metric: D,
}

impl Kmeans<clump::SquaredEuclidean> {
    /// Create a new K-means clusterer with the default squared Euclidean distance.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            seeding_alpha: None,
            metric: clump::SquaredEuclidean,
        }
    }
}

impl<D: DistanceMetric> Kmeans<D> {
    /// Create a new K-means clusterer with a custom distance metric.
    pub fn with_metric(k: usize, metric: D) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            seeding_alpha: None,
            metric,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set seeding alpha (exponent for k-means++ probability weighting).
    ///
    /// Standard k-means++ uses alpha=2.0 (D-squared weighting).
    /// Research (Bamas et al. 2023) suggests alpha > 2 (e.g. 4.0) can
    /// yield better final clustering cost.
    pub fn with_seeding_alpha(mut self, alpha: f32) -> Self {
        self.seeding_alpha = Some(alpha);
        self
    }
}

impl<D: DistanceMetric> Clustering for Kmeans<D> {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        let mut km = ClumpKmeans::with_metric(self.k, self.metric.clone())
            .with_max_iter(self.max_iter)
            .with_tol(self.tol);
        if let Some(seed) = self.seed {
            km = km.with_seed(seed);
        }
        if let Some(alpha) = self.seeding_alpha {
            km = km.with_seeding_alpha(alpha);
        }

        km.fit_predict(data).map_err(map_clump_error)
    }

    fn n_clusters(&self) -> usize {
        self.k
    }
}

fn map_clump_error(e: clump::Error) -> Error {
    match e {
        clump::Error::EmptyInput => Error::EmptyInput,
        clump::Error::InvalidParameter { name, message } => {
            Error::InvalidParameter { name, message }
        }
        clump::Error::InvalidClusterCount { requested, n_items } => {
            Error::InvalidClusterCount { requested, n_items }
        }
        clump::Error::DimensionMismatch { expected, found } => {
            Error::DimensionMismatch { expected, found }
        }
        clump::Error::ConstraintViolation(msg) => Error::ConstraintViolation(msg),
        clump::Error::Other(msg) => Error::Other(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans = Kmeans::new(2).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Points 0,1 should be in same cluster, points 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_kmeans_all_points_assigned() {
        // Property: every point must be assigned to exactly one cluster
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1, (i % 5) as f32])
            .collect();

        let kmeans = Kmeans::new(5).with_seed(123);
        let labels = kmeans.fit_predict(&data).unwrap();

        // All points assigned
        assert_eq!(labels.len(), data.len());

        // All labels in valid range [0, k)
        for &label in &labels {
            assert!(label < 5, "label {} out of range", label);
        }
    }

    #[test]
    fn test_kmeans_k_equals_n() {
        // Edge case: k = n (each point its own cluster)
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let kmeans = Kmeans::new(3).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Each point in different cluster
        let unique: std::collections::HashSet<_> = labels.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_kmeans_deterministic_with_seed() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans1 = Kmeans::new(2).with_seed(42);
        let kmeans2 = Kmeans::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&data).unwrap();

        assert_eq!(labels1, labels2, "same seed should give same result");
    }

    #[test]
    fn test_kmeans_scaling_invariant() {
        // Metamorphic: uniform scaling shouldn't change cluster assignments
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let scaled: Vec<Vec<f32>> = data
            .iter()
            .map(|v| v.iter().map(|x| x * 100.0).collect())
            .collect();

        let kmeans1 = Kmeans::new(2).with_seed(42);
        let kmeans2 = Kmeans::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&scaled).unwrap();

        // Same structure (labels may be permuted)
        assert_eq!(labels1[0], labels1[1]);
        assert_eq!(labels2[0], labels2[1]);
        assert_eq!(labels1[2], labels1[3]);
        assert_eq!(labels2[2], labels2[3]);
        assert_ne!(labels1[0], labels1[2]);
        assert_ne!(labels2[0], labels2[2]);
    }

    #[test]
    fn test_kmeans_empty_input_error() {
        let data: Vec<Vec<f32>> = vec![];
        let kmeans = Kmeans::new(2);
        let result = kmeans.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_larger_than_n_error() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let kmeans = Kmeans::new(5); // k > n
        let result = kmeans.fit_predict(&data);
        assert!(result.is_err());
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]
        #[test]
        fn prop_kmeans_deterministic_with_seed_on_random_data(
            n in 1usize..60,
            dim in 1usize..8,
            k in 1usize..8,
            seed in any::<u64>(),
        ) {
            prop_assume!(k <= n);
            let mut data: Vec<Vec<f32>> = Vec::with_capacity(n);
            // Deterministic pseudo-random-ish floats without pulling in extra RNGs here.
            for i in 0..n {
                let mut v = vec![0.0f32; dim];
                for d in 0..dim {
                    let u = (((i * 53 + d * 19) % 101) as f32 / 101.0) * 2.0 - 1.0;
                    // Add a tiny seed-dependent offset to avoid accidental symmetries, but keep it deterministic.
                    let off = (((seed ^ ((i as u64) << 32) ^ (d as u64)) % 97) as f32 - 48.0) * 1e-4;
                    v[d] = u + off;
                }
                data.push(v);
            }

            let km1 = Kmeans::new(k).with_seed(seed);
            let km2 = Kmeans::new(k).with_seed(seed);
            let a1 = km1.fit_predict(&data).unwrap();
            let a2 = km2.fit_predict(&data).unwrap();
            prop_assert_eq!(a1, a2);
        }
    }
}
