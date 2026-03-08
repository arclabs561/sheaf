//! DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
//!
//! Thin wrapper around [`clump::Dbscan`] that adapts it to sheaf's
//! [`Clustering`] trait and error types.
//!
//! See the [`clump`] crate documentation for algorithm details.

use super::traits::Clustering;
use crate::error::{Error, Result};
use clump::DistanceMetric;

/// Re-export the noise sentinel from clump.
pub use clump::NOISE;

/// DBSCAN clustering algorithm, generic over a distance metric.
///
/// Delegates to [`clump::Dbscan`] for the actual computation. See the
/// [clump docs](https://docs.rs/clump) for algorithm details (Ester et al., 1996).
///
/// The default metric is [`clump::Euclidean`] (L2), matching the original
/// behavior where epsilon is compared against Euclidean distance.
///
/// When using a different metric, epsilon semantics change accordingly:
/// - [`clump::SquaredEuclidean`]: epsilon is compared against squared distance.
/// - [`clump::CosineDistance`]: epsilon is compared against cosine distance (range `[0, 2]`).
///
/// # Core Concepts
///
/// - **Epsilon (ε)**: Maximum distance between two points to be neighbors.
/// - **MinPts**: Minimum neighbors within ε for a point to be "core".
/// - **Core point**: Has at least MinPts neighbors within ε.
/// - **Border point**: Within ε of a core point but not core itself.
/// - **Noise point**: Neither core nor border -- labeled with [`NOISE`] sentinel.
#[derive(Debug, Clone)]
pub struct Dbscan<D: DistanceMetric = clump::Euclidean> {
    inner: clump::Dbscan<D>,
}

impl Dbscan<clump::Euclidean> {
    /// Create a new DBSCAN clusterer with the default Euclidean distance.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Maximum distance between two points to be neighbors.
    /// * `min_pts` - Minimum number of points to form a dense region.
    ///
    /// # Typical Values
    ///
    /// - `epsilon`: Often determined by k-distance plot (k = min_pts - 1).
    /// - `min_pts`: 2 * dimension is a common heuristic. Minimum is 3.
    pub fn new(epsilon: f32, min_pts: usize) -> Self {
        Self {
            inner: clump::Dbscan::new(epsilon, min_pts),
        }
    }
}

impl<D: DistanceMetric> Dbscan<D> {
    /// Create a new DBSCAN clusterer with a custom distance metric.
    ///
    /// **Note**: epsilon semantics change with the metric. For example,
    /// with [`clump::SquaredEuclidean`], epsilon is compared against squared
    /// distances rather than raw Euclidean distances.
    pub fn with_metric(epsilon: f32, min_pts: usize, metric: D) -> Self {
        Self {
            inner: clump::Dbscan::with_metric(epsilon, min_pts, metric),
        }
    }

    /// Set epsilon (neighborhood radius).
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.inner = self.inner.with_epsilon(epsilon);
        self
    }

    /// Set minimum points for core classification.
    pub fn with_min_pts(mut self, min_pts: usize) -> Self {
        self.inner = self.inner.with_min_pts(min_pts);
        self
    }

    /// Check whether a label is the DBSCAN noise sentinel.
    pub fn is_noise(label: usize) -> bool {
        label == NOISE
    }
}

impl Default for Dbscan<clump::Euclidean> {
    fn default() -> Self {
        Self {
            inner: clump::Dbscan::default(),
        }
    }
}

impl<D: DistanceMetric> Clustering for Dbscan<D> {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        clump::Clustering::fit_predict(&self.inner, data).map_err(Error::from)
    }

    /// DBSCAN discovers clusters dynamically, so this returns 0.
    fn n_clusters(&self) -> usize {
        0
    }
}

/// Extended DBSCAN interface with noise detection.
pub trait DbscanExt {
    /// Fit and predict, returning labels where noise is marked as `None`.
    fn fit_predict_with_noise(&self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>>;

    /// Check if a label represents noise.
    fn is_noise(label: usize) -> bool {
        label == NOISE
    }
}

impl<D: DistanceMetric> DbscanExt for Dbscan<D> {
    fn fit_predict_with_noise(&self, data: &[Vec<f32>]) -> Result<Vec<Option<usize>>> {
        clump::DbscanExt::fit_predict_with_noise(&self.inner, data).map_err(Error::from)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn test_dbscan_two_clusters() {
        let data = vec![
            // Cluster 1: around (0, 0)
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            // Cluster 2: around (5, 5)
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
            vec![5.05, 5.05],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 10);

        let cluster1 = labels[0];
        for label in &labels[1..5] {
            assert_eq!(*label, cluster1);
        }

        let cluster2 = labels[5];
        for label in &labels[6..10] {
            assert_eq!(*label, cluster2);
        }

        assert_ne!(cluster1, cluster2);
    }

    #[test]
    fn test_dbscan_with_noise() {
        let data = vec![
            // Cluster 1
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            // Outlier
            vec![100.0, 100.0],
            // Cluster 2
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict_with_noise(&data).unwrap();

        assert_eq!(labels.len(), 9);
        assert!(labels[4].is_none());

        for (i, label) in labels.iter().enumerate() {
            if i != 4 {
                assert!(label.is_some());
            }
        }
    }

    #[test]
    fn test_dbscan_fit_predict_uses_noise_sentinel() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            // Outlier
            vec![100.0, 100.0],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],
            vec![5.1, 5.1],
        ];

        let dbscan = Dbscan::new(0.3, 3);
        let labels = dbscan.fit_predict(&data).unwrap();

        assert_eq!(labels.len(), 9);
        assert_eq!(labels[4], NOISE);
        assert!(Dbscan::<clump::Euclidean>::is_noise(labels[4]));
    }

    #[test]
    fn test_dbscan_all_noise() {
        let data = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
            vec![10.0, 10.0],
        ];

        let dbscan = Dbscan::new(0.5, 3);
        let labels = dbscan.fit_predict_with_noise(&data).unwrap();

        for label in labels {
            assert!(label.is_none());
        }
    }

    #[test]
    fn test_dbscan_all_one_cluster() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
        ];

        let dbscan = Dbscan::new(0.5, 2);
        let labels = dbscan.fit_predict(&data).unwrap();

        let cluster = labels[0];
        for label in labels {
            assert_eq!(label, cluster);
        }
    }

    #[test]
    fn test_dbscan_empty() {
        let data: Vec<Vec<f32>> = vec![];
        let dbscan = Dbscan::new(0.5, 3);
        let result = dbscan.fit_predict(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dbscan_invalid_params() {
        let data = vec![vec![0.0, 0.0]];

        let dbscan = Dbscan::new(0.0, 3);
        assert!(dbscan.fit_predict(&data).is_err());

        let dbscan = Dbscan::new(-1.0, 3);
        assert!(dbscan.fit_predict(&data).is_err());

        let dbscan = Dbscan::new(0.5, 0);
        assert!(dbscan.fit_predict(&data).is_err());
    }

    #[test]
    fn test_dbscan_chain() {
        let data: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.3, 0.0]).collect();

        let dbscan = Dbscan::new(0.5, 2);
        let labels = dbscan.fit_predict(&data).unwrap();

        let cluster = labels[0];
        for label in labels {
            assert_eq!(label, cluster);
        }
    }
}
