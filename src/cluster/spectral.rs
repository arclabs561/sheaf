//! Spectral clustering via graph Laplacian eigenvectors.
//!
//! Spectral clustering works by:
//! 1. Build similarity graph from points
//! 2. Compute normalized Laplacian
//! 3. Find k smallest eigenvectors
//! 4. Run k-means on the embedded points
//!
//! # When to Use Spectral Clustering
//!
//! Spectral clustering excels at finding non-convex clusters that k-means misses.
//! It's particularly good for:
//!
//! - Image segmentation (groups of similar pixels)
//! - Community detection in graphs
//! - Clusters with complex shapes
//!
//! # Trade-offs
//!
//! | Aspect | Spectral | K-means |
//! |--------|----------|---------|
//! | Shape | Any | Convex |
//! | Complexity | O(n³) eigendecomp | O(nkd × iter) |
//! | Memory | O(n²) similarity | O(nd) |
//! | Scalability | < 10k points | Millions |
//!
//! # Algorithm
//!
//! ```text
//! 1. Compute affinity matrix A (Gaussian kernel or kNN)
//! 2. Compute normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}
//! 3. Find k smallest eigenvectors of L_sym (excluding trivial)
//! 4. Form matrix U ∈ R^{n×k} from eigenvectors as columns
//! 5. Normalize rows of U to unit length
//! 6. Run k-means on rows of U
//! ```
//!
//! # Example
//!
//! ```ignore
//! use parti::cluster::spectral::SpectralClustering;
//! use ndarray::array;
//!
//! let points = array![
//!     [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],  // Cluster 1
//!     [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],  // Cluster 2
//! ];
//!
//! let labels = SpectralClustering::new(2)
//!     .sigma(1.0)
//!     .fit(&points)?;
//! ```
//!
//! # References
//!
//! - Ng, Jordan, Weiss (2001). "On Spectral Clustering"
//! - von Luxburg (2007). "A Tutorial on Spectral Clustering"

use lapl::{gaussian_similarity, knn_graph, spectral_embedding, SpectralEmbeddingConfig};
use ndarray::Array2;

use crate::{Error, Result};

/// Spectral clustering configuration and runner.
///
/// Uses graph Laplacian eigenvectors to embed points, then k-means to cluster.
#[derive(Debug, Clone)]
pub struct SpectralClustering {
    /// Number of clusters
    k: usize,
    /// Affinity type
    affinity: AffinityType,
    /// Sigma for Gaussian kernel (if using rbf affinity)
    sigma: f64,
    /// Number of neighbors for kNN affinity
    n_neighbors: usize,
    /// K-means iterations
    kmeans_iter: usize,
}

/// Type of affinity matrix to build.
#[derive(Debug, Clone, Copy)]
pub enum AffinityType {
    /// Gaussian (RBF) kernel: exp(-||x-y||² / 2σ²)
    Rbf,
    /// k-nearest neighbors graph
    Knn,
    /// Precomputed affinity matrix
    Precomputed,
}

impl SpectralClustering {
    /// Create new spectral clustering with k clusters.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            affinity: AffinityType::Rbf,
            sigma: 1.0,
            n_neighbors: 10,
            kmeans_iter: 100,
        }
    }

    /// Set affinity type.
    pub fn affinity(mut self, affinity: AffinityType) -> Self {
        self.affinity = affinity;
        self
    }

    /// Set sigma for Gaussian kernel.
    pub fn sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma;
        self
    }

    /// Set number of neighbors for kNN affinity.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set number of k-means iterations.
    pub fn kmeans_iter(mut self, iter: usize) -> Self {
        self.kmeans_iter = iter;
        self
    }

    /// Fit spectral clustering to points.
    ///
    /// # Arguments
    ///
    /// * `points` - n × d matrix of n points in d dimensions
    ///
    /// # Returns
    ///
    /// Cluster assignments for each point.
    pub fn fit(&self, points: &Array2<f64>) -> Result<Vec<usize>> {
        let n = points.nrows();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if n < self.k {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        // Build affinity matrix
        let affinity = self.build_affinity(points);

        // Delegate spectral embedding to `lapl` (single source of truth for Laplacian + eigensolver policy).
        // Spectral clustering typically uses the first k eigenvectors (Ng–Jordan–Weiss),
        // then row-normalizes. Keeping the constant eigenvector improves stability here.
        let mut cfg = SpectralEmbeddingConfig::default();
        cfg.skip_first = false;
        let embedding = spectral_embedding(&affinity, self.k, &cfg)
            .map_err(|e| Error::Other(format!("lapl spectral_embedding failed: {e}")))?;

        // Run k-means on the embedding.
        self.kmeans_on_embedding(&embedding)
    }

    /// Fit using precomputed affinity matrix.
    pub fn fit_affinity(&self, affinity: &Array2<f64>) -> Result<Vec<usize>> {
        let n = affinity.nrows();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if affinity.ncols() != n {
            return Err(Error::DimensionMismatch {
                expected: n,
                found: affinity.ncols(),
            });
        }

        let mut cfg = SpectralEmbeddingConfig::default();
        cfg.skip_first = false;
        let embedding = spectral_embedding(affinity, self.k, &cfg)
            .map_err(|e| Error::Other(format!("lapl spectral_embedding failed: {e}")))?;
        self.kmeans_on_embedding(&embedding)
    }

    fn build_affinity(&self, points: &Array2<f64>) -> Array2<f64> {
        match self.affinity {
            AffinityType::Rbf => gaussian_similarity(points, self.sigma),
            AffinityType::Knn => {
                // Build distance matrix first
                let n = points.nrows();
                let mut distances = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let mut dist_sq = 0.0;
                        for d in 0..points.ncols() {
                            let diff = points[[i, d]] - points[[j, d]];
                            dist_sq += diff * diff;
                        }
                        distances[[i, j]] = dist_sq.sqrt();
                    }
                }
                knn_graph(&distances, self.n_neighbors)
            }
            AffinityType::Precomputed => {
                // This shouldn't happen via fit()
                panic!("use fit_affinity for precomputed affinity")
            }
        }
    }

    fn kmeans_on_embedding(&self, embedding: &Array2<f64>) -> Result<Vec<usize>> {
        let n = embedding.nrows();
        let d = embedding.ncols();
        let k = self.k;

        // Delegate k-means to `clump` (single source of truth for k-means correctness/perf).
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(d);
            for j in 0..d {
                row.push(embedding[[i, j]] as f32);
            }
            rows.push(row);
        }

        // Use a small number of deterministic restarts and pick the best WCSS.
        // (k-means++ can still pick an unlucky first centroid on tiny problems.)
        let base_seed = 42u64;
        let mut best: Option<(f32, Vec<usize>)> = None;

        for t in 0..4u64 {
            let fit = clump::cluster::Kmeans::new(k)
                .with_max_iter(self.kmeans_iter)
                .with_tol(1e-4)
                .with_seed(base_seed.wrapping_add(t))
                .fit(&rows)
                .map_err(|e| Error::Other(format!("clump kmeans failed: {e}")))?;

            // Compute WCSS in f32: sum_i ||x_i - c_{a_i}||^2
            let mut wcss = 0.0f32;
            for (i, &a) in fit.labels.iter().enumerate() {
                let c = &fit.centroids[a];
                let x = &rows[i];
                let mut d2 = 0.0f32;
                for j in 0..d {
                    let diff = x[j] - c[j];
                    d2 += diff * diff;
                }
                wcss += d2;
            }

            let labels = fit.labels;
            match &mut best {
                None => best = Some((wcss, labels)),
                Some((best_wcss, best_assignments)) => {
                    if wcss < *best_wcss {
                        *best_wcss = wcss;
                        *best_assignments = labels;
                    }
                }
            }
        }

        Ok(best.expect("n>0 implies at least one kmeans run").1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_spectral_two_clusters() {
        // Two clearly separated clusters
        let points = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ];

        let labels = SpectralClustering::new(2)
            .sigma(1.0)
            .kmeans_iter(50)
            .fit(&points)
            .expect("fit should succeed");

        // First 3 should be in same cluster, last 3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_spectral_with_knn() {
        let points = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ];

        let labels = SpectralClustering::new(2)
            .affinity(AffinityType::Knn)
            .n_neighbors(2)
            .fit(&points)
            .expect("fit should succeed");

        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_spectral_empty_error() {
        let points: Array2<f64> = Array2::zeros((0, 2));
        let result = SpectralClustering::new(2).fit(&points);
        assert!(result.is_err());
    }
}
