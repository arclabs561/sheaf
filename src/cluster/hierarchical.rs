//! Hierarchical (agglomerative) clustering.
//!
//! Bottom-up clustering that builds a **dendrogram** by iteratively
//! merging the closest clusters. Unlike K-means or GMM, you don't
//! need to specify k in advance—cut the tree at any height.
//!
//! # Linkage Methods
//!
//! The key choice: how do we define "distance between clusters"?
//!
//! | Linkage | Formula | Effect |
//! |---------|---------|--------|
//! | Single | min(d(a,b)) for a∈A, b∈B | Chaining; elongated clusters |
//! | Complete | max(d(a,b)) | Compact, spherical clusters |
//! | Average | mean(d(a,b)) | Balanced compromise |
//! | Ward | Δ variance | Minimizes within-cluster variance |
//!
//! ## Ward's Method: Variance Minimization
//!
//! Ward linkage minimizes the increase in total within-cluster variance
//! when merging clusters A and B:
//!
//! ```text
//! Δ(A,B) = (nₐ × nᵦ)/(nₐ + nᵦ) × ||μₐ - μᵦ||²
//! ```
//!
//! Where nₐ, nᵦ are cluster sizes and μₐ, μᵦ are centroids.
//!
//! **Intuition**: Merging similar-sized clusters with close centroids
//! increases variance the least. This produces compact, roughly
//! equal-sized clusters.
//!
//! # When to Use
//!
//! - **Exploratory analysis**: View cluster structure at multiple granularities
//! - **Unknown k**: Cut dendrogram at different heights to explore
//! - **Small-medium data**: O(n²) space for distance matrix

use super::traits::Clustering;
use crate::error::{Error, Result};
use crate::hierarchy::Dendrogram;
use kodama::{linkage as kodama_linkage, Method as KodamaMethod};

/// Linkage method for hierarchical clustering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Single linkage: minimum distance between clusters.
    Single,
    /// Complete linkage: maximum distance between clusters.
    Complete,
    /// Average linkage: mean distance between clusters.
    Average,
    /// Ward's method: minimize within-cluster variance.
    Ward,
}

/// Hierarchical (agglomerative) clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalClustering {
    /// Number of clusters to produce.
    n_clusters: usize,
    /// Linkage method.
    linkage: Linkage,
}

impl HierarchicalClustering {
    /// Create a new hierarchical clusterer.
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Average,
        }
    }

    /// Set linkage method.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Fit and return the full dendrogram.
    pub fn fit_dendrogram(&self, data: &[Vec<f32>]) -> Result<Dendrogram> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        let n = data.len();
        let d = data[0].len();
        if let Some((_, p)) = data.iter().enumerate().find(|(_, p)| p.len() != d) {
            return Err(Error::DimensionMismatch {
                expected: d,
                found: p.len(),
            });
        }

        // Build a condensed dissimilarity matrix (upper triangle, row-major).
        // Length is N-choose-2.
        let mut condensed = Vec::with_capacity((n * (n - 1)) / 2);
        for row in 0..(n - 1) {
            for col in (row + 1)..n {
                condensed.push(self.euclidean_distance_f64(&data[row], &data[col]));
            }
        }

        let method = match self.linkage {
            Linkage::Single => KodamaMethod::Single,
            Linkage::Complete => KodamaMethod::Complete,
            Linkage::Average => KodamaMethod::Average,
            Linkage::Ward => KodamaMethod::Ward,
        };

        // Run hierarchical clustering using kodama (BurntSushi).
        //
        // kodama's dendrogram uses SciPy/MATLAB-style cluster labels:
        // - leaves: 0..n-1
        // - each merge i creates cluster id n+i
        let dend = kodama_linkage(&mut condensed, n, method);

        let mut dendro = Dendrogram::new(n);
        for step in dend.steps() {
            dendro.add_merge(step.cluster1, step.cluster2, step.dissimilarity, step.size);
        }

        Ok(dendro)
    }

    /// Fit a dendrogram from a precomputed condensed distance matrix.
    ///
    /// The condensed matrix is the upper triangle of the full pairwise distance
    /// matrix in row-major order (same format as SciPy's `pdist`). For `n` items,
    /// the length must be `n * (n - 1) / 2`.
    ///
    /// This is useful when the distance metric is not Euclidean (e.g., `1 - similarity`
    /// from a learned scoring function).
    ///
    /// **Note**: Ward linkage requires Euclidean distances to be meaningful.
    pub fn fit_dendrogram_from_condensed(
        &self,
        mut condensed: Vec<f64>,
        n: usize,
    ) -> Result<Dendrogram> {
        let expected_len = n * (n - 1) / 2;
        if condensed.len() != expected_len {
            return Err(Error::DimensionMismatch {
                expected: expected_len,
                found: condensed.len(),
            });
        }
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        let method = match self.linkage {
            Linkage::Single => KodamaMethod::Single,
            Linkage::Complete => KodamaMethod::Complete,
            Linkage::Average => KodamaMethod::Average,
            Linkage::Ward => KodamaMethod::Ward,
        };

        let dend = kodama_linkage(&mut condensed, n, method);

        let mut dendro = Dendrogram::new(n);
        for step in dend.steps() {
            dendro.add_merge(step.cluster1, step.cluster2, step.dissimilarity, step.size);
        }

        Ok(dendro)
    }

    /// Euclidean distance between two points.
    #[inline]
    fn euclidean_distance_f64(&self, a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let dx = *x as f64 - *y as f64;
                dx * dx
            })
            .sum::<f64>()
            .sqrt()
    }
}

impl Clustering for HierarchicalClustering {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        let dendro = self.fit_dendrogram(data)?;
        dendro.cut_to_k(self.n_clusters)
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let hc = HierarchicalClustering::new(2);
        let labels = hc.fit_predict(&data).unwrap();

        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_dendrogram() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![10.0, 0.0]];

        let hc = HierarchicalClustering::new(2);
        let dendro = hc.fit_dendrogram(&data).unwrap();

        assert_eq!(dendro.n_items(), 3);
        assert_eq!(dendro.n_merges(), 2);
    }

    #[test]
    fn test_from_condensed_matches_from_vectors() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        let n = data.len();

        // Build condensed distance matrix manually
        let mut condensed = Vec::new();
        for i in 0..(n - 1) {
            for j in (i + 1)..n {
                let d: f64 = data[i]
                    .iter()
                    .zip(data[j].iter())
                    .map(|(a, b)| {
                        let dx = *a as f64 - *b as f64;
                        dx * dx
                    })
                    .sum::<f64>()
                    .sqrt();
                condensed.push(d);
            }
        }

        let hc = HierarchicalClustering::new(2);
        let labels_vec = hc.fit_predict(&data).unwrap();
        let labels_condensed = hc
            .fit_dendrogram_from_condensed(condensed, n)
            .unwrap()
            .cut_to_k(2)
            .unwrap();

        // Same clustering result
        assert_eq!(labels_vec[0], labels_vec[1]);
        assert_eq!(labels_condensed[0], labels_condensed[1]);
        assert_eq!(labels_vec[2], labels_vec[3]);
        assert_eq!(labels_condensed[2], labels_condensed[3]);
        assert_ne!(labels_vec[0], labels_vec[2]);
        assert_ne!(labels_condensed[0], labels_condensed[2]);
    }

    #[test]
    fn test_from_condensed_similarity_to_distance() {
        // Simulate coreference: convert similarity scores to distances
        // Items 0,1 are coreferent (high similarity); 2 is separate
        let similarities = vec![
            0.9, // (0,1) - high similarity
            0.1, // (0,2) - low similarity
            0.2, // (1,2) - low similarity
        ];
        let condensed: Vec<f64> = similarities.iter().map(|s| 1.0 - s).collect();

        let hc = HierarchicalClustering::new(2);
        let labels = hc
            .fit_dendrogram_from_condensed(condensed, 3)
            .unwrap()
            .cut_to_k(2)
            .unwrap();

        // 0 and 1 should be in the same cluster
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[2]);
    }
}
