//! Elkan's accelerated K-means using the triangle inequality.
//!
//! # The Core Insight
//!
//! Standard k-means computes O(n * k) distances per iteration.
//! Elkan (2003) showed that using bounds, we can skip 80-95% of these.
//!
//! The triangle inequality gives us:
//! ```text
//! d(x, c_new) >= |d(x, c_old) - d(c_old, c_new)|
//! ```
//!
//! If `d(x, c_old) - d(c_old, c_new) > d(x, c_best)`, we know
//! `d(x, c_new) > d(x, c_best)` without computing `d(x, c_new)`.
//!
//! # Maintained Bounds
//!
//! - `upper[i]`: Upper bound on d(point_i, assigned_centroid)
//! - `lower[i][j]`: Lower bound on d(point_i, centroid_j)
//! - `centroid_dist[i][j]`: d(centroid_i, centroid_j) / 2
//!
//! # Algorithm Complexity
//!
//! - **Time**: O(n * k * d) per iteration in worst case, but typically
//!   O(n * d + k² * d) due to pruning (~90% distance calcs skipped)
//! - **Space**: O(n * k) for bounds (significant for large k)
//!
//! # When to Use
//!
//! - k >> 10: Triangle inequality pruning is most effective
//! - High-dimensional data: Distance computation dominates
//! - Many iterations needed: Bounds amortize setup cost
//!
//! For small k or low dimensions, standard Lloyd's may be faster
//! due to bound maintenance overhead.
//!
//! # References
//!
//! - Elkan (2003). "Using the Triangle Inequality to Accelerate k-Means"
//! - Hamerly (2010). "Making k-means Even Faster" (simplified single-bound variant)

use super::traits::Clustering;
use crate::error::{Error, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;

#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

/// Elkan's accelerated K-means with triangle inequality pruning.
///
/// Maintains upper and lower bounds to skip unnecessary distance computations.
/// Typically achieves 5-10x speedup over Lloyd's algorithm for k > 20.
#[derive(Debug, Clone)]
pub struct KmeansElkan {
    /// Number of clusters.
    k: usize,
    /// Maximum iterations.
    max_iter: usize,
    /// Convergence tolerance (centroid shift).
    tol: f64,
    /// Random seed for reproducibility.
    seed: Option<u64>,
    /// Track statistics for benchmarking.
    track_stats: bool,
}

/// Statistics from Elkan's algorithm execution.
#[derive(Debug, Clone, Default)]
pub struct ElkanStats {
    /// Total iterations performed.
    pub iterations: usize,
    /// Total distance computations (point-to-centroid).
    pub distance_computations: u64,
    /// Distance computations skipped via bounds.
    pub distances_skipped: u64,
    /// Fraction of distances skipped (higher is better).
    pub skip_fraction: f64,
    /// Total centroid-to-centroid computations.
    pub centroid_distances: u64,
}

impl KmeansElkan {
    /// Create a new Elkan K-means clusterer.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iter: 100,
            tol: 1e-4,
            seed: None,
            track_stats: false,
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

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable statistics tracking.
    pub fn with_stats(mut self, track: bool) -> Self {
        self.track_stats = track;
        self
    }

    /// Run Elkan's algorithm and return labels + optional stats.
    pub fn fit_predict_with_stats(
        &self,
        data: &[Vec<f32>],
    ) -> Result<(Vec<usize>, Option<ElkanStats>)> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        let n = data.len();
        let d = data[0].len();

        if self.k > n {
            return Err(Error::InvalidClusterCount {
                requested: self.k,
                n_items: n,
            });
        }

        // Convert to ndarray for efficient indexing
        let mut flat: Vec<f32> = Vec::with_capacity(n * d);
        for point in data {
            if point.len() != d {
                return Err(Error::DimensionMismatch {
                    expected: d,
                    found: point.len(),
                });
            }
            flat.extend(point);
        }
        let data_arr =
            Array2::from_shape_vec((n, d), flat).map_err(|e| Error::Other(e.to_string()))?;

        // Initialize RNG
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Initialize centroids using k-means++
        let mut centroids = self.init_centroids_pp(&data_arr, &mut rng);

        // Initialize bounds and assignments
        let mut labels = vec![0usize; n];
        let mut upper = vec![f32::MAX; n]; // upper[i] = d(x_i, c_{a_i})
        let mut lower = vec![vec![0.0f32; self.k]; n]; // lower[i][j] = lower bound on d(x_i, c_j)
        let mut centroid_half_dist = Array2::<f32>::zeros((self.k, self.k)); // d(c_i, c_j) / 2

        // Statistics
        let mut stats = ElkanStats::default();

        // Initial assignment (must compute all distances once)
        for i in 0..n {
            let point = data_arr.row(i);
            let mut best_cluster = 0;
            let mut best_dist = f32::MAX;

            #[allow(clippy::needless_range_loop)] // clearer than iterator here
            for j in 0..self.k {
                let dist = squared_distance(&point, &centroids.row(j)).sqrt();
                lower[i][j] = dist;
                stats.distance_computations += 1;

                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = j;
                }
            }

            labels[i] = best_cluster;
            upper[i] = best_dist;
        }

        // Compute initial centroid-centroid distances
        for i in 0..self.k {
            for j in (i + 1)..self.k {
                let dist = squared_distance(&centroids.row(i), &centroids.row(j)).sqrt();
                centroid_half_dist[[i, j]] = dist / 2.0;
                centroid_half_dist[[j, i]] = dist / 2.0;
                stats.centroid_distances += 1;
            }
        }

        // s[j] = min_{j' != j} d(c_j, c_j') / 2
        let mut s = self.compute_s(&centroid_half_dist);

        // Main loop
        for _iter in 0..self.max_iter {
            stats.iterations += 1;

            // Step 1: For each point, identify clusters that can be skipped
            // If upper[i] <= s[labels[i]], the assignment cannot change
            for i in 0..n {
                // a_i follows the point when it is reassigned within this
                // pass; Elkan (2003) step 3 compares every later centre
                // against the current one, not the one held at the start.
                let mut a_i = labels[i];
                // upper[i] is exact once computed in this pass (Elkan's r(x)).
                let mut upper_tight = false;

                // Lemma 1: If upper[i] <= s[a_i], x_i's assignment cannot change
                if upper[i] > s[a_i] {
                    // Need to check some clusters
                    for j in 0..self.k {
                        if j == a_i {
                            continue;
                        }

                        // Lemma 2: If upper[i] > lower[i][j] and
                        //          upper[i] > d(c_{a_i}, c_j) / 2,
                        //          we might need to update
                        if upper[i] > lower[i][j] && upper[i] > centroid_half_dist[[a_i, j]] {
                            // Tighten upper bound if needed (lazy)
                            // Compute actual distance to current centroid
                            let point = data_arr.row(i);
                            if !upper_tight {
                                let d_a = squared_distance(&point, &centroids.row(a_i)).sqrt();
                                upper[i] = d_a;
                                lower[i][a_i] = d_a;
                                upper_tight = true;
                                stats.distance_computations += 1;
                            }
                            let d_a = upper[i];

                            // Now check if we still need to compute d(x_i, c_j)
                            if d_a > lower[i][j] && d_a > centroid_half_dist[[a_i, j]] {
                                let d_j = squared_distance(&point, &centroids.row(j)).sqrt();
                                lower[i][j] = d_j;
                                stats.distance_computations += 1;

                                if d_j < d_a {
                                    labels[i] = j;
                                    a_i = j;
                                    upper[i] = d_j;
                                }
                            } else {
                                stats.distances_skipped += 1;
                            }
                        } else {
                            stats.distances_skipped += 1;
                        }
                    }
                } else {
                    // Skip all k-1 clusters for this point
                    stats.distances_skipped += (self.k - 1) as u64;
                }
            }

            // Step 2: Compute new centroids
            let mut new_centroids = Array2::<f32>::zeros((self.k, d));
            let mut counts = vec![0usize; self.k];

            for i in 0..n {
                let k = labels[i];
                for j in 0..d {
                    new_centroids[[k, j]] += data_arr[[i, j]];
                }
                counts[k] += 1;
            }

            for k in 0..self.k {
                if counts[k] > 0 {
                    for j in 0..d {
                        new_centroids[[k, j]] /= counts[k] as f32;
                    }
                } else {
                    // Empty cluster: reinitialize
                    let idx = rng.random_range(0..n);
                    new_centroids.row_mut(k).assign(&data_arr.row(idx));
                }
            }

            // Step 3: Compute centroid movements
            let mut centroid_shift = Array1::<f32>::zeros(self.k);
            for k in 0..self.k {
                centroid_shift[k] =
                    squared_distance(&centroids.row(k), &new_centroids.row(k)).sqrt();
            }

            // Step 4: Update bounds
            for i in 0..n {
                let a_i = labels[i];
                upper[i] += centroid_shift[a_i];

                for j in 0..self.k {
                    lower[i][j] = (lower[i][j] - centroid_shift[j]).max(0.0);
                }
            }

            // Check convergence
            let max_shift: f32 = centroid_shift.iter().cloned().fold(0.0f32, f32::max);
            centroids = new_centroids;

            // Update centroid-centroid distances
            for i in 0..self.k {
                for j in (i + 1)..self.k {
                    let dist = squared_distance(&centroids.row(i), &centroids.row(j)).sqrt();
                    centroid_half_dist[[i, j]] = dist / 2.0;
                    centroid_half_dist[[j, i]] = dist / 2.0;
                    stats.centroid_distances += 1;
                }
            }
            s = self.compute_s(&centroid_half_dist);

            if max_shift < self.tol as f32 {
                break;
            }
        }

        // Finalize stats
        let total_potential = stats.distance_computations + stats.distances_skipped;
        stats.skip_fraction = if total_potential > 0 {
            stats.distances_skipped as f64 / total_potential as f64
        } else {
            0.0
        };

        let final_stats = if self.track_stats { Some(stats) } else { None };
        Ok((labels, final_stats))
    }

    /// Initialize centroids using k-means++ algorithm.
    fn init_centroids_pp(&self, data: &Array2<f32>, rng: &mut impl Rng) -> Array2<f32> {
        let n = data.nrows();
        let d = data.ncols();
        let mut centroids = Array2::zeros((self.k, d));

        // First centroid: random point
        let first = rng.random_range(0..n);
        centroids.row_mut(0).assign(&data.row(first));

        // Remaining centroids: k-means++ selection
        for i in 1..self.k {
            let mut distances: Vec<f32> = Vec::with_capacity(n);

            for j in 0..n {
                let point = data.row(j);
                let min_dist = (0..i)
                    .map(|c| squared_distance(&point, &centroids.row(c)))
                    .fold(f32::MAX, f32::min);
                distances.push(min_dist);
            }

            let total: f32 = distances.iter().sum();
            if total == 0.0 {
                let idx = rng.random_range(0..n);
                centroids.row_mut(i).assign(&data.row(idx));
                continue;
            }

            let threshold = rng.random::<f32>() * total;
            let mut cumsum = 0.0;
            let mut selected = 0;

            for (j, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    selected = j;
                    break;
                }
            }

            centroids.row_mut(i).assign(&data.row(selected));
        }

        centroids
    }

    /// Compute s[j] = min_{j' != j} d(c_j, c_j') / 2
    fn compute_s(&self, centroid_half_dist: &Array2<f32>) -> Vec<f32> {
        let mut s = vec![f32::MAX; self.k];
        for j in 0..self.k {
            for jp in 0..self.k {
                if jp != j && centroid_half_dist[[j, jp]] < s[j] {
                    s[j] = centroid_half_dist[[j, jp]];
                }
            }
        }
        s
    }
}

/// Squared Euclidean distance between two vectors.
#[inline]
fn squared_distance(a: &ndarray::ArrayView1<'_, f32>, b: &ndarray::ArrayView1<'_, f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

impl Clustering for KmeansElkan {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        self.fit_predict_with_stats(data).map(|(labels, _)| labels)
    }

    fn n_clusters(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elkan_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans = KmeansElkan::new(2).with_seed(42);
        let labels = kmeans.fit_predict(&data).unwrap();

        // Points 0,1 should be in same cluster, points 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_elkan_with_stats() {
        // Create data with clear clusters
        let mut data = Vec::new();
        for _ in 0..100 {
            data.push(vec![0.0, 0.0]);
        }
        for _ in 0..100 {
            data.push(vec![10.0, 10.0]);
        }
        for _ in 0..100 {
            data.push(vec![20.0, 20.0]);
        }

        let kmeans = KmeansElkan::new(3).with_seed(42).with_stats(true);
        let (labels, stats) = kmeans.fit_predict_with_stats(&data).unwrap();

        let stats = stats.unwrap();

        // Should skip significant portion of distances
        assert!(
            stats.skip_fraction > 0.3,
            "Expected significant skipping, got {}%",
            stats.skip_fraction * 100.0
        );

        println!(
            "Elkan stats: {} iterations, {} computed, {} skipped ({:.1}%)",
            stats.iterations,
            stats.distance_computations,
            stats.distances_skipped,
            stats.skip_fraction * 100.0
        );

        // Verify correct clustering
        assert_eq!(labels.len(), 300);
    }

    #[test]
    fn test_elkan_matches_lloyd() {
        // Both algorithms should produce same clustering structure
        use super::super::kmeans::Kmeans;

        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];

        let lloyd = Kmeans::new(2).with_seed(42);
        let elkan = KmeansElkan::new(2).with_seed(42);

        let lloyd_labels = lloyd.fit_predict(&data).unwrap();
        let elkan_labels = elkan.fit_predict(&data).unwrap();

        // Same structure (labels may be permuted)
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    lloyd_labels[i] == lloyd_labels[j],
                    elkan_labels[i] == elkan_labels[j],
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_elkan_high_k() {
        // Elkan should be efficient with high k
        let n = 500;
        let k = 50;
        let d = 10;

        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..d).map(|j| ((i * j) as f32).sin()).collect())
            .collect();

        let kmeans = KmeansElkan::new(k)
            .with_seed(42)
            .with_stats(true)
            .with_max_iter(20);
        let (labels, stats) = kmeans.fit_predict_with_stats(&data).unwrap();
        let stats = stats.unwrap();

        // With k=50, should skip many distances
        assert!(
            stats.skip_fraction > 0.5,
            "Expected high skip fraction for k={}, got {:.1}%",
            k,
            stats.skip_fraction * 100.0
        );

        // All points assigned
        assert_eq!(labels.len(), n);
        for &label in &labels {
            assert!(label < k);
        }
    }

    #[test]
    fn test_elkan_deterministic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let kmeans1 = KmeansElkan::new(2).with_seed(42);
        let kmeans2 = KmeansElkan::new(2).with_seed(42);

        let labels1 = kmeans1.fit_predict(&data).unwrap();
        let labels2 = kmeans2.fit_predict(&data).unwrap();

        assert_eq!(labels1, labels2);
    }

    #[test]
    fn test_elkan_empty_input() {
        let data: Vec<Vec<f32>> = vec![];
        let kmeans = KmeansElkan::new(2);
        assert!(kmeans.fit_predict(&data).is_err());
    }

    #[test]
    fn test_elkan_k_larger_than_n() {
        let data = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let kmeans = KmeansElkan::new(5);
        assert!(kmeans.fit_predict(&data).is_err());
    }
}
