//! Gaussian Mixture Model clustering.
//!
//! GMM provides **soft clustering** with probabilistic assignments,
//! allowing items to belong to multiple clusters with different probabilities.
//!
//! # The Probabilistic Model
//!
//! GMM assumes data is generated from K Gaussian distributions:
//!
//! ```text
//! P(x) = Σₖ πₖ × N(x | μₖ, Σₖ)
//! ```
//!
//! Where:
//! - πₖ = mixing weight (probability of cluster k)
//! - μₖ = mean of cluster k
//! - Σₖ = covariance matrix of cluster k
//!
//! # The EM Algorithm
//!
//! Direct optimization is intractable (sum inside log). EM provides an
//! iterative coordinate-ascent solution:
//!
//! **E-step**: Compute "responsibilities" (soft assignments):
//! ```text
//! γₙₖ = P(z=k | xₙ) = πₖ × N(xₙ | μₖ, Σₖ) / Σⱼ πⱼ × N(xₙ | μⱼ, Σⱼ)
//! ```
//!
//! **M-step**: Update parameters using responsibilities:
//! - μₖ = Σₙ γₙₖ xₙ / Σₙ γₙₖ  (weighted mean)
//! - πₖ = (1/N) Σₙ γₙₖ  (fraction of responsibility)
//!
//! # Why Soft Clustering?
//!
//! In RAPTOR, text chunks often span multiple topics. A chunk about
//! "machine learning for healthcare" should partially belong to both
//! "ML" and "healthcare" clusters. Hard assignment loses this nuance.
//!
//! # Failure Modes
//!
//! - **Local optima**: EM converges to local maxima; initialization matters
//! - **Singular covariance**: Small clusters can collapse; we add regularization
//! - **Wrong K**: Too many components overfit; too few underfit

use super::traits::{Clustering, SoftClustering};
use crate::error::{Error, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
// TODO: use Normal distribution for proper EM initialization
#[allow(unused_imports)]
use rand_distr::Normal;

/// Gaussian Mixture Model clustering.
#[derive(Debug, Clone)]
pub struct Gmm {
    /// Number of components (clusters).
    n_components: usize,
    /// Maximum EM iterations.
    max_iter: usize,
    /// Convergence tolerance (TODO: add early stopping).
    #[allow(dead_code)]
    tol: f64,
    /// Random seed.
    seed: Option<u64>,
    /// Regularization for covariance.
    reg_covar: f64,
}

impl Gmm {
    /// Create a new GMM with specified number of components.
    pub fn new() -> Self {
        Self {
            n_components: 8,
            max_iter: 100,
            tol: 1e-3,
            seed: None,
            reg_covar: 1e-6,
        }
    }

    /// Set number of components.
    pub fn with_n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute log-likelihood of a point under a Gaussian.
    fn log_gaussian(
        point: &ndarray::ArrayView1<'_, f32>,
        mean: &ndarray::ArrayView1<'_, f64>,
        var: &ndarray::ArrayView1<'_, f64>,
    ) -> f64 {
        let d = point.len() as f64;
        let mut log_prob = -0.5 * d * (2.0 * std::f64::consts::PI).ln();

        for i in 0..point.len() {
            let diff = point[i] as f64 - mean[i];
            log_prob -= 0.5 * var[i].ln();
            log_prob -= 0.5 * diff * diff / var[i];
        }

        log_prob
    }

    /// Log-sum-exp for numerical stability.
    fn logsumexp(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val.is_infinite() {
            return max_val;
        }
        max_val
            + values
                .iter()
                .map(|&v| (v - max_val).exp())
                .sum::<f64>()
                .ln()
    }
}

impl Default for Gmm {
    fn default() -> Self {
        Self::new()
    }
}

impl Clustering for Gmm {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        let probs = self.fit_predict_proba(data)?;

        // Hard assignment: argmax
        Ok(probs
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect())
    }

    fn n_clusters(&self) -> usize {
        self.n_components
    }
}

impl SoftClustering for Gmm {
    fn fit_predict_proba(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f64>>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }

        let n = data.len();
        let d = data[0].len();
        let k = self.n_components.min(n);

        if k == 0 {
            return Err(Error::InvalidParameter {
                name: "n_components",
                message: "must be > 0",
            });
        }

        // Convert to ndarray
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

        // Initialize parameters
        // Means: random points from data
        let mut means = Array2::zeros((k, d));
        let indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let idx = indices[rng.random_range(0..n)];
            for j in 0..d {
                means[[i, j]] = data_arr[[idx, j]] as f64;
            }
        }

        // Variances: initialized to data variance
        let mut variances = Array2::from_elem((k, d), 1.0);

        // Weights: uniform
        let mut weights = Array1::from_elem(k, 1.0 / k as f64);

        // Responsibilities
        let mut resp = Array2::zeros((n, k));

        // EM iterations
        for _iter in 0..self.max_iter {
            // E-step: compute responsibilities
            for i in 0..n {
                let point = data_arr.row(i);
                let mut log_probs = vec![0.0; k];

                for c in 0..k {
                    log_probs[c] = weights[c].ln()
                        + Self::log_gaussian(&point, &means.row(c), &variances.row(c));
                }

                let log_sum = Self::logsumexp(&log_probs);

                for c in 0..k {
                    resp[[i, c]] = (log_probs[c] - log_sum).exp();
                }
            }

            // M-step: update parameters
            let resp_sum: Vec<f64> = (0..k).map(|c| resp.column(c).sum()).collect();
            let total: f64 = resp_sum.iter().sum();

            // Update weights
            for c in 0..k {
                weights[c] = resp_sum[c] / total;
            }

            // Update means
            let mut new_means = Array2::zeros((k, d));
            for c in 0..k {
                if resp_sum[c] > 1e-10 {
                    for i in 0..n {
                        for j in 0..d {
                            new_means[[c, j]] += resp[[i, c]] * data_arr[[i, j]] as f64;
                        }
                    }
                    for j in 0..d {
                        new_means[[c, j]] /= resp_sum[c];
                    }
                } else {
                    new_means.row_mut(c).assign(&means.row(c));
                }
            }

            // Update variances
            let mut new_variances = Array2::from_elem((k, d), self.reg_covar);
            for c in 0..k {
                if resp_sum[c] > 1e-10 {
                    for i in 0..n {
                        for j in 0..d {
                            let diff = data_arr[[i, j]] as f64 - new_means[[c, j]];
                            new_variances[[c, j]] += resp[[i, c]] * diff * diff;
                        }
                    }
                    for j in 0..d {
                        new_variances[[c, j]] /= resp_sum[c];
                        new_variances[[c, j]] = new_variances[[c, j]].max(self.reg_covar);
                    }
                }
            }

            means = new_means;
            variances = new_variances;
        }

        // Return responsibilities as Vec<Vec<f64>>
        Ok((0..n)
            .map(|i| (0..k).map(|c| resp[[i, c]]).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmm_basic() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let gmm = Gmm::new().with_n_components(2).with_seed(42);
        let labels = gmm.fit_predict(&data).unwrap();

        // Should find 2 clusters
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
    }

    #[test]
    fn test_gmm_soft_assignments() {
        let data = vec![
            vec![0.0, 0.0],
            vec![5.0, 5.0], // Point between clusters
            vec![10.0, 10.0],
        ];

        let gmm = Gmm::new().with_n_components(2).with_seed(42);
        let probs = gmm.fit_predict_proba(&data).unwrap();

        // Each row should sum to ~1
        for row in &probs {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
