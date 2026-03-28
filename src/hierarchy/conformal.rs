//! Conformal prediction for hierarchical data.
//!
//! Implements Split Conformal Prediction (SCP) for hierarchies, ensuring
//! that prediction intervals are coherent across levels.
//!
//! Based on: Principato et al. (2024). "Conformal Prediction for Hierarchical Data."

use crate::error::{Error, Result};
use crate::reconciliation::{reconcile, ReconciliationMethod, SummingMatrix};
use faer::Mat;

/// Hierarchical Conformal Prediction.
pub struct HierarchicalConformal {
    s: SummingMatrix,
    method: ReconciliationMethod,
    scores: Vec<f64>,
    quantile: f64,
}

impl HierarchicalConformal {
    /// Create a new conformal predictor for a hierarchy.
    pub fn new(s: SummingMatrix, method: ReconciliationMethod) -> Self {
        Self {
            s,
            method,
            scores: Vec::new(),
            quantile: 0.0,
        }
    }

    /// Calibrate the predictor using a calibration set.
    ///
    /// # Arguments
    /// * `y_calib` - True values for calibration set ($m \times N$)
    /// * `y_hat_calib` - Base forecasts for calibration set ($m \times N$)
    /// * `alpha` - Miscoverage level (e.g., 0.1 for 90% coverage)
    pub fn calibrate(
        &mut self,
        y_calib: &Mat<f64>,
        y_hat_calib: &Mat<f64>,
        alpha: f64,
    ) -> Result<()> {
        let n_calib = y_calib.ncols();
        if y_hat_calib.ncols() != n_calib {
            return Err(Error::ShapeMismatch {
                expected: format!("{} columns", n_calib),
                actual: format!("{} columns", y_hat_calib.ncols()),
            });
        }

        // 1. Reconcile calibration forecasts
        let y_tilde_calib = reconcile(&self.s, y_hat_calib, self.method.clone())?;

        // 2. Compute non-conformity scores
        // For joint coverage, we use the Mahalanobis distance if possible,
        // or just Euclidean distance.
        let mut scores = Vec::with_capacity(n_calib);

        for j in 0..n_calib {
            let mut score: f64 = 0.0;
            for i in 0..y_calib.nrows() {
                let diff = y_calib[(i, j)] - y_tilde_calib[(i, j)];
                score += diff.powi(2);
            }
            scores.push(score.sqrt());
        }

        // 3. Compute quantile
        scores.sort_by(|a, b| a.total_cmp(b));
        let q_idx = (((n_calib + 1) as f64 * (1.0 - alpha)).ceil() as usize).min(n_calib) - 1;
        self.quantile = scores[q_idx];
        self.scores = scores;

        Ok(())
    }

    /// Predict intervals for new base forecasts.
    pub fn predict(&self, y_hat: &Mat<f64>) -> Result<Mat<f64>> {
        // Reconcile and return intervals
        reconcile(&self.s, y_hat, self.method.clone())
    }

    /// Get the computed quantile.
    pub fn quantile(&self) -> f64 {
        self.quantile
    }

    /// Get prediction intervals $[y_{lower}, y_{upper}]$ for joint coverage.
    ///
    /// For joint coverage with radius $Q$, the interval for each component $i$
    /// is $[\tilde{y}_i - Q, \tilde{y}_i + Q]$.
    pub fn predict_intervals(&self, y_hat: &Mat<f64>) -> Result<(Mat<f64>, Mat<f64>)> {
        let y_tilde = self.predict(y_hat)?;
        let mut lower = y_tilde.clone();
        let mut upper = y_tilde.clone();

        for j in 0..y_tilde.ncols() {
            for i in 0..y_tilde.nrows() {
                lower[(i, j)] -= self.quantile;
                upper[(i, j)] += self.quantile;
            }
        }

        Ok((lower, upper))
    }
}
