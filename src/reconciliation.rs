//! Reconciliation methods for hierarchical forecasts.
//!
//! Hierarchical data follows a structural constraint: $y = S \cdot b$, where
//! $b$ are the "bottom-level" items and $S$ is the summing matrix.
//!
//! Reconciliation ensures that base forecasts $\hat{y}$ are adjusted to
//! $\tilde{y}$ such that $\tilde{y}$ satisfies the constraints.

use crate::error::{Error, Result};
use faer::prelude::*;
use faer::{Mat, MatRef};

/// A structural summing matrix for a hierarchy.
///
/// For a hierarchy with $m$ total nodes and $n$ leaf nodes,
/// $S$ is an $m \times n$ binary matrix where $S_{ij} = 1$ if
/// leaf $j$ is a descendant of node $i$.
#[derive(Debug, Clone)]
pub struct SummingMatrix {
    inner: Mat<f64>,
}

impl SummingMatrix {
    /// Create a new summing matrix from a faer matrix.
    pub fn new(inner: Mat<f64>) -> Self {
        Self { inner }
    }

    /// Number of total nodes (rows).
    pub fn m(&self) -> usize {
        self.inner.nrows()
    }

    /// Number of leaf nodes (columns).
    pub fn n(&self) -> usize {
        self.inner.ncols()
    }

    /// Get the matrix reference.
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        self.inner.as_ref()
    }

    /// Generate S for a simple 2-level hierarchy (root + n leaves).
    pub fn simple_star(n_leaves: usize) -> Self {
        let mut mat = Mat::<f64>::zeros(n_leaves + 1, n_leaves);
        // Root (row 0) is sum of all leaves
        for j in 0..n_leaves {
            mat[(0, j)] = 1.0;
        }
        // Leaves (rows 1..n+1)
        for j in 0..n_leaves {
            mat[(j + 1, j)] = 1.0;
        }
        Self { inner: mat }
    }
}

/// Reconciliation strategies.
#[derive(Debug, Clone)]
pub enum ReconciliationMethod {
    /// Ordinary Least Squares: $G = (S^T S)^{-1} S^T$
    Ols,
    /// Weighted Least Squares: $G = (S^T W^{-1} S)^{-1} S^T W^{-1}$
    Wls {
        /// Diagonal weights (m-dimensional).
        weights: Vec<f64>,
    },
    /// Minimum Trace (MinT): $G = (S^T \Sigma^{-1} S)^{-1} S^T \Sigma^{-1}$
    MinT {
        /// Full covariance matrix (m x m).
        covariance: Mat<f64>,
    },
}

/// Reconcile base forecasts $\hat{y}$ using the structural matrix $S$.
///
/// Returns $\tilde{y} = S \cdot G \cdot \hat{y}$.
pub fn reconcile(
    s: &SummingMatrix,
    base_forecasts: &Mat<f64>, // m x 1 or m x k
    method: ReconciliationMethod,
) -> Result<Mat<f64>> {
    let s_mat = s.as_ref();
    let m = s.m();
    let n = s.n();

    if base_forecasts.nrows() != m {
        return Err(Error::ShapeMismatch {
            expected: format!("{} rows", m),
            actual: format!("{} rows", base_forecasts.nrows()),
        });
    }

    // We solve the problem: (S^T W^-1 S) b = S^T W^-1 y_hat
    // Then y_tilde = S b

    let b = match method {
        ReconciliationMethod::Ols => {
            let st = s_mat.transpose();
            let sts = st * s_mat;
            let sty = st * base_forecasts;
            sts.full_piv_lu().solve(&sty)
        }
        ReconciliationMethod::Wls { weights } => {
            if weights.len() != m {
                return Err(Error::ShapeMismatch {
                    expected: format!("{} weights", m),
                    actual: format!("{} weights", weights.len()),
                });
            }
            if weights.iter().any(|&w| w <= 0.0) {
                return Err(Error::InvalidParameter {
                    name: "weights",
                    message: "all weights must be positive",
                });
            }

            // W^-1 S and W^-1 y_hat
            let mut winv_s = Mat::<f64>::zeros(m, n);
            let mut winv_y = Mat::<f64>::zeros(m, base_forecasts.ncols());
            for i in 0..m {
                let w_i_inv = 1.0 / weights[i];
                for j in 0..n {
                    winv_s[(i, j)] = w_i_inv * s_mat[(i, j)];
                }
                for k in 0..base_forecasts.ncols() {
                    winv_y[(i, k)] = w_i_inv * base_forecasts[(i, k)];
                }
            }

            let st = s_mat.transpose();
            let st_winv_s = st * &winv_s;
            let st_winv_y = st * &winv_y;
            st_winv_s.full_piv_lu().solve(&st_winv_y)
        }
        ReconciliationMethod::MinT { covariance } => {
            if covariance.nrows() != m || covariance.ncols() != m {
                return Err(Error::ShapeMismatch {
                    expected: format!("{}x{} covariance", m, m),
                    actual: format!("{}x{} covariance", covariance.nrows(), covariance.ncols()),
                });
            }

            // Solve Sigma * X = S and Sigma * Y = y_hat
            // Then X = Sigma^-1 S and Y = Sigma^-1 y_hat
            let lu = covariance.full_piv_lu();
            let sigmainv_s = lu.solve(s_mat);
            let sigmainv_y = lu.solve(base_forecasts);

            let st = s_mat.transpose();
            let lhs = st * &sigmainv_s;
            let rhs = st * &sigmainv_y;
            lhs.full_piv_lu().solve(&rhs)
        }
    };

    Ok(s_mat * b)
}
