//! Distribution-distance utilities for comparing clusters as point clouds.
//!
//! This module is intentionally small and feature-gated:
//! - `rkhs` enables MMD (kernel two-sample distance).
//! - `wass` enables sliced Wasserstein (random 1D projections + exact 1D W₁).
//!
//! ## Public invariants (must never change)
//!
//! - **No silent shape coercion**: dimension mismatches return an error.
//! - **Determinism**: the `wass::sliced_wasserstein_view` implementation is deterministic
//!   (fixed seed) and intended for tests/bench baselines.
//! - **No hidden normalization**: inputs are treated as empirical measures over the given points.
//!
//! ## Swappable (can change)
//!
//! - The particular default kernel bandwidth heuristic (if we ever add one).
//! - Whether we expose more kernels/costs (keep the contract explicit).
//!
//! ## References and Future Directions
//!
//! - Schrab et al. (2021), "MMD Aggregated Two-Sample Test" -- aggregates over multiple
//!   kernel bandwidths; avoids the single-sigma choice in `rbf_sigma`. Potential extension:
//!   `mmd_aggregated` that tests over a grid of bandwidths automatically.
//! - Nguyen & Ho (2023), "Energy-Based Sliced Wasserstein Distance" -- learns informative
//!   projection directions instead of random ones; could reduce `sw_projections` needed
//! - Carlier et al. (2025), "Sharp Comparisons Between Sliced and Standard 1-Wasserstein"
//!   -- tight bounds establishing when sliced W1 is a faithful proxy for standard W1
//! - Chaibub Neto (2024), "Computationally Efficient Permutation Tests for Energy Distance
//!   or MMD" -- achieves O(n log n) vs current O(n^2) for large-scale two-sample testing

use crate::{Error, Result};

#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};

/// Configuration for distribution distances.
#[derive(Debug, Clone)]
pub struct DistributionDistanceConfig {
    /// RBF bandwidth for MMD (σ). Only used with `rkhs`.
    pub rbf_sigma: f64,
    /// Number of random projections for sliced Wasserstein. Only used with `wass`.
    pub sw_projections: usize,
}

impl Default for DistributionDistanceConfig {
    fn default() -> Self {
        Self {
            rbf_sigma: 1.0,
            sw_projections: 32,
        }
    }
}

/// Distribution distances between two point clouds.
///
/// Fields are `None` when the corresponding feature is disabled.
#[derive(Debug, Clone, Default)]
pub struct DistributionDistance {
    /// Biased MMD² with an RBF kernel (non-negative; O(n²)).
    pub mmd_rbf_biased: Option<f64>,
    /// Sliced Wasserstein distance (approximate W₁; O(k·n·d + k·n log n)).
    pub sliced_wasserstein: Option<f32>,
}

impl DistributionDistance {
    /// Compute all enabled distances for two point clouds.
    ///
    /// Contract:
    /// - Both inputs are treated as *uniform empirical measures* over their rows.
    /// - Dimension mismatch is an error (no truncation/padding).
    pub fn compute(
        x: ArrayView2<'_, f32>,
        y: ArrayView2<'_, f32>,
        cfg: &DistributionDistanceConfig,
    ) -> Result<Self> {
        let dx = x.ncols();
        let dy = y.ncols();
        if dx != dy {
            return Err(Error::DimensionMismatch {
                expected: dx,
                found: dy,
            });
        }
        if x.nrows() == 0 || y.nrows() == 0 {
            return Ok(Self::default());
        }

        let mut out = Self::default();

        #[cfg(feature = "rkhs")]
        {
            out.mmd_rbf_biased = Some(mmd_rbf_biased(x, y, cfg.rbf_sigma)?);
        }

        #[cfg(feature = "wass")]
        {
            out.sliced_wasserstein = Some(sliced_wasserstein(x, y, cfg.sw_projections)?);
        }

        Ok(out)
    }
}

#[cfg(feature = "rkhs")]
fn mmd_rbf_biased(x: ArrayView2<'_, f32>, y: ArrayView2<'_, f32>, sigma: f64) -> Result<f64> {
    if sigma <= 0.0 {
        return Err(Error::InvalidParameter {
            name: "rbf_sigma",
            message: "must be > 0",
        });
    }

    // rkhs' public API uses Vec<Vec<f64>>. This is explicit and keeps `sheaf`
    // independent of rkhs' internal ndarray integration choices.
    fn to_vecs(a: ArrayView2<'_, f32>) -> Vec<Vec<f64>> {
        a.outer_iter()
            .map(|row| row.iter().map(|v| *v as f64).collect::<Vec<f64>>())
            .collect()
    }

    let xa = to_vecs(x);
    let ya = to_vecs(y);
    let mmd = rkhs::mmd_biased(&xa, &ya, |a, b| rkhs::rbf(a, b, sigma));
    Ok(mmd)
}

#[cfg(feature = "wass")]
fn sliced_wasserstein(
    x: ArrayView2<'_, f32>,
    y: ArrayView2<'_, f32>,
    n_projections: usize,
) -> Result<f32> {
    if n_projections == 0 {
        return Ok(0.0);
    }
    // Note: the crates.io `wass` API currently accepts owned `Array2` references.
    // We keep `sheaf`'s input type as `ArrayView2` and make the conversion explicit here.
    //
    // If/when `wass` exposes a `*_view` API publicly, we can remove this allocation.
    let x_owned: Array2<f32> = x.to_owned();
    let y_owned: Array2<f32> = y.to_owned();
    Ok(wass::sliced_wasserstein(
        &x_owned,
        &y_owned,
        n_projections,
        42,
        2.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn empty_inputs_return_zeroes() {
        let x = Array2::<f32>::zeros((0, 3));
        let y = Array2::<f32>::zeros((0, 3));
        let d = DistributionDistance::compute(
            x.view(),
            y.view(),
            &DistributionDistanceConfig::default(),
        )
        .unwrap();
        assert!(d.mmd_rbf_biased.is_none() || d.mmd_rbf_biased == Some(0.0));
        assert!(d.sliced_wasserstein.is_none() || d.sliced_wasserstein == Some(0.0));
    }

    #[test]
    fn dim_mismatch_errors() {
        let x = Array2::<f32>::zeros((5, 2));
        let y = Array2::<f32>::zeros((5, 3));
        let err = DistributionDistance::compute(
            x.view(),
            y.view(),
            &DistributionDistanceConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::DimensionMismatch { .. }));
    }

    #[cfg(feature = "wass")]
    #[test]
    fn sliced_wasserstein_is_symmetric_on_simple_case() {
        let x = array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y = array![[10.0f32, 10.0], [10.5, 10.0], [10.0, 10.5]];
        let cfg = DistributionDistanceConfig {
            sw_projections: 16,
            ..Default::default()
        };
        let d_xy = DistributionDistance::compute(x.view(), y.view(), &cfg)
            .unwrap()
            .sliced_wasserstein
            .unwrap();
        let d_yx = DistributionDistance::compute(y.view(), x.view(), &cfg)
            .unwrap()
            .sliced_wasserstein
            .unwrap();
        assert!((d_xy - d_yx).abs() < 1e-4);
        assert!(d_xy > 0.0);
    }

    #[cfg(feature = "rkhs")]
    #[test]
    fn mmd_rbf_biased_is_nonnegative_and_small_for_same_distribution() {
        let x = array![[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = x.clone();
        let cfg = DistributionDistanceConfig {
            rbf_sigma: 1.0,
            ..Default::default()
        };
        let m = DistributionDistance::compute(x.view(), y.view(), &cfg)
            .unwrap()
            .mmd_rbf_biased
            .unwrap();
        assert!(m >= -1e-12);
        assert!(m < 1e-6, "expected ~0 for identical point clouds; got {m}");
    }

    #[cfg(feature = "rkhs")]
    #[test]
    fn mmd_rbf_biased_increases_for_separated_clouds() {
        let x = array![[0.0f32, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1]];
        let y = array![[10.0f32, 10.0], [10.1, 10.0], [10.0, 10.1], [10.1, 10.1]];
        let cfg = DistributionDistanceConfig {
            rbf_sigma: 1.0,
            ..Default::default()
        };
        let m = DistributionDistance::compute(x.view(), y.view(), &cfg)
            .unwrap()
            .mmd_rbf_biased
            .unwrap();
        assert!(
            m > 0.1,
            "expected noticeably positive MMD for separated clouds; got {m}"
        );
    }
}
