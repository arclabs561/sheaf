#[cfg(test)]
mod tests {
    use crate::hierarchy::HierarchicalConformal;
    use crate::reconciliation::{reconcile, ReconciliationMethod, SummingMatrix};
    use crate::Result;
    use faer::Mat;

    #[test]
    fn test_reconciliation_ols_star() -> Result<()> {
        // Star hierarchy: 1 root, 2 leaves
        // S = [1 1] (root)
        //     [1 0] (leaf 1)
        //     [0 1] (leaf 2)
        let s = SummingMatrix::simple_star(2);

        // Base forecasts: root=3, leaf1=1, leaf2=1
        // (Not coherent: 1+1 = 2 != 3)
        let mut y_hat = Mat::<f64>::zeros(3, 1);
        y_hat[(0, 0)] = 3.0;
        y_hat[(1, 0)] = 1.0;
        y_hat[(2, 0)] = 1.0;

        let y_tilde = reconcile(&s, &y_hat, ReconciliationMethod::Ols)?;

        assert!((y_tilde[(0, 0)] - 2.6666666666666665).abs() < 1e-10);
        assert!((y_tilde[(1, 0)] - 1.3333333333333333).abs() < 1e-10);
        assert!((y_tilde[(2, 0)] - 1.3333333333333333).abs() < 1e-10);

        // Check coherence: root = sum(leaves)
        assert!((y_tilde[(0, 0)] - (y_tilde[(1, 0)] + y_tilde[(2, 0)])).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn test_conformal_calibration() -> Result<()> {
        let s = SummingMatrix::simple_star(2);
        let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);

        // 10 calibration points
        let y_calib = Mat::<f64>::zeros(3, 10); // True values (all zero for simplicity)

        // Forecasts with known errors
        // Error = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] (Euclidean norms)
        let mut y_hat_calib = Mat::<f64>::zeros(3, 10);
        for j in 0..10 {
            y_hat_calib[(1, j)] = (j + 1) as f64; // Error in leaf 1
        }

        // alpha = 0.2 (80% coverage)
        // n=10, (n+1)*(1-alpha) = 11 * 0.8 = 8.8 -> 9th score
        cp.calibrate(&y_calib, &y_hat_calib, 0.2)?;

        // Reconciled scores will be slightly different from base errors,
        // but they should be monotonic with j.
        assert!(cp.quantile() > 0.0);
        Ok(())
    }

    fn calibration_with_errors(n: usize) -> (Mat<f64>, Mat<f64>) {
        let y_calib = Mat::<f64>::zeros(3, n);
        let mut y_hat_calib = Mat::<f64>::zeros(3, n);
        for j in 0..n {
            y_hat_calib[(1, j)] = (j + 1) as f64;
        }
        (y_calib, y_hat_calib)
    }

    #[test]
    fn conformal_quantile_is_infinite_when_rank_exceeds_n() -> Result<()> {
        // Split conformal takes the ceil((n + 1)(1 - alpha))-th smallest score.
        // n = 5, alpha = 0.1: ceil(6 * 0.9) = 6 > 5, so no finite score
        // guarantees 90% coverage and the radius must be unbounded.
        let (y_calib, y_hat_calib) = calibration_with_errors(5);
        let mut cp =
            HierarchicalConformal::new(SummingMatrix::simple_star(2), ReconciliationMethod::Ols);
        cp.calibrate(&y_calib, &y_hat_calib, 0.1)?;
        assert_eq!(cp.quantile(), f64::INFINITY);

        let (lower, upper) = cp.predict_intervals(&y_hat_calib)?;
        assert_eq!(lower[(0, 0)], f64::NEG_INFINITY);
        assert_eq!(upper[(0, 0)], f64::INFINITY);
        Ok(())
    }

    #[test]
    fn conformal_quantile_takes_finite_rank_when_n_is_large_enough() -> Result<()> {
        // OLS reconciliation is linear, so score j is c * (j + 1) for a fixed c.
        let (y_calib, y_hat_calib) = calibration_with_errors(9);
        let mut cp =
            HierarchicalConformal::new(SummingMatrix::simple_star(2), ReconciliationMethod::Ols);
        // n = 9, alpha = 0.1: ceil(10 * 0.9) = 9, the largest score (9c).
        cp.calibrate(&y_calib, &y_hat_calib, 0.1)?;
        let q_90 = cp.quantile();
        // alpha = 0.5: ceil(10 * 0.5) = 5, the 5th-smallest score (5c).
        cp.calibrate(&y_calib, &y_hat_calib, 0.5)?;
        let q_50 = cp.quantile();
        assert!(q_90.is_finite() && q_50 > 0.0);
        assert!((q_90 / q_50 - 9.0 / 5.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn conformal_rejects_alpha_outside_unit_interval_and_empty_calibration() {
        let (y_calib, y_hat_calib) = calibration_with_errors(5);
        for alpha in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            let mut cp = HierarchicalConformal::new(
                SummingMatrix::simple_star(2),
                ReconciliationMethod::Ols,
            );
            assert!(
                cp.calibrate(&y_calib, &y_hat_calib, alpha).is_err(),
                "alpha = {alpha} must be rejected"
            );
        }

        let (y_empty, y_hat_empty) = calibration_with_errors(0);
        let mut cp =
            HierarchicalConformal::new(SummingMatrix::simple_star(2), ReconciliationMethod::Ols);
        assert!(cp.calibrate(&y_empty, &y_hat_empty, 0.1).is_err());
    }

    #[test]
    fn conformal_calibration_sorts_nan_scores_without_panicking() -> Result<()> {
        // A NaN truth value gives a NaN score. total_cmp sorts it after every
        // number, so the 5th-smallest of 9 scores is still the finite 5c.
        let (mut y_calib, y_hat_calib) = calibration_with_errors(9);
        y_calib[(0, 8)] = f64::NAN;
        let mut cp =
            HierarchicalConformal::new(SummingMatrix::simple_star(2), ReconciliationMethod::Ols);
        cp.calibrate(&y_calib, &y_hat_calib, 0.5)?;
        assert!(cp.quantile().is_finite());
        Ok(())
    }

    #[test]
    fn test_reconciliation_mint() -> Result<()> {
        let s = SummingMatrix::simple_star(2);
        let m = s.m();

        // Base forecasts: root=3, leaf1=1, leaf2=1
        let mut y_hat = Mat::<f64>::zeros(m, 1);
        y_hat[(0, 0)] = 3.0;
        y_hat[(1, 0)] = 1.0;
        y_hat[(2, 0)] = 1.0;

        // Use identity as covariance (should match OLS)
        let covariance = Mat::<f64>::identity(m, m);

        let y_tilde = reconcile(&s, &y_hat, ReconciliationMethod::MinT { covariance })?;

        assert!((y_tilde[(0, 0)] - 2.6666666666666665).abs() < 1e-10);
        assert!((y_tilde[(0, 0)] - (y_tilde[(1, 0)] + y_tilde[(2, 0)])).abs() < 1e-10);
        Ok(())
    }
}
