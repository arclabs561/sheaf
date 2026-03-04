//! Hierarchical conformal prediction: coherent prediction intervals across tree levels.
//!
//! Demonstrates split conformal prediction on a simple 2-level hierarchy
//! (1 root + 3 leaves) where the root value equals the sum of leaves.
//!
//! The key property: after reconciliation, prediction intervals are *coherent*
//! -- the root interval is consistent with the leaf intervals, because both
//! are derived from the same reconciled forecasts.
//!
//! References:
//! - Principato et al. (2024), "Conformal Prediction for Hierarchical Data"
//! - den Hengst et al. (2025), "Hierarchical Conformal Classification"

use faer::Mat;
use sheaf::{HierarchicalConformal, ReconciliationMethod, SummingMatrix};

fn main() -> sheaf::Result<()> {
    // --- Hierarchy: 1 root (= sum of 3 leaves), so m=4 nodes, n=3 bottom-level ---
    let n_leaves = 3;
    let s = SummingMatrix::simple_star(n_leaves);
    let m = n_leaves + 1; // 4

    // --- Calibration data ---
    // Generate 50 calibration observations where:
    //   y_true[leaf_j, t] = base_j + noise_j(t)
    //   y_true[root, t]   = sum of leaves
    //
    // Base forecasts have systematic bias (e.g., model underestimates leaf 0).
    let n_calib = 50;
    let mut y_calib = Mat::<f64>::zeros(m, n_calib);
    let mut y_hat_calib = Mat::<f64>::zeros(m, n_calib);

    // Deterministic pseudo-random noise via LCG.
    let mut lcg: u64 = 42;
    let next_noise = |state: &mut u64, scale: f64| -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = ((*state >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (*state >> 11) as f64 / (1u64 << 53) as f64;
        scale * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let leaf_bases = [10.0, 20.0, 30.0];
    let leaf_bias = [2.0, -1.0, 0.5]; // systematic forecast bias per leaf
    let noise_std = 3.0;

    for t in 0..n_calib {
        let mut root_true = 0.0;
        let mut root_hat = 0.0;
        for (j, &base) in leaf_bases.iter().enumerate() {
            let noise = next_noise(&mut lcg, noise_std);
            let y_true = base + noise;
            let y_hat = y_true + leaf_bias[j] + next_noise(&mut lcg, 1.0); // biased + noisy forecast

            y_calib[(j + 1, t)] = y_true;
            y_hat_calib[(j + 1, t)] = y_hat;
            root_true += y_true;
            root_hat += y_hat;
        }
        y_calib[(0, t)] = root_true;
        y_hat_calib[(0, t)] = root_hat;
    }

    // --- Calibrate conformal predictor ---
    let alpha = 0.1; // target 90% coverage
    let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);
    cp.calibrate(&y_calib, &y_hat_calib, alpha)?;

    println!("Hierarchical Conformal Prediction");
    println!("  hierarchy: 1 root + {n_leaves} leaves (root = sum of leaves)");
    println!("  calibration samples: {n_calib}");
    println!("  target coverage: {:.0}%", (1.0 - alpha) * 100.0);
    println!("  calibrated quantile (radius): {:.4}", cp.quantile());
    println!();

    // --- Predict intervals for a test point ---
    let mut y_hat_test = Mat::<f64>::zeros(m, 1);
    y_hat_test[(1, 0)] = 12.0; // leaf 0 forecast
    y_hat_test[(2, 0)] = 19.0; // leaf 1 forecast
    y_hat_test[(3, 0)] = 31.0; // leaf 2 forecast
    y_hat_test[(0, 0)] = 62.0; // root forecast (= 12+19+31, already coherent)

    let (lower, upper) = cp.predict_intervals(&y_hat_test)?;

    println!("Test point intervals:");
    let node_names = ["root ", "leaf0", "leaf1", "leaf2"];
    println!(
        "{:>6} {:>10} {:>10} {:>10}",
        "node", "lower", "recon", "upper"
    );
    println!("{}", "-".repeat(42));
    for i in 0..m {
        let mid = (lower[(i, 0)] + upper[(i, 0)]) / 2.0;
        println!(
            "{:>6} {:>10.2} {:>10.2} {:>10.2}",
            node_names[i],
            lower[(i, 0)],
            mid,
            upper[(i, 0)]
        );
    }

    // --- Coherence check ---
    // After OLS reconciliation, the point forecast for root equals sum of leaf point forecasts.
    let recon_root = (lower[(0, 0)] + upper[(0, 0)]) / 2.0;
    let recon_leaf_sum: f64 = (1..m).map(|i| (lower[(i, 0)] + upper[(i, 0)]) / 2.0).sum();
    println!();
    println!("Coherence of reconciled point forecasts:");
    println!("  root midpoint:      {recon_root:.4}");
    println!("  sum of leaf midpts: {recon_leaf_sum:.4}");
    println!(
        "  difference:         {:.2e} (should be ~0 after OLS reconciliation)",
        (recon_root - recon_leaf_sum).abs()
    );

    // --- Coverage verification on calibration set ---
    // Check what fraction of calibration points fall inside their intervals.
    let mut covered = 0usize;
    for t in 0..n_calib {
        let mut col = Mat::<f64>::zeros(m, 1);
        for i in 0..m {
            col[(i, 0)] = y_hat_calib[(i, t)];
        }
        let (lo, hi) = cp.predict_intervals(&col)?;
        let all_in = (0..m).all(|i| y_calib[(i, t)] >= lo[(i, 0)] && y_calib[(i, t)] <= hi[(i, 0)]);
        if all_in {
            covered += 1;
        }
    }
    let empirical_coverage = covered as f64 / n_calib as f64;
    println!();
    println!("Empirical joint coverage on calibration set:");
    println!(
        "  {covered}/{n_calib} = {empirical_coverage:.2} (target >= {:.2})",
        1.0 - alpha
    );

    Ok(())
}
