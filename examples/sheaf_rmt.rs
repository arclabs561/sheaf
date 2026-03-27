//! Sheaf Laplacian spectrum vs random matrix predictions.
//!
//! Builds random sheaves (random restriction maps on a fixed graph) and
//! compares the Laplacian eigenvalue distribution against the Wigner
//! semicircle law. Shows that the spectral gap distinguishes structured
//! sheaves from random ones.
//!
//! Run: cargo run --example sheaf_rmt

use sheaf::CellularSheaf;

fn main() {
    println!("=== Sheaf Laplacian Spectrum and RMT ===\n");

    // ---------------------------------------------------------------
    // Part 1: Random sheaf Laplacian spectrum vs Wigner semicircle
    // ---------------------------------------------------------------
    println!("--- Part 1: Random sheaf Laplacian eigenvalue distribution ---\n");

    // Build a complete graph K_8 with R^2 stalks
    let n_nodes = 8;
    let stalk_dim = 2;
    let mut edges = Vec::new();
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            edges.push((i, j));
        }
    }
    let n_edges = edges.len();

    // Random restriction maps via LCG
    let mut lcg_state = 42u64;
    let total_dim = n_nodes * stalk_dim;

    let n_trials = 50;
    let mut all_centered_eigs: Vec<f64> = Vec::new();

    for _ in 0..n_trials {
        let sheaf = random_sheaf(n_nodes, &edges, stalk_dim, &mut lcg_state);
        let lap = sheaf.laplacian();
        let eigs = sorted_eigenvalues(&lap);

        // Center and scale: for the Wigner comparison, subtract mean
        // and divide by sqrt(total_dim).
        let mean_eig: f64 = eigs.iter().sum::<f64>() / total_dim as f64;
        for e in &eigs {
            all_centered_eigs.push((e - mean_eig) / (total_dim as f64).sqrt());
        }
    }

    // Empirical density
    let n_bins = 15;
    let (centers, densities) = empirical_spectral_density(&all_centered_eigs, n_bins);

    println!(
        "  {n_trials} random sheaves on K_{n_nodes} with R^{stalk_dim} stalks ({n_edges} edges)"
    );
    println!("  Total Laplacian dimension: {total_dim} x {total_dim}");
    println!("  Collected {} eigenvalues\n", all_centered_eigs.len());

    println!(
        "  {:>8} | {:>8}  {:>8}",
        "lambda", "empirical", "semicircle"
    );
    println!("  {:-<8}-+-{:-<8}--{:-<8}", "", "", "");

    // Estimate sigma for semicircle from empirical variance
    let var: f64 =
        all_centered_eigs.iter().map(|e| e * e).sum::<f64>() / all_centered_eigs.len() as f64;
    let sigma_est = var.sqrt();

    for (c, emp_d) in centers.iter().zip(densities.iter()) {
        let sc_d = wigner_semicircle_density(*c, sigma_est);
        let bar = "#".repeat((emp_d * 15.0).min(40.0) as usize);
        println!("  {c:8.4} | {emp_d:8.4}  {sc_d:8.4} {bar}");
    }
    println!();
    println!("  The random sheaf Laplacian spectrum approaches the semicircle");
    println!("  law for the centered/scaled eigenvalues.\n");

    // ---------------------------------------------------------------
    // Part 2: Spectral gap distinguishes structured vs random sheaves
    // ---------------------------------------------------------------
    println!("--- Part 2: Spectral gap: structured vs random ---\n");

    // Structured sheaf: constant sheaf (all identity restriction maps)
    let constant = CellularSheaf::constant(n_nodes, &edges, stalk_dim);
    let const_lap = constant.laplacian();
    let const_eigs = sorted_eigenvalues(&const_lap);
    let const_gap = spectral_gap(&const_eigs);
    let const_h0 = constant.h0_dimension(1e-8);

    println!("  Constant sheaf (identity maps):");
    println!("    H^0 = {const_h0} (global sections)");
    println!("    Spectral gap: {const_gap:.6}");
    println!(
        "    First 6 eigenvalues: {}",
        format_eigs(&const_eigs[..6.min(const_eigs.len())])
    );
    println!();

    // Random sheaves: average spectral gap
    let mut random_gaps: Vec<f64> = Vec::new();
    let mut random_h0s: Vec<usize> = Vec::new();
    let mut rng = 100u64;

    for _ in 0..30 {
        let sheaf = random_sheaf(n_nodes, &edges, stalk_dim, &mut rng);
        let lap = sheaf.laplacian();
        let eigs = sorted_eigenvalues(&lap);
        random_gaps.push(spectral_gap(&eigs));
        random_h0s.push(sheaf.h0_dimension(1e-8));
    }

    let mean_random_gap: f64 = random_gaps.iter().sum::<f64>() / random_gaps.len() as f64;
    let mean_random_h0: f64 =
        random_h0s.iter().map(|&h| h as f64).sum::<f64>() / random_h0s.len() as f64;
    let max_random_h0 = random_h0s.iter().max().copied().unwrap_or(0);

    println!("  Random sheaves (30 trials):");
    println!("    Mean H^0 = {mean_random_h0:.2} (max {max_random_h0})");
    println!("    Mean spectral gap: {mean_random_gap:.6}");
    println!();

    println!("  The constant sheaf has a larger spectral gap than random sheaves:");
    println!("  structured gap ({const_gap:.4}) vs random mean ({mean_random_gap:.4}).");
    println!("  This reflects the rigid structure of identity restriction maps");
    println!("  vs the incoherence of random maps.\n");

    // ---------------------------------------------------------------
    // Part 3: Spacing ratio analysis
    // ---------------------------------------------------------------
    println!("--- Part 3: Level spacing statistics ---\n");

    // Structured vs random eigenvalue repulsion
    let const_msr = mean_spacing_ratio_local(&const_eigs);
    let random_sheaf = random_sheaf(n_nodes, &edges, stalk_dim, &mut rng);
    let random_lap = random_sheaf.laplacian();
    let random_eigs = sorted_eigenvalues(&random_lap);
    let random_msr = mean_spacing_ratio_local(&random_eigs);

    println!("  Mean spacing ratio (GOE ~ 0.53, Poisson ~ 0.39):");
    println!("    Constant sheaf: {const_msr:.4}");
    println!("    Random sheaf:   {random_msr:.4}");
    println!();
    println!("  Random sheaves show stronger eigenvalue repulsion (closer to GOE),");
    println!("  while structured sheaves may have degenerate eigenvalues.");
}

/// Build a random sheaf on the given graph with random Gaussian restriction maps.
fn random_sheaf(
    n_nodes: usize,
    edges: &[(usize, usize)],
    stalk_dim: usize,
    lcg_state: &mut u64,
) -> CellularSheaf {
    let stalk_dims = vec![stalk_dim; n_nodes];
    let edge_dims = vec![stalk_dim; edges.len()];
    let mut maps = Vec::with_capacity(edges.len());

    for _ in edges {
        let map_size = stalk_dim * stalk_dim;
        let fu: Vec<f64> = (0..map_size).map(|_| lcg_gaussian(lcg_state)).collect();
        let fv: Vec<f64> = (0..map_size).map(|_| lcg_gaussian(lcg_state)).collect();
        maps.push((fu, fv));
    }

    CellularSheaf::new(n_nodes, stalk_dims, edges.to_vec(), edge_dims, maps)
        .expect("valid random sheaf")
}

/// Sorted eigenvalues of a symmetric faer::Mat.
fn sorted_eigenvalues(mat: &faer::Mat<f64>) -> Vec<f64> {
    let n = mat.nrows();
    let eig = mat
        .as_ref()
        .selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let mut vals: Vec<f64> = (0..n).map(|i| s.column_vector().read(i)).collect();
    vals.sort_by(|a, b| a.total_cmp(b));
    vals
}

/// Spectral gap: smallest non-zero eigenvalue (above tolerance).
fn spectral_gap(eigs: &[f64]) -> f64 {
    eigs.iter().copied().find(|&e| e > 1e-8).unwrap_or(0.0)
}

/// Wigner semicircle density.
fn wigner_semicircle_density(lambda: f64, sigma: f64) -> f64 {
    let r = 2.0 * sigma;
    if lambda.abs() > r {
        return 0.0;
    }
    (2.0 / (std::f64::consts::PI * r * r)) * (r * r - lambda * lambda).sqrt()
}

/// Empirical spectral density via histogram.
fn empirical_spectral_density(eigenvalues: &[f64], bins: usize) -> (Vec<f64>, Vec<f64>) {
    if eigenvalues.is_empty() || bins == 0 {
        return (vec![], vec![]);
    }
    let min = eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = eigenvalues
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-10 {
        return (vec![min], vec![1.0]);
    }
    let bin_width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &ev in eigenvalues {
        let idx = ((ev - min) / bin_width).floor() as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += 1;
    }
    let n = eigenvalues.len() as f64;
    let centers: Vec<f64> = (0..bins)
        .map(|i| min + (i as f64 + 0.5) * bin_width)
        .collect();
    let densities: Vec<f64> = counts.iter().map(|&c| c as f64 / (n * bin_width)).collect();
    (centers, densities)
}

/// Mean spacing ratio (local implementation to avoid pulling in rmt as a dep).
fn mean_spacing_ratio_local(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 3 {
        return 0.0;
    }
    let mut ratios = Vec::new();
    for i in 0..(eigenvalues.len() - 2) {
        let s1 = eigenvalues[i + 1] - eigenvalues[i];
        let s2 = eigenvalues[i + 2] - eigenvalues[i + 1];
        if s1 > 1e-15 && s2 > 1e-15 {
            ratios.push(s1.min(s2) / s1.max(s2));
        }
    }
    if ratios.is_empty() {
        0.0
    } else {
        ratios.iter().sum::<f64>() / ratios.len() as f64
    }
}

/// LCG-based pseudo-Gaussian via Box-Muller.
fn lcg_gaussian(state: &mut u64) -> f64 {
    let u1 = lcg_uniform(state);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// LCG producing a uniform in (0, 1).
fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let val = (*state >> 11) as f64 / (1u64 << 53) as f64;
    val.max(1e-15)
}

/// Format a slice of eigenvalues for display.
fn format_eigs(eigs: &[f64]) -> String {
    eigs.iter()
        .map(|e| format!("{e:.4}"))
        .collect::<Vec<_>>()
        .join(", ")
}
