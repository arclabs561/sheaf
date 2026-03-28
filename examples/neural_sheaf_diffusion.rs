//! Neural Sheaf Diffusion: gradient-based optimization of restriction maps.
//!
//! Demonstrates learning optimal sheaf structure from node features.
//! The loss function encourages smooth signals over the sheaf: nodes
//! with similar features should have low sheaf Laplacian energy.
//!
//! Training loop:
//! 1. Build restriction maps from current parameters
//! 2. Compute sheaf Laplacian
//! 3. Compute energy: x^T L x (smoothness of node features on the sheaf)
//! 4. Estimate gradients via finite differences
//! 5. Update parameters with SGD
//!
//! Reference: Bodnar et al., "Neural Sheaf Diffusion", ICML 2022.

use sheaf::learnable_sheaf::{LearnableSheaf, RestrictionFamily};

#[allow(clippy::expect_used)]
fn main() {
    // Graph: pentagon with cross-links (Petersen-like subgraph).
    let num_nodes = 5;
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)];
    let stalk_dim = 2;

    // Node features: 2D positions arranged in a circle.
    let features: Vec<Vec<f64>> = (0..num_nodes)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_nodes as f64;
            vec![angle.cos(), angle.sin()]
        })
        .collect();

    // Flatten features into a signal vector (node 0 dim 0, node 0 dim 1, node 1 dim 0, ...).
    let signal: Vec<f64> = features.iter().flat_map(|f| f.iter().copied()).collect();

    let mut sheaf = LearnableSheaf::new(num_nodes, stalk_dim, edges, RestrictionFamily::Diagonal)
        .expect("valid sheaf");
    sheaf.init_random(42);

    let lr = 0.01;
    let eps = 1e-5;
    let steps = 200;

    println!("Training neural sheaf diffusion");
    println!(
        "  nodes: {num_nodes}, stalk_dim: {stalk_dim}, params: {}",
        sheaf.num_params()
    );
    println!("  family: Diagonal, lr: {lr}, steps: {steps}");
    println!();

    let initial_energy = sheaf_energy(&sheaf, &signal);
    println!("Step   0: energy = {initial_energy:.6}");

    for step in 1..=steps {
        // Finite-difference gradient estimation.
        let base_energy = sheaf_energy(&sheaf, &signal);
        let n_params = sheaf.num_params();
        let mut grad = vec![0.0; n_params];

        for (i, g) in grad.iter_mut().enumerate().take(n_params) {
            sheaf.params_mut()[i] += eps;
            let e_plus = sheaf_energy(&sheaf, &signal);
            sheaf.params_mut()[i] -= eps;
            *g = (e_plus - base_energy) / eps;
        }

        // SGD update (minimize energy = maximize smoothness).
        for (i, g) in grad.iter().enumerate().take(n_params) {
            sheaf.params_mut()[i] -= lr * g;
        }

        if step % 50 == 0 || step == 1 {
            let energy = sheaf_energy(&sheaf, &signal);
            println!("Step {step:>3}: energy = {energy:.6}");
        }
    }

    let final_energy = sheaf_energy(&sheaf, &signal);
    println!();
    println!("Initial energy: {initial_energy:.6}");
    println!("Final energy:   {final_energy:.6}");
    println!(
        "Reduction:      {:.1}%",
        (1.0 - final_energy / initial_energy) * 100.0
    );

    // Show H0 dimension (global sections / kernel dimension).
    let h0 = sheaf.h0_dimension(1e-6);
    println!("H0 dimension:   {h0} (measures global consistency)");

    // Verify the restriction maps are reasonable.
    let maps = sheaf.build_maps();
    println!();
    println!("Learned restriction maps (diagonal entries):");
    for (i, (src, tgt)) in maps.iter().enumerate() {
        let src_diag: Vec<f64> = (0..stalk_dim).map(|d| src[d * stalk_dim + d]).collect();
        let tgt_diag: Vec<f64> = (0..stalk_dim).map(|d| tgt[d * stalk_dim + d]).collect();
        println!("  edge {i}: src={src_diag:.3?}, tgt={tgt_diag:.3?}");
    }
}

/// Compute x^T L x -- the sheaf Laplacian quadratic form (energy/smoothness).
fn sheaf_energy(sheaf: &LearnableSheaf, signal: &[f64]) -> f64 {
    let lap = sheaf.laplacian();
    let n = lap.nrows();
    assert_eq!(
        signal.len(),
        n,
        "signal length must match Laplacian dimension"
    );

    let mut energy = 0.0;
    for i in 0..n {
        let mut lx_i = 0.0;
        for j in 0..n {
            lx_i += lap[(i, j)] * signal[j];
        }
        energy += signal[i] * lx_i;
    }
    energy
}
