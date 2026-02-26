//! Hierarchical forecast reconciliation: fix incoherent base forecasts.
//!
//! Builds a product hierarchy:
//!
//! ```text
//!   Total
//!   ├── Electronics
//!   │   ├── Phones
//!   │   └── Laptops
//!   └── Clothing
//!       ├── Shirts
//!       └── Pants
//! ```
//!
//! Base forecasts are generated so that leaf values do *not* sum to their
//! parent -- the incoherence that reconciliation corrects. OLS and WLS
//! reconciliation are applied, and the output shows how each method
//! adjusts forecasts to satisfy the summing constraints.

use faer::Mat;
use parti::{reconcile, ReconciliationMethod, SummingMatrix};

fn main() -> parti::Result<()> {
    // -- Build the summing matrix S (6 nodes x 4 leaves) --
    //
    // Row order: Total, Electronics, Clothing, Phones, Laptops, Shirts, Pants
    // Leaf order (columns): Phones, Laptops, Shirts, Pants
    let m = 7; // total nodes
    let n = 4; // leaves
    let mut s_data = Mat::<f64>::zeros(m, n);

    // Total = Phones + Laptops + Shirts + Pants (all leaves)
    for j in 0..n {
        s_data[(0, j)] = 1.0;
    }
    // Electronics = Phones + Laptops
    s_data[(1, 0)] = 1.0;
    s_data[(1, 1)] = 1.0;
    // Clothing = Shirts + Pants
    s_data[(2, 2)] = 1.0;
    s_data[(2, 3)] = 1.0;
    // Identity block for leaves
    s_data[(3, 0)] = 1.0; // Phones
    s_data[(4, 1)] = 1.0; // Laptops
    s_data[(5, 2)] = 1.0; // Shirts
    s_data[(6, 3)] = 1.0; // Pants

    let s = SummingMatrix::new(s_data);

    // -- Incoherent base forecasts --
    // Leaf forecasts that intentionally do not sum to their parents.
    let mut base = Mat::<f64>::zeros(m, 1);
    base[(0, 0)] = 250.0; // Total (but leaves sum to 230)
    base[(1, 0)] = 120.0; // Electronics (but Phones+Laptops = 130)
    base[(2, 0)] = 90.0; // Clothing (but Shirts+Pants = 100)
    base[(3, 0)] = 80.0; // Phones
    base[(4, 0)] = 50.0; // Laptops
    base[(5, 0)] = 55.0; // Shirts
    base[(6, 0)] = 45.0; // Pants

    let names = ["Total", "Electronics", "Clothing", "Phones", "Laptops", "Shirts", "Pants"];

    println!("Hierarchical Forecast Reconciliation");
    println!("====================================");
    println!();
    print_column("Base (incoherent)", &names, &base);
    print_coherence_check("Base", &names, &base);

    // -- OLS reconciliation --
    let recon_ols = reconcile(&s, &base, ReconciliationMethod::Ols)?;
    println!();
    print_column("OLS reconciled", &names, &recon_ols);
    print_coherence_check("OLS", &names, &recon_ols);

    // -- WLS reconciliation (weight by inverse variance proxy) --
    // Use base forecast magnitude as a variance proxy: larger series are noisier.
    let weights: Vec<f64> = (0..m).map(|i| base[(i, 0)].abs().max(1.0)).collect();
    let recon_wls = reconcile(
        &s,
        &base,
        ReconciliationMethod::Wls {
            weights: weights.clone(),
        },
    )?;
    println!();
    print_column("WLS reconciled", &names, &recon_wls);
    print_coherence_check("WLS", &names, &recon_wls);

    // -- Residuals: how much each forecast shifted --
    println!();
    println!("Residuals (reconciled - base):");
    println!("{:>13} {:>10} {:>10}", "node", "OLS", "WLS");
    println!("{}", "-".repeat(35));
    for i in 0..m {
        let d_ols = recon_ols[(i, 0)] - base[(i, 0)];
        let d_wls = recon_wls[(i, 0)] - base[(i, 0)];
        println!("{:>13} {:>+10.2} {:>+10.2}", names[i], d_ols, d_wls);
    }

    // -- Verify coherence: leaves sum to parents --
    println!();
    println!("Coherence verification (OLS):");
    let leaf_sum_ols: f64 = (3..7).map(|i| recon_ols[(i, 0)]).sum();
    let elec_sum_ols = recon_ols[(3, 0)] + recon_ols[(4, 0)];
    let cloth_sum_ols = recon_ols[(5, 0)] + recon_ols[(6, 0)];
    println!(
        "  Phones+Laptops+Shirts+Pants = {:.4}, Total = {:.4}, diff = {:.2e}",
        leaf_sum_ols,
        recon_ols[(0, 0)],
        (leaf_sum_ols - recon_ols[(0, 0)]).abs()
    );
    println!(
        "  Phones+Laptops              = {:.4}, Electronics = {:.4}, diff = {:.2e}",
        elec_sum_ols,
        recon_ols[(1, 0)],
        (elec_sum_ols - recon_ols[(1, 0)]).abs()
    );
    println!(
        "  Shirts+Pants                = {:.4}, Clothing = {:.4}, diff = {:.2e}",
        cloth_sum_ols,
        recon_ols[(2, 0)],
        (cloth_sum_ols - recon_ols[(2, 0)]).abs()
    );

    Ok(())
}

fn print_column(label: &str, names: &[&str], mat: &Mat<f64>) {
    println!("{label}:");
    println!("{:>13} {:>10}", "node", "forecast");
    println!("{}", "-".repeat(25));
    for (i, name) in names.iter().enumerate() {
        println!("{:>13} {:>10.2}", name, mat[(i, 0)]);
    }
}

fn print_coherence_check(label: &str, names: &[&str], mat: &Mat<f64>) {
    let leaf_sum: f64 = (3..7).map(|i| mat[(i, 0)]).sum();
    let diff = (mat[(0, 0)] - leaf_sum).abs();
    let _ = names;
    println!(
        "  {label} coherence gap (Total vs leaf sum): {diff:.2e}{}",
        if diff < 1e-8 { " [coherent]" } else { " [INCOHERENT]" }
    );
}
