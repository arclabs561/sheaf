//! Compare clustering algorithms on the same synthetic data.
//!
//! Generates 300 points in 5D arranged in 4 known clusters with added noise,
//! then runs K-means, GMM, and hierarchical (Ward) clustering. Results are
//! compared using ARI, NMI, and purity from `sheaf::metrics`.

use sheaf::cluster::{Clustering, Gmm, HierarchicalClustering, Kmeans, Linkage, SoftClustering};
use sheaf::metrics::{ari, nmi, purity};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_per = 75;
    let k = 4;
    let dim = 5;
    let n_total = n_per * k;

    // Cluster centers: well-separated in 5D.
    let centers: [[f32; 5]; 4] = [
        [5.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.0, 0.0],
    ];

    // Deterministic pseudo-random noise (LCG + Box-Muller).
    let mut lcg: u64 = 98765;
    let next_normal = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((*state >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (*state >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut data: Vec<Vec<f32>> = Vec::with_capacity(n_total);
    let mut truth: Vec<usize> = Vec::with_capacity(n_total);
    let noise_scale = 1.5;

    for (cid, center) in centers.iter().enumerate() {
        for _ in 0..n_per {
            let point: Vec<f32> = center
                .iter()
                .map(|&c| c + noise_scale * next_normal(&mut lcg) as f32)
                .collect();
            data.push(point);
            truth.push(cid);
        }
    }

    println!("Clustering Comparison");
    println!("=====================");
    println!("{n_total} points, {dim}D, {k} ground-truth clusters, noise_scale={noise_scale}");
    println!();

    // -- K-means --
    let km = Kmeans::new(k).with_seed(42);
    let km_labels = km.fit_predict(&data)?;

    // -- GMM --
    let gmm = Gmm::new().with_n_components(k).with_seed(42).with_max_iter(200);
    let gmm_probs = gmm.fit_predict_proba(&data)?;
    // Hard labels from soft assignments.
    let gmm_labels: Vec<usize> = gmm_probs
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    // -- Hierarchical (Ward) --
    let hc = HierarchicalClustering::new(k).with_linkage(Linkage::Ward);
    let hc_labels = hc.fit_predict(&data)?;

    // -- Evaluate --
    struct Score {
        name: &'static str,
        ari: f64,
        nmi: f64,
        purity: f64,
    }

    let scores = [
        Score {
            name: "K-means",
            ari: ari(&km_labels, &truth),
            nmi: nmi(&km_labels, &truth),
            purity: purity(&km_labels, &truth),
        },
        Score {
            name: "GMM",
            ari: ari(&gmm_labels, &truth),
            nmi: nmi(&gmm_labels, &truth),
            purity: purity(&gmm_labels, &truth),
        },
        Score {
            name: "Hierarchical (Ward)",
            ari: ari(&hc_labels, &truth),
            nmi: nmi(&hc_labels, &truth),
            purity: purity(&hc_labels, &truth),
        },
    ];

    println!("{:<22} {:>8} {:>8} {:>8}", "algorithm", "ARI", "NMI", "purity");
    println!("{}", "-".repeat(50));
    for s in &scores {
        println!(
            "{:<22} {:>8.4} {:>8.4} {:>8.4}",
            s.name, s.ari, s.nmi, s.purity
        );
    }

    // -- GMM soft assignment summary --
    println!();
    println!("GMM soft assignment entropy (first 5 points per cluster):");
    for cid in 0..k {
        let start = cid * n_per;
        let entropies: Vec<f64> = (start..start + 5)
            .map(|i| {
                gmm_probs[i]
                    .iter()
                    .filter(|&&p| p > 1e-15)
                    .map(|&p| -p * p.ln())
                    .sum::<f64>()
                    .max(0.0) // avoid -0.0 display
            })
            .collect();
        let avg: f64 = entropies.iter().sum::<f64>() / entropies.len() as f64;
        println!("  cluster {cid}: avg entropy = {avg:.4} (0 = certain, ln({k}) = {:.4} = uniform)", (k as f64).ln());
    }

    Ok(())
}
