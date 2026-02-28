//! Embedding clustering: kNN graph + Leiden community detection.
//!
//! Generates 250 points in 8D arranged in 4 clusters with Gaussian noise,
//! builds a kNN graph via HNSW, and runs Leiden to recover communities.
//! Prints cluster sizes and a purity metric comparing recovered communities
//! to ground-truth labels.
//!
//! Pipeline:
//!   synthetic embeddings -> vicinity (HNSW) -> kNN graph -> Leiden -> communities
//!
//! Requires: `cargo run -p sheaf --example embedding_clustering --features knn-graph`

use sheaf::community::CommunityDetection;
use sheaf::{knn_graph_with_config, KnnGraphConfig, Leiden, WeightFunction};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n_per_cluster = 60;
    let n_clusters = 4;
    let dim = 8;
    let n_total = n_per_cluster * n_clusters;
    // 10 "noise" points placed far from any cluster center.
    let n_noise = 10;

    // Cluster centers: well-separated along different axes.
    let centers: Vec<Vec<f32>> = vec![
        {
            let mut v = vec![0.0f32; dim];
            v[0] = 5.0;
            v
        },
        {
            let mut v = vec![0.0f32; dim];
            v[2] = 5.0;
            v
        },
        {
            let mut v = vec![0.0f32; dim];
            v[4] = 5.0;
            v
        },
        {
            let mut v = vec![0.0f32; dim];
            v[6] = 5.0;
            v
        },
    ];

    // Generate points with deterministic pseudo-random noise (LCG + Box-Muller).
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n_total + n_noise);
    let mut ground_truth: Vec<usize> = Vec::with_capacity(n_total + n_noise);
    let mut lcg_state: u64 = 12345;

    let next_uniform = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64
    };

    let next_normal = |state: &mut u64| -> f64 {
        let u1 = next_uniform(state).max(1e-15);
        let u2 = next_uniform(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let noise_scale = 0.4;
    for (cluster_id, center) in centers.iter().enumerate() {
        for _ in 0..n_per_cluster {
            let point: Vec<f32> = center
                .iter()
                .map(|&c| c + noise_scale * next_normal(&mut lcg_state) as f32)
                .collect();
            embeddings.push(point);
            ground_truth.push(cluster_id);
        }
    }

    // Add noise points (uniform in a large box).
    for _ in 0..n_noise {
        let point: Vec<f32> = (0..dim)
            .map(|_| (next_uniform(&mut lcg_state) * 20.0 - 10.0) as f32)
            .collect();
        embeddings.push(point);
        ground_truth.push(n_clusters); // label "noise" as its own group
    }

    let total = embeddings.len();
    println!("Generated {total} points: {n_clusters} clusters x {n_per_cluster} + {n_noise} noise, dim={dim}");

    // Build kNN graph.
    let config = KnnGraphConfig {
        k: 10,
        symmetric: true,
        weight_fn: WeightFunction::InverseDistance,
        ..Default::default()
    };
    let graph = knn_graph_with_config(&embeddings, &config)?;
    println!(
        "kNN graph: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Run Leiden.
    let leiden = Leiden::new().with_resolution(1.0);
    let labels = leiden.detect(&graph)?;

    // Community summary.
    let mut by_comm: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, &comm) in labels.iter().enumerate() {
        by_comm.entry(comm).or_default().push(idx);
    }

    println!("\nDetected {} communities:", by_comm.len());
    println!("{:>5} {:>8}", "comm", "size");
    println!("{}", "-".repeat(15));
    for (&cid, members) in &by_comm {
        println!("{:>5} {:>8}", cid, members.len());
    }

    // Purity: for each detected community, find the most common ground-truth label
    // and count how many members match it.
    let mut correct = 0usize;
    for members in by_comm.values() {
        let mut label_counts: BTreeMap<usize, usize> = BTreeMap::new();
        for &idx in members {
            *label_counts.entry(ground_truth[idx]).or_default() += 1;
        }
        let max_count = label_counts.values().max().copied().unwrap_or(0);
        correct += max_count;
    }
    let purity = correct as f64 / total as f64;
    println!("\nPurity: {correct}/{total} = {purity:.4}");
    println!("(Purity 1.0 means every detected community contains only one ground-truth cluster.)");

    Ok(())
}
