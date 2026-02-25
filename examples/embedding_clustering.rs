use parti::community::CommunityDetection;
use parti::{knn_graph_from_embeddings, KnnGraphConfig, Leiden, WeightFunction};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Minimal end-to-end: embeddings -> kNN graph -> Leiden communities.
    //
    // This example is gated behind `required-features = ["knn-graph"]` in `Cargo.toml`.
    // It intentionally stays small: it exists primarily to validate that the integration
    // path builds and runs.

    // Two obvious clusters in 2D.
    let embeddings: Vec<Vec<f32>> = vec![
        // Cluster A (near (0,0))
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![0.1, 0.1],
        // Cluster B (near (10,10))
        vec![10.0, 10.0],
        vec![10.1, 10.0],
        vec![10.0, 10.1],
        vec![10.1, 10.1],
    ];

    // Build kNN graph (k=3). This uses jin (HNSW) under the hood.
    let _graph_fast = knn_graph_from_embeddings(&embeddings, 3)?;

    // Also show the configurable variant (mostly to keep the public API exercised).
    // Note: we keep the config conservative because this is just a smoke example.
    let config = KnnGraphConfig {
        k: 3,
        symmetric: true,
        weight_fn: WeightFunction::InverseDistance,
        ..Default::default()
    };
    let graph = parti::knn_graph_with_config(&embeddings, &config)?;

    // Detect communities.
    let leiden = Leiden::new().with_resolution(1.0);
    let labels = leiden.detect(&graph)?;

    // Print a small summary: community id -> indices.
    // `labels` is `Vec<usize>` in this crate; keep the key type consistent.
    let mut by_comm: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, comm) in labels.iter().enumerate() {
        by_comm.entry(*comm).or_default().push(idx);
    }

    println!(
        "n_nodes={} n_edges={}",
        graph.node_count(),
        graph.edge_count()
    );
    println!("communities={}", by_comm.len());
    for (cid, ids) in by_comm {
        println!("  community {}: {:?}", cid, ids);
    }

    Ok(())
}
