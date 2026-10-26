//! Community detection on Zachary's karate club, compared with networkx.
//!
//! Edges are `networkx.karate_club_graph()` (78 edges, unweighted). Over seeds
//! 0..20, `networkx.algorithms.community.louvain_communities` reaches
//! modularity between 0.3952 and 0.4198; 0.4198 is the known optimum for this
//! graph. Modularity is computed here from its definition, independent of the
//! crate under test.
//!
//! With graphops before 0.5.2 both returned a 7-community partition with
//! modularity 0.335: aggregation counted a super-node's self-loop as a link to
//! its own community, so nothing moved after the first level.
#![cfg(feature = "community")]
#![allow(clippy::unwrap_used)]

use petgraph::graph::UnGraph;
use sheaf::{CommunityDetection, Louvain, LouvainConnected};

const KARATE_EDGES: [(u32, u32); 78] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 10),
    (0, 11),
    (0, 12),
    (0, 13),
    (0, 17),
    (0, 19),
    (0, 21),
    (0, 31),
    (1, 2),
    (1, 3),
    (1, 7),
    (1, 13),
    (1, 17),
    (1, 19),
    (1, 21),
    (1, 30),
    (2, 3),
    (2, 7),
    (2, 8),
    (2, 9),
    (2, 13),
    (2, 27),
    (2, 28),
    (2, 32),
    (3, 7),
    (3, 12),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 6),
    (5, 10),
    (5, 16),
    (6, 16),
    (8, 30),
    (8, 32),
    (8, 33),
    (9, 33),
    (13, 33),
    (14, 32),
    (14, 33),
    (15, 32),
    (15, 33),
    (18, 32),
    (18, 33),
    (19, 33),
    (20, 32),
    (20, 33),
    (22, 32),
    (22, 33),
    (23, 25),
    (23, 27),
    (23, 29),
    (23, 32),
    (23, 33),
    (24, 25),
    (24, 27),
    (24, 31),
    (25, 31),
    (26, 29),
    (26, 33),
    (27, 33),
    (28, 31),
    (28, 33),
    (29, 32),
    (29, 33),
    (30, 32),
    (30, 33),
    (31, 32),
    (31, 33),
    (32, 33),
];

/// Newman-Girvan modularity: Q = sum_c [L_c / m - (d_c / 2m)^2].
fn modularity(labels: &[usize]) -> f64 {
    let m = KARATE_EDGES.len() as f64;
    let n_comm = labels.iter().max().map_or(0, |&c| c + 1);
    let mut internal = vec![0.0; n_comm];
    let mut degree = vec![0.0; n_comm];
    for &(u, v) in &KARATE_EDGES {
        let (cu, cv) = (labels[u as usize], labels[v as usize]);
        degree[cu] += 1.0;
        degree[cv] += 1.0;
        if cu == cv {
            internal[cu] += 1.0;
        }
    }
    (0..n_comm)
        .map(|c| internal[c] / m - (degree[c] / (2.0 * m)).powi(2))
        .sum()
}

fn karate() -> UnGraph<(), ()> {
    UnGraph::from_edges(KARATE_EDGES)
}

fn assert_in_networkx_range(name: &str, labels: &[usize]) {
    assert_eq!(labels.len(), 34);
    let q = modularity(labels);
    assert!(
        (0.38..=0.4198 + 1e-4).contains(&q),
        "{name}: modularity {q:.4}, networkx range 0.3952..0.4198, labels {labels:?}"
    );
}

#[test]
fn louvain_connected_karate_modularity_is_in_networkx_range() {
    let g = karate();
    for seed in 0..5 {
        let labels = LouvainConnected::new().with_seed(seed).detect(&g).unwrap();
        assert_in_networkx_range(&format!("LouvainConnected seed {seed}"), &labels);
    }
}

#[test]
fn louvain_karate_modularity_is_in_networkx_range() {
    let g = karate();
    assert_in_networkx_range("Louvain", &Louvain::new().detect(&g).unwrap());
}
