//! Regression tests for clustering, community detection and summarization.
#![allow(dead_code, clippy::unwrap_used)]

fn n_clusters(labels: &[usize]) -> usize {
    let mut u = labels.to_vec();
    u.sort_unstable();
    u.dedup();
    u.len()
}

// Points 0, 1, 10, 11 under single linkage merge (0,1)@1 and (2,3)@1, then @9.
// Cutting at the first merge's height used to apply both tied merges.
#[cfg(feature = "cluster")]
#[test]
fn hierarchical_fit_predict_returns_requested_k_on_symmetric_data() {
    use sheaf::cluster::{Clustering, HierarchicalClustering, Linkage};
    let data = vec![vec![0.0f32], vec![1.0], vec![10.0], vec![11.0]];
    let labels = HierarchicalClustering::new(3)
        .with_linkage(Linkage::Single)
        .fit_predict(&data)
        .unwrap();
    assert_eq!(
        n_clusters(&labels),
        3,
        "asked for 3 clusters, got {labels:?}"
    );
}

#[test]
fn dendrogram_cut_to_k_with_tied_merge_heights() {
    let mut d = sheaf::Dendrogram::new(4);
    d.add_merge(0, 1, 1.0, 2);
    d.add_merge(2, 3, 1.0, 2);
    d.add_merge(4, 5, 9.0, 4);
    let labels = d.cut_to_k(3).unwrap();
    assert_eq!(n_clusters(&labels), 3, "cut_to_k(3) gave {labels:?}");
}

#[cfg(feature = "community")]
#[test]
fn label_propagation_with_seed_is_reproducible() {
    use petgraph::graph::UnGraph;
    use sheaf::community::{CommunityDetection, LabelPropagation};
    // A ring of 30 nodes: every node starts with a 2-way tie between neighbour labels.
    let n = 30;
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
    for i in 0..n {
        let _ = g.add_edge(nodes[i], nodes[(i + 1) % n], ());
    }
    let first = LabelPropagation::new().with_seed(7).detect(&g).unwrap();
    for run in 0..20 {
        let again = LabelPropagation::new().with_seed(7).detect(&g).unwrap();
        assert_eq!(
            first, again,
            "run {run}: same seed gave a different partition"
        );
    }
}

#[cfg(feature = "knn-graph")]
#[test]
fn knn_graph_mutual_neighbours_get_one_edge() {
    use sheaf::community::knn_graph_from_embeddings;
    // Two pairs of points, each point the other's only neighbour.
    let emb = vec![
        vec![1.0f32, 0.0],
        vec![0.6, 0.8],
        vec![-1.0, 0.0],
        vec![-0.6, -0.8],
    ];
    let g = knn_graph_from_embeddings(&emb, 1).unwrap();
    let mut pairs: Vec<(usize, usize)> = g
        .raw_edges()
        .iter()
        .map(|e| {
            let (a, b) = (e.source().index(), e.target().index());
            (a.min(b), a.max(b))
        })
        .collect();
    let total = pairs.len();
    pairs.sort();
    pairs.dedup();
    assert_eq!(
        total,
        pairs.len(),
        "parallel edges in kNN graph: {} edges for {} pairs",
        total,
        pairs.len()
    );
}

#[cfg(all(feature = "cluster", feature = "spectral"))]
#[test]
fn spectral_fit_with_precomputed_affinity_returns_err_not_panic() {
    use sheaf::cluster::{AffinityType, SpectralClustering};
    let pts = ndarray::array![[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]];
    let r = std::panic::catch_unwind(|| {
        SpectralClustering::new(2)
            .affinity(AffinityType::Precomputed)
            .fit(&pts)
            .map(|_| ())
    });
    assert!(
        r.is_ok(),
        "fit() with AffinityType::Precomputed panicked (fit returns Result)"
    );
}

#[cfg(feature = "cluster")]
#[test]
fn it_dendrogram_k_zero_is_an_error_not_a_panic() {
    use sheaf::cluster::{Clustering, ItDendrogram};
    let data = vec![vec![0.0f32], vec![1.0], vec![2.0]];
    let r = std::panic::catch_unwind(|| ItDendrogram::new(0).fit_predict(&data));
    assert!(r.is_ok(), "k = 0 panicked");
}

/// Two blobs 100 apart. Every point links to its nearest denser point, so the
/// in-tree has one root and one long edge between the blobs; cutting the single
/// longest edge must separate them.
#[cfg(feature = "cluster")]
fn two_distant_blobs() -> Vec<Vec<f32>> {
    let mut data = Vec::new();
    for i in 0..20 {
        let a = i as f32 * 0.7;
        data.push(vec![
            a.cos() * (0.2 + 0.03 * i as f32),
            a.sin() * (0.2 + 0.02 * i as f32),
        ]);
        data.push(vec![
            100.0 + a.sin() * (0.25 + 0.01 * i as f32),
            a.cos() * (0.3 + 0.02 * i as f32),
        ]);
    }
    data
}

#[cfg(feature = "cluster")]
#[test]
fn it_dendrogram_with_n_clusters_separates_two_distant_blobs() {
    use sheaf::cluster::{Clustering, ItDendrogram};
    let data = two_distant_blobs();
    let l = ItDendrogram::new(5)
        .with_n_clusters(2)
        .fit_predict(&data)
        .unwrap();
    assert_eq!(n_clusters(&l), 2, "labels {l:?}");
    let left: std::collections::HashSet<usize> = l.iter().step_by(2).copied().collect();
    let right: std::collections::HashSet<usize> = l.iter().skip(1).step_by(2).copied().collect();
    assert_eq!(left.len(), 1, "left blob split: {l:?}");
    assert_eq!(right.len(), 1, "right blob split: {l:?}");
    assert!(left.is_disjoint(&right), "labels {l:?}");
}

/// Without `with_n_clusters` there is no pruning: a single density maximum
/// gives a single cluster, as documented.
#[cfg(feature = "cluster")]
#[test]
fn it_dendrogram_without_n_clusters_does_not_prune() {
    use sheaf::cluster::{Clustering, ItDendrogram};
    let l = ItDendrogram::new(5)
        .fit_predict(&two_distant_blobs())
        .unwrap();
    assert_eq!(n_clusters(&l), 1, "labels {l:?}");
}

#[cfg(feature = "cluster")]
#[test]
fn it_dendrogram_with_n_clusters_rejects_zero_and_more_than_n() {
    use sheaf::cluster::{Clustering, ItDendrogram};
    use sheaf::Error;
    let data = vec![vec![0.0f32], vec![1.0], vec![5.0]];
    assert!(matches!(
        ItDendrogram::new(1).with_n_clusters(0).fit_predict(&data),
        Err(Error::InvalidClusterCount { .. })
    ));
    assert!(matches!(
        ItDendrogram::new(1).with_n_clusters(4).fit_predict(&data),
        Err(Error::InvalidClusterCount { .. })
    ));
    let l = ItDendrogram::new(1)
        .with_n_clusters(3)
        .fit_predict(&data)
        .unwrap();
    assert_eq!(n_clusters(&l), 3, "labels {l:?}");
}

/// Initial means drawn with replacement can put two components on one point,
/// where they stay identical and every label collapses to one cluster.
#[cfg(feature = "cluster")]
#[test]
fn gmm_two_obvious_clusters_for_every_seed() {
    use sheaf::cluster::{Clustering, Gmm};
    let data = vec![
        vec![0.0f32, 0.0],
        vec![0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 10.1],
    ];
    let mut bad = Vec::new();
    for seed in 0..20u64 {
        let l = Gmm::new()
            .with_n_components(2)
            .with_seed(seed)
            .fit_predict(&data)
            .unwrap();
        if !(l[0] == l[1] && l[2] == l[3] && l[0] != l[2]) {
            bad.push((seed, l));
        }
    }
    assert!(bad.is_empty(), "seeds with collapsed components: {bad:?}");
}

#[cfg(feature = "summarize")]
#[test]
fn concat_summarizer_truncates_non_ascii_without_panic() {
    use sheaf::summarize::{ConcatSummarizer, Summarizer};
    let s = ConcatSummarizer::new().with_max_len(7);
    let a = "café".to_string();
    let b = "über".to_string();
    let r = std::panic::catch_unwind(|| s.summarize(&[&a, &b]));
    assert!(r.is_ok(), "panicked on UTF-8 text");
}
