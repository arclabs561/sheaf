//! Spectral clustering via sheaf Laplacian eigenvectors + k-means.
//!
//! Demonstrates that sheaf-based spectral clustering captures community
//! structure that standard spectral clustering (trivial sheaf) can miss.
//!
//! # Setup
//!
//! Three communities of 4 nodes each, connected by inter-community edges
//! that form a near-complete bipartite bridge. The topology alone is
//! ambiguous -- a standard spectral cut might split incorrectly.
//!
//! Within-community edges use identity restriction maps (consistent "reference
//! frame"). Between-community edges use rotation maps (inconsistent). The sheaf
//! Laplacian's smallest eigenvectors encode this inconsistency, giving k-means
//! clean separation even when the graph Laplacian does not.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example sheaf_clustering --features cluster
//! ```

use clump::Kmeans;
use sheaf::CellularSheaf;

/// Number of communities and nodes per community.
const K: usize = 3;
const NODES_PER_COMMUNITY: usize = 4;
const N: usize = K * NODES_PER_COMMUNITY;

fn main() {
    println!("=== Sheaf Spectral Clustering ===\n");

    // -- Build the graph --
    // Three cliques of 4 nodes with inter-clique bridges.
    let (edges, intra_mask) = build_graph();

    println!("Graph: {N} nodes, {} edges", edges.len());
    println!(
        "  intra-community: {}  inter-community: {}\n",
        intra_mask.iter().filter(|&&b| b).count(),
        intra_mask.iter().filter(|&&b| !b).count(),
    );

    // -- Sheaf with rotation maps on inter-community edges --
    let sheaf = build_sheaf(&edges, &intra_mask);
    let sheaf_labels = spectral_cluster(&sheaf, K);

    println!("Sheaf-based clustering (rotation maps on bridges):");
    print_labels(&sheaf_labels);
    let sheaf_acc = cluster_accuracy(&sheaf_labels);
    println!("  accuracy: {:.0}%\n", sheaf_acc * 100.0);

    // -- Trivial sheaf (= standard spectral clustering) for comparison --
    let trivial = CellularSheaf::trivial(N, &edges);
    let trivial_labels = spectral_cluster(&trivial, K);

    println!("Standard spectral clustering (trivial sheaf):");
    print_labels(&trivial_labels);
    let trivial_acc = cluster_accuracy(&trivial_labels);
    println!("  accuracy: {:.0}%\n", trivial_acc * 100.0);

    // -- Summary --
    println!("--- Summary ---\n");
    println!(
        "Sheaf clustering accuracy:    {:.0}% (rotation maps encode community frames)",
        sheaf_acc * 100.0
    );
    println!(
        "Standard clustering accuracy:  {:.0}% (topology only, no edge structure)",
        trivial_acc * 100.0
    );
    if sheaf_acc > trivial_acc {
        println!("\nSheaf Laplacian eigenvectors provide richer features for k-means.");
    }

    assert!(
        sheaf_acc >= 1.0,
        "Sheaf clustering should perfectly separate communities"
    );
}

/// Build a graph with 3 cliques of 4 nodes and dense inter-clique bridges.
///
/// Returns `(edges, intra_mask)` where `intra_mask[i]` is true iff edge `i`
/// is within a community.
fn build_graph() -> (Vec<(usize, usize)>, Vec<bool>) {
    let mut edges = Vec::new();
    let mut intra = Vec::new();

    // Intra-community: complete subgraphs (cliques).
    for c in 0..K {
        let base = c * NODES_PER_COMMUNITY;
        for i in 0..NODES_PER_COMMUNITY {
            for j in (i + 1)..NODES_PER_COMMUNITY {
                edges.push((base + i, base + j));
                intra.push(true);
            }
        }
    }

    // Inter-community: dense bridges so topology alone is ambiguous.
    // Each pair of adjacent communities shares 4 cross-edges (half the
    // boundary nodes connect to half in the other community). This gives
    // each boundary node as many inter-community neighbors as
    // intra-community ones, making a pure graph-Laplacian cut unreliable.
    let bridges = [
        // Community 0 <-> 1
        (2, 4),
        (2, 5),
        (3, 4),
        (3, 5),
        // Community 1 <-> 2
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
        // Community 0 <-> 2
        (0, 10),
        (0, 11),
        (1, 10),
        (1, 11),
    ];
    for &(u, v) in &bridges {
        edges.push((u, v));
        intra.push(false);
    }

    (edges, intra)
}

/// Build a cellular sheaf where intra-community edges have identity maps
/// and inter-community edges have rotation maps.
fn build_sheaf(edges: &[(usize, usize)], intra_mask: &[bool]) -> CellularSheaf {
    let stalk_dim = 2; // R^2 stalks
    let stalk_dims = vec![stalk_dim; N];
    let edge_dims = vec![stalk_dim; edges.len()];

    // Column-major 2x2 identity.
    let eye = vec![1.0, 0.0, 0.0, 1.0];

    // 90-degree rotation (column-major): [[0, -1], [1, 0]].
    let rot90 = vec![0.0, 1.0, -1.0, 0.0];

    let restriction_maps: Vec<(Vec<f64>, Vec<f64>)> = intra_mask
        .iter()
        .map(|&is_intra| {
            if is_intra {
                // Consistent: both sides see the same reference frame.
                (eye.clone(), eye.clone())
            } else {
                // Inconsistent: the target community's "reference frame" is rotated.
                (eye.clone(), rot90.clone())
            }
        })
        .collect();

    CellularSheaf::new(N, stalk_dims, edges.to_vec(), edge_dims, restriction_maps)
        .expect("valid sheaf")
}

/// Spectral clustering: extract k eigenvectors from the sheaf Laplacian,
/// aggregate per node, then run k-means.
fn spectral_cluster(sheaf: &CellularSheaf, k: usize) -> Vec<usize> {
    let lap = sheaf.laplacian();
    let n = lap.nrows();

    // Eigendecomposition.
    let eig = lap
        .as_ref()
        .selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let u = eig.u();

    // Sort eigenvalue indices by value.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        s.column_vector()
            .read(a)
            .total_cmp(&s.column_vector().read(b))
    });

    // Skip near-zero eigenvalues, take k smallest non-zero.
    let nonzero_start = order
        .iter()
        .position(|&i| s.column_vector().read(i) > 1e-8)
        .unwrap_or(n);
    let selected: Vec<usize> = order[nonzero_start..].iter().copied().take(k).collect();

    // Build per-node features by averaging stalk rows for each node.
    let stalk_dims = sheaf.stalk_dims();
    let mut offset = 0;
    let mut node_features: Vec<Vec<f32>> = Vec::with_capacity(N);

    for &sd in stalk_dims {
        // Average the sd rows belonging to this node.
        let mut feat = vec![0.0f32; selected.len()];
        for row in offset..(offset + sd) {
            for (j, &col) in selected.iter().enumerate() {
                feat[j] += u.read(row, col) as f32;
            }
        }
        // Normalize by stalk dimension.
        for v in &mut feat {
            *v /= sd as f32;
        }
        node_features.push(feat);
        offset += sd;
    }

    // K-means on the node features.
    Kmeans::new(k)
        .with_seed(42)
        .fit_predict(&node_features)
        .expect("k-means succeeds")
}

/// Compute clustering accuracy using the known ground truth
/// (nodes 0..4 = community 0, 4..8 = community 1, 8..12 = community 2).
///
/// Since cluster labels are arbitrary, we try all permutations of label
/// assignments and return the best match.
fn cluster_accuracy(labels: &[usize]) -> f64 {
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let ground_truth: Vec<usize> = (0..N).map(|i| i / NODES_PER_COMMUNITY).collect();

    let mut best = 0usize;
    for perm in &perms {
        let correct = labels
            .iter()
            .zip(&ground_truth)
            .filter(|(&predicted, &gt)| perm[predicted] == gt)
            .count();
        best = best.max(correct);
    }
    best as f64 / N as f64
}

fn print_labels(labels: &[usize]) {
    for c in 0..K {
        let base = c * NODES_PER_COMMUNITY;
        let node_labels: Vec<usize> = (base..base + NODES_PER_COMMUNITY)
            .map(|i| labels[i])
            .collect();
        println!(
            "  community {c} (nodes {base}-{}): labels {:?}",
            base + NODES_PER_COMMUNITY - 1,
            node_labels
        );
    }
}
