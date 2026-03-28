//! Sheaf Laplacian meets spectral methods.
//!
//! Demonstrates how the cellular sheaf Laplacian (sheaf) generalizes the
//! standard graph Laplacian (lapl) and enables richer spectral embeddings.
//!
//! # What this example shows
//!
//! 1. **Trivial sheaf = graph Laplacian**: a sheaf with R^1 stalks and identity
//!    restriction maps reproduces the classical graph Laplacian exactly.
//!
//! 2. **Constant sheaf (R^2, identity maps)**: H^0 = d on a connected graph.
//!    Each stalk dimension contributes one independent global section.
//!
//! 3. **Twisted sheaf breaks consistency**: a rotation on one edge of a
//!    triangle creates an inconsistent cycle, reducing H^0 to zero.
//!
//! 4. **Sheaf-based spectral embedding**: projecting nodes onto the smallest
//!    non-zero eigenvectors of the sheaf Laplacian gives a richer embedding
//!    than the standard spectral embedding.
//!
//! # Graph
//!
//! Barbell graph: two triangles connected by a bridge.
//!
//! ```text
//!   Clique A: {0, 1, 2}     Clique B: {3, 4, 5}
//!   Bridge: 2 -- 3
//! ```

use lapl::adjacency_to_laplacian;
use ndarray::Array2;
use sheaf::CellularSheaf;

#[allow(clippy::expect_used)]
fn main() {
    println!("=== Sheaf Laplacian vs Graph Laplacian ===\n");

    // -- Build the barbell graph --
    let n = 6;
    let edges: Vec<(usize, usize)> = vec![
        // Clique A
        (0, 1),
        (0, 2),
        (1, 2),
        // Bridge
        (2, 3),
        // Clique B
        (3, 4),
        (3, 5),
        (4, 5),
    ];

    // Adjacency matrix for lapl
    let mut adj = Array2::<f64>::zeros((n, n));
    for &(u, v) in &edges {
        adj[[u, v]] = 1.0;
        adj[[v, u]] = 1.0;
    }

    // ---------------------------------------------------------------
    // Part 1: Trivial sheaf Laplacian == graph Laplacian
    // ---------------------------------------------------------------
    println!("--- Part 1: Trivial sheaf matches graph Laplacian ---\n");

    let trivial = CellularSheaf::trivial(n, &edges);
    let sheaf_lap = trivial.laplacian();

    let graph_lap = adjacency_to_laplacian(&adj);

    let mut max_diff = 0.0_f64;
    for r in 0..n {
        for c in 0..n {
            let diff = (sheaf_lap[(r, c)] - graph_lap[[r, c]]).abs();
            max_diff = max_diff.max(diff);
        }
    }
    println!("Max entry-wise difference: {max_diff:.2e}");
    assert!(
        max_diff < 1e-12,
        "Trivial sheaf should match graph Laplacian"
    );
    println!("PASSED: trivial sheaf Laplacian == graph Laplacian\n");

    let trivial_eigs = sorted_eigenvalues(&sheaf_lap);
    println!("Trivial sheaf eigenvalues:");
    for (i, ev) in trivial_eigs.iter().enumerate() {
        println!("  lambda_{i} = {ev:.6}");
    }
    println!();

    let h0_trivial = trivial.h0_dimension(1e-8);
    println!("H^0 dimension (trivial): {h0_trivial}");
    assert_eq!(h0_trivial, 1, "Connected graph should have H^0 = 1");
    println!("PASSED: connected trivial sheaf has H^0 = 1\n");

    // ---------------------------------------------------------------
    // Part 2: Constant sheaf (R^2, all identity maps)
    // ---------------------------------------------------------------
    println!("--- Part 2: Constant sheaf (R^2 stalks, identity maps) ---\n");

    let constant = CellularSheaf::constant(n, &edges, 2);
    let const_lap = constant.laplacian();
    let const_eigs = sorted_eigenvalues(&const_lap);
    let total_dim = constant.total_dim(); // 6 * 2 = 12

    println!("Constant sheaf Laplacian size: {total_dim} x {total_dim}");
    println!("Eigenvalues (each graph eigenvalue appears with multiplicity 2):");
    for (i, ev) in const_eigs.iter().enumerate() {
        println!("  lambda_{i} = {ev:.6}");
    }
    println!();

    let h0_constant = constant.h0_dimension(1e-8);
    println!("H^0 dimension (constant R^2): {h0_constant}");
    assert_eq!(
        h0_constant, 2,
        "Constant R^d sheaf on connected graph has H^0 = d"
    );
    println!("PASSED: constant R^2 sheaf has H^0 = 2\n");

    // ---------------------------------------------------------------
    // Part 3: Twisted sheaf -- inconsistent cycle kills global sections
    // ---------------------------------------------------------------
    println!("--- Part 3: Twisted triangle (rotation breaks consistency) ---\n");

    // Inconsistency requires a cycle. On a tree, any invertible restriction
    // maps still allow global sections (adjust vectors along the unique path).
    // On a triangle, composing maps around the cycle must equal the identity
    // for a global section to exist. A 90-degree rotation breaks this.

    let identity_2x2 = vec![1.0, 0.0, 0.0, 1.0]; // column-major I_2

    let tri_n = 3;
    let tri_edges = vec![(0, 1), (1, 2), (0, 2)];
    let tri_stalk_dims = vec![2; tri_n];
    let tri_edge_dims = vec![2; tri_edges.len()];

    // Around the triangle: identity on (0,1) and (1,2), 90-deg rotation on (0,2).
    // A global section must satisfy:
    //   x_0 = x_1  (edge 0-1, identity)
    //   x_1 = x_2  (edge 1-2, identity)
    //   x_0 = R * x_2  (edge 0-2, rotation)
    // Combined: x_0 = R * x_0 => only x_0 = 0 satisfies this for 90-deg rotation.
    // So H^0 = 0.
    let rotation_90 = vec![0.0, 1.0, -1.0, 0.0]; // column-major [[0,-1],[1,0]]

    let tri_maps = vec![
        (identity_2x2.clone(), identity_2x2.clone()), // edge (0,1)
        (identity_2x2.clone(), identity_2x2.clone()), // edge (1,2)
        (identity_2x2.clone(), rotation_90),          // edge (0,2): target rotated
    ];

    let twisted_triangle = CellularSheaf::new(
        tri_n,
        tri_stalk_dims,
        tri_edges.clone(),
        tri_edge_dims,
        tri_maps,
    )
    .expect("valid sheaf construction");

    let twisted_lap = twisted_triangle.laplacian();
    let twisted_eigs = sorted_eigenvalues(&twisted_lap);

    println!("Twisted triangle (R^2 stalks, rotation on one edge):");
    println!(
        "Laplacian size: {} x {}",
        twisted_triangle.total_dim(),
        twisted_triangle.total_dim()
    );
    println!("Eigenvalues:");
    for (i, ev) in twisted_eigs.iter().enumerate() {
        println!("  lambda_{i} = {ev:.6}");
    }

    let h0_twisted = twisted_triangle.h0_dimension(1e-8);
    println!("H^0 dimension: {h0_twisted}");
    println!();

    // Compare with constant sheaf on the same triangle
    let const_triangle = CellularSheaf::constant(tri_n, &tri_edges, 2);
    let h0_const_tri = const_triangle.h0_dimension(1e-8);
    println!("Constant R^2 sheaf on triangle: H^0 = {h0_const_tri}");
    println!("Twisted sheaf on triangle:      H^0 = {h0_twisted}");
    println!();
    println!("The rotation around the cycle creates an inconsistency:");
    println!("following identity maps 0->1->2 gives x_0 = x_2, but the");
    println!("direct edge 0->2 requires x_0 = R(x_2). Both hold only for x = 0.");
    assert_eq!(h0_const_tri, 2);
    assert_eq!(h0_twisted, 0, "Inconsistent cycle should have H^0 = 0");
    println!("PASSED: twisted triangle has H^0 = 0\n");

    // ---------------------------------------------------------------
    // Part 4: Sheaf-based spectral embedding
    // ---------------------------------------------------------------
    println!("--- Part 4: Spectral embedding comparison ---\n");

    // Use the constant R^2 sheaf on the barbell for embedding comparison
    // (it has non-degenerate eigenvectors).
    let embedding_dim = 2;

    let const_barbell_lap = constant.laplacian();
    let sheaf_emb = sheaf_spectral_embedding(&const_barbell_lap, embedding_dim);

    println!("Sheaf spectral embedding (constant R^2 sheaf, 2D):");
    for node in 0..n {
        let row_start = node * 2;
        println!(
            "  Node {node}: stalk[0]=({:.4}, {:.4})  stalk[1]=({:.4}, {:.4})",
            sheaf_emb[row_start][0],
            sheaf_emb[row_start][1],
            sheaf_emb[row_start + 1][0],
            sheaf_emb[row_start + 1][1],
        );
    }
    println!();

    // Standard spectral embedding via lapl
    let std_cfg = lapl::SpectralEmbeddingConfig {
        skip_first: true,
        row_normalize: false,
        ..Default::default()
    };
    let std_embedding =
        lapl::spectral_embedding(&adj, embedding_dim, &std_cfg).expect("embedding succeeds");

    println!("Standard spectral embedding (2D, via lapl):");
    for node in 0..n {
        println!(
            "  Node {node}: ({:.4}, {:.4})",
            std_embedding[[node, 0]],
            std_embedding[[node, 1]]
        );
    }
    println!();

    println!("--- Summary ---\n");
    println!("1. Trivial sheaf Laplacian exactly equals the graph Laplacian.");
    println!("2. Constant R^2 sheaf doubles each eigenvalue (multiplicity 2),");
    println!("   with H^0 = 2 (one global section per stalk dimension).");
    println!("3. Twisted sheaf on a triangle has H^0 = 0: the rotation around");
    println!("   the cycle is inconsistent, so no non-zero global sections exist.");
    println!("4. Sheaf spectral embedding operates in a {total_dim}-dimensional space");
    println!("   (vs {n} for scalar), encoding per-edge constraints in the geometry.");
}

/// Compute sorted eigenvalues of a symmetric faer::Mat.
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

/// Extract a spectral embedding from the sheaf Laplacian.
///
/// Returns one row per stalk-dimension (total_dim rows), each row having
/// `k` coordinates from the smallest non-zero eigenvectors.
fn sheaf_spectral_embedding(lap: &faer::Mat<f64>, k: usize) -> Vec<Vec<f64>> {
    let n = lap.nrows();
    let eig = lap
        .as_ref()
        .selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let u = eig.u();

    // Sort eigenvalues, skip zeros (< 1e-8), take k smallest non-zero.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        s.column_vector()
            .read(a)
            .total_cmp(&s.column_vector().read(b))
    });

    let nonzero_start = order
        .iter()
        .position(|&i| s.column_vector().read(i) > 1e-8)
        .unwrap_or(n);

    let selected: Vec<usize> = order[nonzero_start..].iter().copied().take(k).collect();

    // Build embedding: row i gets coordinates [u[i, selected[0]], u[i, selected[1]], ...]
    let mut embedding = Vec::with_capacity(n);
    for row in 0..n {
        let coords: Vec<f64> = selected.iter().map(|&col| u.read(row, col)).collect();
        embedding.push(coords);
    }
    embedding
}
