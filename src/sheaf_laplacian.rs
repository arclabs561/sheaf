//! Cellular sheaf Laplacian on graphs.
//!
//! A cellular sheaf on a graph assigns a vector space (stalk) to each node
//! and each edge, plus linear restriction maps from node stalks to incident
//! edge stalks. The sheaf Laplacian generalizes the graph Laplacian to this
//! setting.
//!
//! When all stalks are $$\mathbb{R}^1$$ and all restriction maps are the
//! identity, the sheaf Laplacian reduces to the standard graph Laplacian.
//!
//! Reference: Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves" (2019).
//!
//! # Related crates
//! - [`lapl`]: Standard graph Laplacians and spectral methods. The trivial sheaf Laplacian equals the graph Laplacian.
//! - [`graphops`]: Graph algorithms (PageRank, walks) that operate on the same graph structures.

use crate::error::{Error, Result};
use faer::Mat;

/// A cellular sheaf on a graph.
///
/// Assigns to each node `v` a stalk of dimension `stalk_dims[v]`,
/// to each edge a stalk of dimension `edge_dims[i]`, and for each
/// edge `(u, v)` a pair of restriction maps from the source and
/// target node stalks into the edge stalk.
#[derive(Debug, Clone)]
pub struct CellularSheaf {
    num_nodes: usize,
    stalk_dims: Vec<usize>,
    edges: Vec<(usize, usize)>,
    /// For each edge: (source_map, target_map) stored as column-major
    /// flattened matrices of shape (edge_dim x node_stalk_dim).
    restriction_maps: Vec<(Vec<f64>, Vec<f64>)>,
    edge_dims: Vec<usize>,
}

impl CellularSheaf {
    /// Create a new cellular sheaf.
    ///
    /// # Arguments
    /// - `num_nodes`: number of graph nodes
    /// - `stalk_dims`: stalk dimension for each node (length `num_nodes`)
    /// - `edges`: edge list as `(source, target)` pairs
    /// - `edge_dims`: stalk dimension for each edge
    /// - `restriction_maps`: for each edge, `(F_source, F_target)` where each
    ///   is a column-major flattened matrix of shape `(edge_dim x node_stalk_dim)`
    ///
    /// # Errors
    /// Returns an error if array lengths are inconsistent or restriction map
    /// dimensions do not match the declared stalk dimensions.
    pub fn new(
        num_nodes: usize,
        stalk_dims: Vec<usize>,
        edges: Vec<(usize, usize)>,
        edge_dims: Vec<usize>,
        restriction_maps: Vec<(Vec<f64>, Vec<f64>)>,
    ) -> Result<Self> {
        if stalk_dims.len() != num_nodes {
            return Err(Error::DimensionMismatch {
                expected: num_nodes,
                found: stalk_dims.len(),
            });
        }
        if edge_dims.len() != edges.len() {
            return Err(Error::DimensionMismatch {
                expected: edges.len(),
                found: edge_dims.len(),
            });
        }
        if restriction_maps.len() != edges.len() {
            return Err(Error::DimensionMismatch {
                expected: edges.len(),
                found: restriction_maps.len(),
            });
        }
        for (i, &(u, v)) in edges.iter().enumerate() {
            if u >= num_nodes || v >= num_nodes {
                return Err(Error::Other(format!(
                    "edge {i} references node {} but only {num_nodes} nodes exist",
                    u.max(v)
                )));
            }
            let de = edge_dims[i];
            let du = stalk_dims[u];
            let dv = stalk_dims[v];
            let (ref fu, ref fv) = restriction_maps[i];
            if fu.len() != de * du {
                return Err(Error::ShapeMismatch {
                    expected: format!(
                        "{}x{} = {} entries for source map of edge {i}",
                        de,
                        du,
                        de * du
                    ),
                    actual: format!("{} entries", fu.len()),
                });
            }
            if fv.len() != de * dv {
                return Err(Error::ShapeMismatch {
                    expected: format!(
                        "{}x{} = {} entries for target map of edge {i}",
                        de,
                        dv,
                        de * dv
                    ),
                    actual: format!("{} entries", fv.len()),
                });
            }
        }
        Ok(Self {
            num_nodes,
            stalk_dims,
            edges,
            restriction_maps,
            edge_dims,
        })
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Stalk dimensions per node.
    pub fn stalk_dims(&self) -> &[usize] {
        &self.stalk_dims
    }

    /// Edges as `(source, target)` pairs.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Edge stalk dimensions.
    pub fn edge_dims(&self) -> &[usize] {
        &self.edge_dims
    }

    /// Total dimension of the node-stalk vector space (sum of all stalk dims).
    pub fn total_dim(&self) -> usize {
        self.stalk_dims.iter().sum()
    }

    /// Compute the sheaf Laplacian as a dense matrix.
    ///
    /// Returns a `total_dim x total_dim` matrix where `total_dim = sum(stalk_dims)`.
    ///
    /// The Laplacian is defined as:
    /// - Diagonal block `L[v,v] = sum_{e incident to v} F_{v,e}^T F_{v,e}`
    /// - Off-diagonal block `L[u,v] = -F_{u,e}^T F_{v,e}` for edge `e = (u,v)`
    pub fn laplacian(&self) -> Mat<f64> {
        let n = self.total_dim();
        let mut lap = Mat::<f64>::zeros(n, n);

        // Precompute node offsets into the block matrix.
        let offsets: Vec<usize> = {
            let mut o = vec![0usize; self.num_nodes + 1];
            for i in 0..self.num_nodes {
                o[i + 1] = o[i] + self.stalk_dims[i];
            }
            o
        };

        for (idx, &(u, v)) in self.edges.iter().enumerate() {
            let de = self.edge_dims[idx];
            let du = self.stalk_dims[u];
            let dv = self.stalk_dims[v];
            let (ref fu_flat, ref fv_flat) = self.restriction_maps[idx];

            // Read restriction maps as faer matrices (column-major).
            // fu: de x du, fv: de x dv
            let fu = faer::mat::from_column_major_slice::<f64>(fu_flat, de, du);
            let fv = faer::mat::from_column_major_slice::<f64>(fv_flat, de, dv);

            let ou = offsets[u];
            let ov = offsets[v];

            // Diagonal block for u: += fu^T * fu  (du x du)
            // Diagonal block for v: += fv^T * fv  (dv x dv)
            // Off-diagonal (u,v): -= fu^T * fv    (du x dv)
            // Off-diagonal (v,u): -= fv^T * fu    (dv x du)
            for r in 0..du {
                for c in 0..du {
                    let mut val = 0.0;
                    for k in 0..de {
                        val += fu[(k, r)] * fu[(k, c)];
                    }
                    lap[(ou + r, ou + c)] += val;
                }
            }
            for r in 0..dv {
                for c in 0..dv {
                    let mut val = 0.0;
                    for k in 0..de {
                        val += fv[(k, r)] * fv[(k, c)];
                    }
                    lap[(ov + r, ov + c)] += val;
                }
            }
            for r in 0..du {
                for c in 0..dv {
                    let mut val = 0.0;
                    for k in 0..de {
                        val += fu[(k, r)] * fv[(k, c)];
                    }
                    lap[(ou + r, ov + c)] -= val;
                    lap[(ov + c, ou + r)] -= val;
                }
            }
        }

        lap
    }

    /// Compute the dimension of the 0th sheaf cohomology (kernel of the Laplacian).
    ///
    /// `H^0(F)` measures global consistency of the sheaf: sections that are
    /// compatible across all restriction maps. For a trivial sheaf on a connected
    /// graph, `dim H^0 = 1`. For a disconnected graph with `k` components,
    /// `dim H^0 = k`.
    ///
    /// Eigenvalues below `tol` are treated as zero.
    pub fn h0_dimension(&self, tol: f64) -> usize {
        let lap = self.laplacian();
        let n = lap.nrows();
        if n == 0 {
            return 0;
        }
        eigenvalues_below_tol(&lap, tol)
    }

    /// Construct a trivial sheaf: all stalks `R^1`, all restriction maps the identity.
    ///
    /// The Laplacian of this sheaf equals the standard graph Laplacian.
    ///
    /// # Panics
    ///
    /// Panics if any edge references a node index >= `num_nodes`.
    #[allow(clippy::expect_used)]
    pub fn trivial(num_nodes: usize, edges: &[(usize, usize)]) -> Self {
        let stalk_dims = vec![1; num_nodes];
        let edge_dims = vec![1; edges.len()];
        let restriction_maps = vec![(vec![1.0], vec![1.0]); edges.len()];
        Self::new(
            num_nodes,
            stalk_dims,
            edges.to_vec(),
            edge_dims,
            restriction_maps,
        )
        .expect("trivial sheaf edges must reference valid nodes")
    }

    /// Construct a constant sheaf: all stalks `R^d`, all restriction maps the
    /// `d x d` identity matrix.
    ///
    /// # Panics
    ///
    /// Panics if any edge references a node index >= `num_nodes`.
    #[allow(clippy::expect_used)]
    pub fn constant(num_nodes: usize, edges: &[(usize, usize)], d: usize) -> Self {
        let stalk_dims = vec![d; num_nodes];
        let edge_dims = vec![d; edges.len()];
        // Identity matrix in column-major: d*d entries
        let eye: Vec<f64> = {
            let mut m = vec![0.0; d * d];
            for i in 0..d {
                m[i * d + i] = 1.0; // column-major: (row, col) at index col*nrows + row
            }
            m
        };
        let restriction_maps = vec![(eye.clone(), eye); edges.len()];
        Self::new(
            num_nodes,
            stalk_dims,
            edges.to_vec(),
            edge_dims,
            restriction_maps,
        )
        .expect("constant sheaf edges must reference valid nodes")
    }
}

/// Count eigenvalues of a symmetric matrix below `tol`.
fn eigenvalues_below_tol(mat: &Mat<f64>, tol: f64) -> usize {
    let n = mat.nrows();
    // Use faer's symmetric eigenvalue decomposition.
    let eigenvalues = mat
        .as_ref()
        .selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eigenvalues.s();
    let mut count = 0;
    for i in 0..n {
        if s.column_vector().read(i).abs() < tol {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Triangle graph: 0-1, 1-2, 0-2
    fn triangle_edges() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2), (0, 2)]
    }

    /// Path graph: 0-1-2
    fn path_edges() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2)]
    }

    #[test]
    fn trivial_sheaf_equals_graph_laplacian_triangle() {
        // Standard graph Laplacian of a triangle:
        //  [ 2 -1 -1 ]
        //  [-1  2 -1 ]
        //  [-1 -1  2 ]
        let sheaf = CellularSheaf::trivial(3, &triangle_edges());
        let lap = sheaf.laplacian();
        assert_eq!(lap.nrows(), 3);
        assert_eq!(lap.ncols(), 3);

        let expected = [[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]];
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (lap[(r, c)] - expected[r][c]).abs() < 1e-12,
                    "mismatch at ({r},{c}): got {}, expected {}",
                    lap[(r, c)],
                    expected[r][c]
                );
            }
        }
    }

    #[test]
    fn trivial_sheaf_equals_graph_laplacian_path() {
        // Path 0-1-2:
        //  [ 1 -1  0 ]
        //  [-1  2 -1 ]
        //  [ 0 -1  1 ]
        let sheaf = CellularSheaf::trivial(3, &path_edges());
        let lap = sheaf.laplacian();

        let expected = [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]];
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (lap[(r, c)] - expected[r][c]).abs() < 1e-12,
                    "mismatch at ({r},{c}): got {}, expected {}",
                    lap[(r, c)],
                    expected[r][c]
                );
            }
        }
    }

    #[test]
    fn laplacian_is_symmetric() {
        // Non-trivial sheaf: R^2 stalks on a triangle with random-ish maps
        let edges = triangle_edges();
        let stalk_dims = vec![2, 2, 2];
        let edge_dims = vec![2, 2, 2];

        // Some non-identity restriction maps (column-major 2x2)
        let maps = vec![
            (vec![1.0, 0.0, 0.5, 1.0], vec![1.0, 0.3, 0.0, 1.0]),
            (vec![0.8, 0.2, 0.1, 0.9], vec![1.0, 0.0, 0.0, 1.0]),
            (vec![1.0, 0.0, 0.0, 1.0], vec![0.7, 0.4, 0.1, 0.6]),
        ];
        let sheaf = CellularSheaf::new(3, stalk_dims, edges, edge_dims, maps).unwrap();
        let lap = sheaf.laplacian();
        let n = lap.nrows();
        for r in 0..n {
            for c in 0..n {
                assert!(
                    (lap[(r, c)] - lap[(c, r)]).abs() < 1e-12,
                    "not symmetric at ({r},{c}): {} vs {}",
                    lap[(r, c)],
                    lap[(c, r)]
                );
            }
        }
    }

    #[test]
    fn laplacian_is_positive_semidefinite() {
        let edges = triangle_edges();
        let stalk_dims = vec![2, 2, 2];
        let edge_dims = vec![2, 2, 2];
        let maps = vec![
            (vec![1.0, 0.0, 0.5, 1.0], vec![1.0, 0.3, 0.0, 1.0]),
            (vec![0.8, 0.2, 0.1, 0.9], vec![1.0, 0.0, 0.0, 1.0]),
            (vec![1.0, 0.0, 0.0, 1.0], vec![0.7, 0.4, 0.1, 0.6]),
        ];
        let sheaf = CellularSheaf::new(3, stalk_dims, edges, edge_dims, maps).unwrap();
        let lap = sheaf.laplacian();
        let n = lap.nrows();
        let eig = lap
            .as_ref()
            .selfadjoint_eigendecomposition(faer::Side::Lower);
        let s = eig.s();
        for i in 0..n {
            assert!(
                s.column_vector().read(i) >= -1e-10,
                "negative eigenvalue at index {i}: {}",
                s.column_vector().read(i)
            );
        }
    }

    #[test]
    fn h0_connected_trivial_is_one() {
        let sheaf = CellularSheaf::trivial(3, &triangle_edges());
        assert_eq!(sheaf.h0_dimension(1e-8), 1);
    }

    #[test]
    fn h0_disconnected_trivial_equals_components() {
        // Two disconnected edges: {0-1} and {2-3} => 2 components
        let edges = vec![(0, 1), (2, 3)];
        let sheaf = CellularSheaf::trivial(4, &edges);
        assert_eq!(sheaf.h0_dimension(1e-8), 2);
    }

    #[test]
    fn h0_isolated_nodes() {
        // 3 nodes, no edges => 3 components
        let sheaf = CellularSheaf::trivial(3, &[]);
        assert_eq!(sheaf.h0_dimension(1e-8), 3);
        // Laplacian should be zero matrix
        let lap = sheaf.laplacian();
        for r in 0..3 {
            for c in 0..3 {
                assert!((lap[(r, c)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn h0_inconsistent_sheaf() {
        // Edge 0-1 with restriction maps that are incompatible:
        // F_0 = [1], F_1 = [2] means the "agreement" equation is
        // 1*x_0 = 2*x_1, which with the Laplacian structure means
        // no nonzero kernel vector. H^0 = 0.
        let sheaf = CellularSheaf::new(
            2,
            vec![1, 1],
            vec![(0, 1)],
            vec![1],
            vec![(vec![1.0], vec![2.0])],
        )
        .unwrap();
        // Laplacian: [[1, -2], [-2, 4]]
        // det = 4 - 4 = 0, so there IS a kernel vector (2, 1).
        // This is actually consistent: x0=2, x1=1 satisfies 1*2 = 2*1.
        // For true inconsistency, we need overconstrained edges.
        // With a single edge and scalar stalks, there's always a solution.
        assert_eq!(sheaf.h0_dimension(1e-8), 1);
    }

    #[test]
    fn h0_overconstrained_sheaf() {
        // Triangle with R^1 stalks and edge dim 1.
        // Restriction maps chosen so the system is overconstrained:
        // Edge (0,1): F_0=1, F_1=1  => x0 = x1
        // Edge (1,2): F_1=1, F_2=1  => x1 = x2
        // Edge (0,2): F_0=1, F_2=-1 => x0 = -x2
        // Combined: x0 = x1 = x2 AND x0 = -x2 => only x=0 satisfies all.
        // H^0 = 0.
        let sheaf = CellularSheaf::new(
            3,
            vec![1, 1, 1],
            vec![(0, 1), (1, 2), (0, 2)],
            vec![1, 1, 1],
            vec![
                (vec![1.0], vec![1.0]),
                (vec![1.0], vec![1.0]),
                (vec![1.0], vec![-1.0]),
            ],
        )
        .unwrap();
        assert_eq!(sheaf.h0_dimension(1e-8), 0);
    }

    #[test]
    fn constant_sheaf_h0_equals_components_times_d() {
        // Constant sheaf with d=3 on a connected triangle.
        // H^0 should equal d = 3 (one independent constant section per dimension).
        let sheaf = CellularSheaf::constant(3, &triangle_edges(), 3);
        assert_eq!(sheaf.h0_dimension(1e-8), 3);
    }

    #[test]
    fn constant_sheaf_disconnected() {
        // 2 components, d=2 => H^0 = 2*2 = 4
        let edges = vec![(0, 1), (2, 3)];
        let sheaf = CellularSheaf::constant(4, &edges, 2);
        assert_eq!(sheaf.h0_dimension(1e-8), 4);
    }

    #[test]
    fn validation_rejects_bad_dimensions() {
        let result = CellularSheaf::new(
            2,
            vec![1], // wrong length
            vec![(0, 1)],
            vec![1],
            vec![(vec![1.0], vec![1.0])],
        );
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_bad_map_size() {
        let result = CellularSheaf::new(
            2,
            vec![2, 1],
            vec![(0, 1)],
            vec![1],
            vec![(vec![1.0], vec![1.0])], // source map should be 1x2=2 entries
        );
        assert!(result.is_err());
    }

    #[test]
    fn validation_rejects_out_of_bounds_node() {
        let result = CellularSheaf::new(
            2,
            vec![1, 1],
            vec![(0, 5)], // node 5 doesn't exist
            vec![1],
            vec![(vec![1.0], vec![1.0])],
        );
        assert!(result.is_err());
    }
}
