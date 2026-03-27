//! Learnable (parametric) restriction maps for the sheaf Laplacian.
//!
//! Three parameterization families for restriction maps $F_{v,e}$:
//!
//! - **Diagonal**: $F = \mathrm{diag}(d_1, \ldots, d_k)$. Parameters: $k$ per endpoint.
//! - **Orthogonal**: $F = \exp(A)$ where $A$ is skew-symmetric. Parameters: $k(k-1)/2$ per endpoint.
//! - **General**: arbitrary $k \times k$ matrix. Parameters: $k^2$ per endpoint.
//!
//! Reference: Bodnar et al., "Neural Sheaf Diffusion", ICML 2022.

use faer::Mat;

use crate::error::{Error, Result};
use crate::sheaf_laplacian::CellularSheaf;

/// Restriction map parameterization family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestrictionFamily {
    /// Each restriction map is a diagonal matrix.
    Diagonal,
    /// Each restriction map is orthogonal, parameterized via matrix exponential
    /// of a skew-symmetric matrix.
    Orthogonal,
    /// Each restriction map is an arbitrary dense matrix.
    General,
}

/// Learnable sheaf with parametric restriction maps.
///
/// All stalks have uniform dimension `stalk_dim`. Edge stalks equal `stalk_dim`
/// (square restriction maps). Parameters are stored in a flat vector; use
/// [`LearnableSheaf::to_cellular_sheaf`] to materialize the sheaf.
#[derive(Debug, Clone)]
pub struct LearnableSheaf {
    num_nodes: usize,
    stalk_dim: usize,
    edges: Vec<(usize, usize)>,
    family: RestrictionFamily,
    params: Vec<f64>,
}

impl LearnableSheaf {
    /// Create a new learnable sheaf with zero-initialized parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if any edge references a node index >= `num_nodes`.
    pub fn new(
        num_nodes: usize,
        stalk_dim: usize,
        edges: Vec<(usize, usize)>,
        family: RestrictionFamily,
    ) -> Result<Self> {
        for (i, &(u, v)) in edges.iter().enumerate() {
            if u >= num_nodes || v >= num_nodes {
                return Err(Error::Other(format!(
                    "edge {i} references node {} but only {num_nodes} nodes exist",
                    u.max(v)
                )));
            }
        }
        let n = params_per_endpoint(stalk_dim, family);
        let total = 2 * edges.len() * n;
        Ok(Self {
            num_nodes,
            stalk_dim,
            edges,
            family,
            params: vec![0.0; total],
        })
    }

    /// Number of learnable parameters.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Mutable access to the parameter vector.
    pub fn params_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }

    /// Read-only access to the parameter vector.
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Number of graph nodes.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Stalk dimension (uniform across all nodes and edges).
    pub fn stalk_dim(&self) -> usize {
        self.stalk_dim
    }

    /// Edge list.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Parameterization family.
    pub fn family(&self) -> RestrictionFamily {
        self.family
    }

    /// Build restriction maps from current parameters.
    ///
    /// Returns one `(source_map, target_map)` pair per edge, each a column-major
    /// flattened `stalk_dim x stalk_dim` matrix.
    pub fn build_maps(&self) -> Vec<(Vec<f64>, Vec<f64>)> {
        let d = self.stalk_dim;
        let n = params_per_endpoint(d, self.family);
        self.edges
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let base = 2 * i * n;
                let src_params = &self.params[base..base + n];
                let tgt_params = &self.params[base + n..base + 2 * n];
                let src_map = params_to_matrix(d, self.family, src_params);
                let tgt_map = params_to_matrix(d, self.family, tgt_params);
                (src_map, tgt_map)
            })
            .collect()
    }

    /// Build a [`CellularSheaf`] from current parameters.
    pub fn to_cellular_sheaf(&self) -> CellularSheaf {
        let d = self.stalk_dim;
        let stalk_dims = vec![d; self.num_nodes];
        let edge_dims = vec![d; self.edges.len()];
        let maps = self.build_maps();
        // Dimensions are consistent by construction (uniform stalk_dim).
        #[allow(clippy::expect_used)]
        CellularSheaf::new(
            self.num_nodes,
            stalk_dims,
            self.edges.clone(),
            edge_dims,
            maps,
        )
        .expect("learnable sheaf parameters are dimensionally consistent")
    }

    /// Compute the sheaf Laplacian from current parameters.
    pub fn laplacian(&self) -> Mat<f64> {
        self.to_cellular_sheaf().laplacian()
    }

    /// Compute the dimension of $H^0$ (kernel of the Laplacian).
    ///
    /// Measures global consistency: how many independent sections are
    /// compatible across all restriction maps.
    pub fn h0_dimension(&self, tol: f64) -> usize {
        self.to_cellular_sheaf().h0_dimension(tol)
    }

    /// Initialize parameters so that restriction maps are identity matrices.
    ///
    /// - Diagonal: all diagonal entries set to 1.
    /// - Orthogonal: all skew-symmetric entries set to 0 (exp(0) = I).
    /// - General: identity matrix entries.
    pub fn init_identity(&mut self) {
        let d = self.stalk_dim;
        let n = params_per_endpoint(d, self.family);
        match self.family {
            RestrictionFamily::Diagonal => {
                // Each endpoint: d parameters, all 1.0
                for p in &mut self.params {
                    *p = 1.0;
                }
            }
            RestrictionFamily::Orthogonal => {
                // All zeros => exp(0) = I
                for p in &mut self.params {
                    *p = 0.0;
                }
            }
            RestrictionFamily::General => {
                // Each endpoint: d*d parameters forming identity matrix (column-major)
                for endpoint in 0..(2 * self.edges.len()) {
                    let base = endpoint * n;
                    for j in 0..n {
                        self.params[base + j] = 0.0;
                    }
                    for i in 0..d {
                        // column-major: (row, col) at index col * d + row
                        self.params[base + i * d + i] = 1.0;
                    }
                }
            }
        }
    }

    /// Initialize parameters with deterministic pseudo-random values.
    ///
    /// Uses a simple xorshift64 PRNG seeded by `seed`. Values are drawn
    /// from a small range around the identity initialization to avoid
    /// starting too far from consistent maps.
    pub fn init_random(&mut self, seed: u64) {
        self.init_identity();
        let mut rng = seed.wrapping_add(1);
        for p in &mut self.params {
            // xorshift64
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Small perturbation in [-0.1, 0.1]
            let u = (rng as f64) / (u64::MAX as f64);
            *p += 0.2 * u - 0.1;
        }
    }
}

/// Number of parameters per endpoint for a given stalk dimension and family.
fn params_per_endpoint(d: usize, family: RestrictionFamily) -> usize {
    match family {
        RestrictionFamily::Diagonal => d,
        RestrictionFamily::Orthogonal => d * (d - 1) / 2,
        RestrictionFamily::General => d * d,
    }
}

/// Convert a parameter slice to a column-major flattened `d x d` matrix.
fn params_to_matrix(d: usize, family: RestrictionFamily, params: &[f64]) -> Vec<f64> {
    match family {
        RestrictionFamily::Diagonal => {
            let mut m = vec![0.0; d * d];
            for i in 0..d {
                m[i * d + i] = params[i];
            }
            m
        }
        RestrictionFamily::Orthogonal => skew_symmetric_exp(d, params),
        RestrictionFamily::General => params.to_vec(),
    }
}

/// Compute exp(A) where A is the skew-symmetric matrix whose upper triangle
/// is given by `params` (row-major upper triangle, k*(k-1)/2 values).
///
/// For d=1: identity (no parameters).
/// For d=2: Givens rotation with angle `params[0]`.
/// For d=3: Rodrigues formula.
/// For d>=4: Pade(6,6) approximation with scaling-and-squaring.
fn skew_symmetric_exp(d: usize, params: &[f64]) -> Vec<f64> {
    if d == 0 {
        return vec![];
    }
    if d == 1 {
        return vec![1.0];
    }

    // Build skew-symmetric matrix A (column-major d x d).
    let mut a = vec![0.0; d * d];
    let mut idx = 0;
    for row in 0..d {
        for col in (row + 1)..d {
            let v = params[idx];
            // A[row, col] = v, A[col, row] = -v
            // column-major: (r, c) at c * d + r
            a[col * d + row] = v;
            a[row * d + col] = -v;
            idx += 1;
        }
    }

    if d == 2 {
        // exp([[0, -theta], [theta, 0]]) = [[cos, -sin], [sin, cos]]
        let theta = params[0];
        let (s, c) = theta.sin_cos();
        // column-major 2x2
        return vec![c, s, -s, c];
    }

    if d == 3 {
        // Rodrigues: exp(A) = I + sin(theta)/theta * A + (1 - cos(theta))/theta^2 * A^2
        // where theta = ||omega|| and A is the skew-symmetric matrix of omega.
        // params = [a01, a02, a12] => omega = (a12, -a02, a01)
        let a01 = params[0];
        let a02 = params[1];
        let a12 = params[2];
        let theta_sq = a01 * a01 + a02 * a02 + a12 * a12;
        if theta_sq < 1e-30 {
            let mut eye = vec![0.0; 9];
            eye[0] = 1.0;
            eye[4] = 1.0;
            eye[8] = 1.0;
            return eye;
        }
        let theta = theta_sq.sqrt();
        let sinc = theta.sin() / theta;
        let cosc = (1.0 - theta.cos()) / theta_sq;

        // A^2 in column-major
        let a2 = mat_mul_col_major(d, &a, &a);

        let mut result = vec![0.0; 9];
        for i in 0..9 {
            result[i] = sinc * a[i] + cosc * a2[i];
        }
        // Add identity
        result[0] += 1.0;
        result[4] += 1.0;
        result[8] += 1.0;
        return result;
    }

    // General case: scaling-and-squaring with Pade(6,6) approximation.
    // Find s such that ||A / 2^s|| < 0.5, then compute pade(A/2^s), then square s times.
    let norm = matrix_inf_norm(d, &a);
    let s = if norm > 0.5 {
        (norm / 0.5).log2().ceil() as u32
    } else {
        0
    };

    let scale = 0.5_f64.powi(s as i32);
    let scaled: Vec<f64> = a.iter().map(|&x| x * scale).collect();

    let exp_scaled = pade_exp(d, &scaled);

    // Repeated squaring
    let mut result = exp_scaled;
    for _ in 0..s {
        result = mat_mul_col_major(d, &result, &result);
    }

    result
}

/// Column-major matrix multiplication: C = A * B, each d x d.
fn mat_mul_col_major(d: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; d * d];
    for col in 0..d {
        for k in 0..d {
            let b_kc = b[col * d + k];
            for row in 0..d {
                c[col * d + row] += a[k * d + row] * b_kc;
            }
        }
    }
    c
}

/// Infinity norm of a column-major d x d matrix: max row sum of absolute values.
fn matrix_inf_norm(d: usize, a: &[f64]) -> f64 {
    let mut max_row_sum = 0.0_f64;
    for row in 0..d {
        let mut s = 0.0;
        for col in 0..d {
            s += a[col * d + row].abs();
        }
        max_row_sum = max_row_sum.max(s);
    }
    max_row_sum
}

/// Pade(6,6) approximation of exp(A) for a column-major d x d matrix.
///
/// Uses the standard coefficients b_0..b_6 and computes
/// `exp(A) = (D + N)^{-1} (D - N)` where `N, D` are polynomial evaluations.
/// (Actually: `exp(A) = D^{-1} N` where `N = sum b_i A^i`, `D = sum (-1)^i b_i A^i`.)
fn pade_exp(d: usize, a: &[f64]) -> Vec<f64> {
    // Pade(6,6) coefficients
    let b: [f64; 7] = [
        1.0,
        1.0 / 2.0,
        1.0 / 9.0,
        1.0 / 72.0,
        1.0 / 1008.0,
        1.0 / 30240.0,
        1.0 / 1209600.0,
    ];

    // Compute powers of A: A^1 through A^6
    let a2 = mat_mul_col_major(d, a, a);
    let a3 = mat_mul_col_major(d, &a2, a);
    let a4 = mat_mul_col_major(d, &a3, a);
    let a5 = mat_mul_col_major(d, &a4, a);
    let a6 = mat_mul_col_major(d, &a5, a);

    let nn = d * d;
    let mut eye = vec![0.0; nn];
    for i in 0..d {
        eye[i * d + i] = 1.0;
    }

    // U = b[1]*A + b[3]*A^3 + b[5]*A^5  (odd terms)
    // V = b[0]*I + b[2]*A^2 + b[4]*A^4 + b[6]*A^6  (even terms)
    let mut u = vec![0.0; nn];
    let mut v = vec![0.0; nn];
    for i in 0..nn {
        u[i] = b[1] * a[i] + b[3] * a3[i] + b[5] * a5[i];
        v[i] = b[0] * eye[i] + b[2] * a2[i] + b[4] * a4[i] + b[6] * a6[i];
    }

    // exp(A) = (V - U)^{-1} (V + U)
    let mut numer = vec![0.0; nn];
    let mut denom = vec![0.0; nn];
    for i in 0..nn {
        numer[i] = v[i] + u[i];
        denom[i] = v[i] - u[i];
    }

    solve_linear_col_major(d, &denom, &numer)
}

/// Solve A X = B for X where A, B are column-major d x d matrices.
/// Uses Gaussian elimination with partial pivoting.
fn solve_linear_col_major(d: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let nn = d * d;
    // Augmented: [A | B] stored as two separate column-major matrices.
    let mut aug_a = a.to_vec();
    let mut aug_b = b.to_vec();

    for col in 0..d {
        // Partial pivoting
        let mut max_val = aug_a[col * d + col].abs();
        let mut max_row = col;
        for row in (col + 1)..d {
            let v = aug_a[col * d + row].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_row != col {
            // Swap rows in both aug_a and aug_b
            for c in 0..d {
                aug_a.swap(c * d + col, c * d + max_row);
                aug_b.swap(c * d + col, c * d + max_row);
            }
        }

        let pivot = aug_a[col * d + col];
        if pivot.abs() < 1e-30 {
            // Near-singular; return identity as fallback
            let mut eye = vec![0.0; nn];
            for i in 0..d {
                eye[i * d + i] = 1.0;
            }
            return eye;
        }

        for row in (col + 1)..d {
            let factor = aug_a[col * d + row] / pivot;
            for c in col..d {
                aug_a[c * d + row] -= factor * aug_a[c * d + col];
            }
            for c in 0..d {
                aug_b[c * d + row] -= factor * aug_b[c * d + col];
            }
        }
    }

    // Back-substitution for each column of B
    let mut x = vec![0.0; nn];
    for rhs_col in 0..d {
        for row in (0..d).rev() {
            let mut val = aug_b[rhs_col * d + row];
            for c in (row + 1)..d {
                val -= aug_a[c * d + row] * x[rhs_col * d + c];
            }
            x[rhs_col * d + row] = val / aug_a[row * d + row];
        }
    }

    x
}

/// Spectral diagnostic: determine if learnable maps are likely beneficial.
///
/// Compares the spectral gap (smallest nonzero eigenvalue) of a trivial sheaf
/// Laplacian against a diagonal sheaf whose restriction maps scale each
/// dimension by the feature variance along that edge. A large ratio indicates
/// heterophilous structure that benefits from non-trivial maps.
///
/// Returns `true` when the ratio exceeds a threshold (empirically 1.5),
/// suggesting the graph has heterophilous structure.
///
/// `node_features[v]` is the feature vector for node `v` (length `stalk_dim`).
pub fn needs_learnable_maps(
    adj: &[(usize, usize)],
    node_features: &[Vec<f64>],
    stalk_dim: usize,
) -> bool {
    if adj.is_empty() || node_features.is_empty() || stalk_dim == 0 {
        return false;
    }

    let num_nodes = node_features.len();

    // Trivial sheaf: identity maps => standard graph Laplacian (per dimension).
    let trivial = CellularSheaf::constant(num_nodes, adj, stalk_dim);
    let trivial_lap = trivial.laplacian();

    // Diagonal sheaf: scale each dimension by feature difference magnitude.
    let mut learnable = match LearnableSheaf::new(
        num_nodes,
        stalk_dim,
        adj.to_vec(),
        RestrictionFamily::Diagonal,
    ) {
        Ok(ls) => ls,
        Err(_) => return false,
    };
    learnable.init_identity();

    let n_per = params_per_endpoint(stalk_dim, RestrictionFamily::Diagonal);
    for (i, &(u, v)) in adj.iter().enumerate() {
        if u >= num_nodes || v >= num_nodes {
            continue;
        }
        let fu = &node_features[u];
        let fv = &node_features[v];
        if fu.len() < stalk_dim || fv.len() < stalk_dim {
            continue;
        }
        let base = 2 * i * n_per;
        for dim in 0..stalk_dim {
            let diff = (fu[dim] - fv[dim]).abs();
            // Scale inversely with difference: similar => 1, different => smaller
            let scale = 1.0 / (1.0 + diff);
            learnable.params_mut()[base + dim] = scale;
            learnable.params_mut()[base + n_per + dim] = scale;
        }
    }

    let diag_lap = learnable.laplacian();

    // Compare spectral gaps.
    let trivial_gap = spectral_gap(&trivial_lap);
    let diag_gap = spectral_gap(&diag_lap);

    if trivial_gap < 1e-12 {
        return false;
    }

    // If the diagonal sheaf achieves a substantially different spectral gap,
    // non-trivial maps are beneficial.
    let ratio = (diag_gap / trivial_gap - 1.0).abs();
    ratio > 0.5
}

/// Smallest nonzero eigenvalue of a symmetric matrix.
fn spectral_gap(mat: &Mat<f64>) -> f64 {
    let n = mat.nrows();
    if n == 0 {
        return 0.0;
    }
    let eig = mat
        .as_ref()
        .selfadjoint_eigendecomposition(faer::Side::Lower);
    let s = eig.s();
    let mut min_nonzero = f64::MAX;
    for i in 0..n {
        let v = s.column_vector().read(i);
        if v.abs() > 1e-10 && v < min_nonzero {
            min_nonzero = v;
        }
    }
    if min_nonzero == f64::MAX {
        0.0
    } else {
        min_nonzero
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_edges() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2), (0, 2)]
    }

    fn path_edges() -> Vec<(usize, usize)> {
        vec![(0, 1), (1, 2)]
    }

    #[test]
    fn param_count_diagonal() {
        let d = 3;
        let edges = triangle_edges();
        let ls = LearnableSheaf::new(3, d, edges.clone(), RestrictionFamily::Diagonal).unwrap();
        // 2 endpoints per edge * |E| edges * d params per endpoint
        assert_eq!(ls.num_params(), 2 * edges.len() * d);
    }

    #[test]
    fn param_count_orthogonal() {
        let d = 4;
        let edges = path_edges();
        let ls = LearnableSheaf::new(3, d, edges.clone(), RestrictionFamily::Orthogonal).unwrap();
        // 2 * |E| * d*(d-1)/2
        assert_eq!(ls.num_params(), 2 * edges.len() * d * (d - 1) / 2);
    }

    #[test]
    fn param_count_general() {
        let d = 3;
        let edges = triangle_edges();
        let ls = LearnableSheaf::new(3, d, edges.clone(), RestrictionFamily::General).unwrap();
        // 2 * |E| * d^2
        assert_eq!(ls.num_params(), 2 * edges.len() * d * d);
    }

    #[test]
    fn diagonal_identity_produces_identity_maps() {
        let d = 3;
        let mut ls =
            LearnableSheaf::new(3, d, triangle_edges(), RestrictionFamily::Diagonal).unwrap();
        ls.init_identity();
        let maps = ls.build_maps();
        for (src, tgt) in &maps {
            assert_eq!(src.len(), d * d);
            // Check it is the identity matrix (column-major)
            for row in 0..d {
                for col in 0..d {
                    let expected = if row == col { 1.0 } else { 0.0 };
                    assert!(
                        (src[col * d + row] - expected).abs() < 1e-12,
                        "src[{row},{col}] = {}, expected {expected}",
                        src[col * d + row]
                    );
                    assert!(
                        (tgt[col * d + row] - expected).abs() < 1e-12,
                        "tgt[{row},{col}] = {}, expected {expected}",
                        tgt[col * d + row]
                    );
                }
            }
        }
    }

    #[test]
    fn orthogonal_maps_are_orthogonal() {
        let d = 4;
        let mut ls =
            LearnableSheaf::new(3, d, triangle_edges(), RestrictionFamily::Orthogonal).unwrap();
        ls.init_random(42);
        let maps = ls.build_maps();
        for (src, tgt) in &maps {
            // Check M^T M = I for both maps
            for map in [src, tgt] {
                let mtm = mat_mul_col_major_transpose_left(d, map, map);
                for row in 0..d {
                    for col in 0..d {
                        let expected = if row == col { 1.0 } else { 0.0 };
                        assert!(
                            (mtm[col * d + row] - expected).abs() < 1e-8,
                            "M^T M [{row},{col}] = {}, expected {expected}",
                            mtm[col * d + row]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn orthogonal_d2_is_rotation() {
        let d = 2;
        let mut ls =
            LearnableSheaf::new(2, d, vec![(0, 1)], RestrictionFamily::Orthogonal).unwrap();
        // Set angle to pi/4
        let angle = std::f64::consts::FRAC_PI_4;
        ls.params_mut()[0] = angle;
        ls.params_mut()[1] = 0.0; // target is identity

        let maps = ls.build_maps();
        let src = &maps[0].0;
        let (s, c) = angle.sin_cos();
        // Column-major: [[c, s], [-s, c]]
        assert!((src[0] - c).abs() < 1e-12);
        assert!((src[1] - s).abs() < 1e-12);
        assert!((src[2] - (-s)).abs() < 1e-12);
        assert!((src[3] - c).abs() < 1e-12);
    }

    #[test]
    fn orthogonal_d3_rodrigues() {
        let d = 3;
        let mut ls =
            LearnableSheaf::new(2, d, vec![(0, 1)], RestrictionFamily::Orthogonal).unwrap();
        ls.init_random(123);
        let maps = ls.build_maps();
        let src = &maps[0].0;
        // Verify orthogonality
        let mtm = mat_mul_col_major_transpose_left(d, src, src);
        for row in 0..d {
            for col in 0..d {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (mtm[col * d + row] - expected).abs() < 1e-10,
                    "M^T M [{row},{col}] = {}, expected {expected}",
                    mtm[col * d + row]
                );
            }
        }
        // Verify det = +1 (proper rotation)
        let det = det3_col_major(src);
        assert!((det - 1.0).abs() < 1e-10, "det = {det}, expected 1.0");
    }

    #[test]
    fn general_identity_produces_graph_laplacian() {
        let d = 2;
        let edges = triangle_edges();
        let mut ls = LearnableSheaf::new(3, d, edges.clone(), RestrictionFamily::General).unwrap();
        ls.init_identity();

        // Compare against constant sheaf
        let constant = CellularSheaf::constant(3, &edges, d);
        let lap_learnable = ls.laplacian();
        let lap_constant = constant.laplacian();

        let n = lap_learnable.nrows();
        assert_eq!(n, lap_constant.nrows());
        for r in 0..n {
            for c in 0..n {
                assert!(
                    (lap_learnable[(r, c)] - lap_constant[(r, c)]).abs() < 1e-12,
                    "mismatch at ({r},{c}): learnable={}, constant={}",
                    lap_learnable[(r, c)],
                    lap_constant[(r, c)]
                );
            }
        }
    }

    #[test]
    fn to_cellular_sheaf_laplacian_matches() {
        let d = 3;
        let mut ls =
            LearnableSheaf::new(3, d, triangle_edges(), RestrictionFamily::General).unwrap();
        ls.init_random(99);

        let lap_direct = ls.laplacian();
        let lap_via_sheaf = ls.to_cellular_sheaf().laplacian();

        let n = lap_direct.nrows();
        for r in 0..n {
            for c in 0..n {
                assert!(
                    (lap_direct[(r, c)] - lap_via_sheaf[(r, c)]).abs() < 1e-12,
                    "mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn needs_learnable_maps_homophilous() {
        // All nodes have the same features => no heterophily
        let features = vec![vec![1.0, 2.0, 3.0]; 4];
        let adj = vec![(0, 1), (1, 2), (2, 3), (0, 3)];
        assert!(!needs_learnable_maps(&adj, &features, 3));
    }

    #[test]
    fn needs_learnable_maps_empty() {
        assert!(!needs_learnable_maps(&[], &[], 3));
        assert!(!needs_learnable_maps(&[(0, 1)], &[vec![1.0]], 0));
    }

    #[test]
    fn orthogonal_identity_init() {
        let d = 3;
        let mut ls =
            LearnableSheaf::new(3, d, triangle_edges(), RestrictionFamily::Orthogonal).unwrap();
        ls.init_identity();
        // All params should be zero => exp(0) = I
        for &p in ls.params() {
            assert!((p).abs() < 1e-15);
        }
        let maps = ls.build_maps();
        for (src, tgt) in &maps {
            for row in 0..d {
                for col in 0..d {
                    let expected = if row == col { 1.0 } else { 0.0 };
                    assert!(
                        (src[col * d + row] - expected).abs() < 1e-12,
                        "src[{row},{col}] = {}",
                        src[col * d + row]
                    );
                    assert!(
                        (tgt[col * d + row] - expected).abs() < 1e-12,
                        "tgt[{row},{col}] = {}",
                        tgt[col * d + row]
                    );
                }
            }
        }
    }

    #[test]
    fn orthogonal_d5_pade_is_orthogonal() {
        // d=5 exercises the general Pade code path (d >= 4)
        let d = 5;
        let mut ls =
            LearnableSheaf::new(2, d, vec![(0, 1)], RestrictionFamily::Orthogonal).unwrap();
        ls.init_random(77);
        let maps = ls.build_maps();
        for (src, tgt) in &maps {
            for map in [src, tgt] {
                let mtm = mat_mul_col_major_transpose_left(d, map, map);
                for row in 0..d {
                    for col in 0..d {
                        let expected = if row == col { 1.0 } else { 0.0 };
                        assert!(
                            (mtm[col * d + row] - expected).abs() < 1e-6,
                            "d=5 M^T M [{row},{col}] = {}, expected {expected}",
                            mtm[col * d + row]
                        );
                    }
                }
            }
        }
    }

    // --- Test helpers ---

    /// Compute A^T B in column-major layout (both d x d).
    fn mat_mul_col_major_transpose_left(d: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; d * d];
        for col in 0..d {
            for row in 0..d {
                let mut val = 0.0;
                for k in 0..d {
                    // a^T[row, k] = a[k, row] at a[row * d + k] ... no.
                    // column-major: a(k, row) is at a[row * d + k]
                    val += a[row * d + k] * b[col * d + k];
                }
                c[col * d + row] = val;
            }
        }
        c
    }

    /// Determinant of a 3x3 column-major matrix.
    fn det3_col_major(m: &[f64]) -> f64 {
        // column-major: m(r,c) = m[c*3 + r]
        let a = |r: usize, c: usize| m[c * 3 + r];
        a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
            - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
            + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0))
    }
}
