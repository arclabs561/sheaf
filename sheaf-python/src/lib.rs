use pyo3::prelude::*;

use faer::Mat;
use petgraph::graph::UnGraph;
use sheaf::community::{CommunityDetection, LabelPropagation, Leiden, Louvain};
use sheaf::reconciliation::{ReconciliationMethod, SummingMatrix};

// ---------------------------------------------------------------------------
// Helpers: Python list <-> faer::Mat
// ---------------------------------------------------------------------------

/// Convert a Python 2D list (list[list[float]]) to a faer Mat<f64>.
/// Expects row-major: outer list = rows, inner list = columns.
fn list2d_to_mat(rows: Vec<Vec<f64>>) -> PyResult<Mat<f64>> {
    if rows.is_empty() {
        return Ok(Mat::<f64>::zeros(0, 0));
    }
    let nrows = rows.len();
    let ncols = rows[0].len();
    for (i, row) in rows.iter().enumerate() {
        if row.len() != ncols {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "row {} has length {}, expected {}",
                i,
                row.len(),
                ncols
            )));
        }
    }
    let mut mat = Mat::<f64>::zeros(nrows, ncols);
    for (i, row) in rows.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            mat[(i, j)] = val;
        }
    }
    Ok(mat)
}

/// Convert a Python 1D list to a faer column vector (m x 1).
fn list1d_to_col(vals: Vec<f64>) -> Mat<f64> {
    let m = vals.len();
    let mut mat = Mat::<f64>::zeros(m, 1);
    for (i, &v) in vals.iter().enumerate() {
        mat[(i, 0)] = v;
    }
    mat
}

/// Convert a faer Mat<f64> column vector (m x 1) to a Python list.
fn col_to_list(mat: &Mat<f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| mat[(i, 0)]).collect()
}

/// Convert a faer Mat<f64> to a Python 2D list.
fn mat_to_list2d(mat: &Mat<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| mat[(i, j)]).collect())
        .collect()
}

/// Build a petgraph UnGraph from an edge list.
fn edges_to_ungraph(edges: Vec<(usize, usize, f64)>, n_nodes: usize) -> UnGraph<(), f32> {
    let mut graph = UnGraph::<(), f32>::new_undirected();
    for _ in 0..n_nodes {
        graph.add_node(());
    }
    for (i, j, w) in edges {
        let ni = petgraph::graph::NodeIndex::new(i);
        let nj = petgraph::graph::NodeIndex::new(j);
        graph.add_edge(ni, nj, w as f32);
    }
    graph
}

fn parse_method(
    method: &str,
    m: usize,
    weights: Option<Vec<f64>>,
    covariance: Option<Vec<Vec<f64>>>,
) -> PyResult<ReconciliationMethod> {
    match method {
        "ols" => Ok(ReconciliationMethod::Ols),
        "wls" => {
            let w = weights.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("weights required for wls")
            })?;
            if w.len() != m {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "weights length {} != m {}",
                    w.len(),
                    m
                )));
            }
            Ok(ReconciliationMethod::Wls { weights: w })
        }
        "mint" => {
            let cov_rows = covariance.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("covariance required for mint")
            })?;
            let cov = list2d_to_mat(cov_rows)?;
            if cov.nrows() != m || cov.ncols() != m {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "covariance shape {}x{} != {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    m,
                    m
                )));
            }
            Ok(ReconciliationMethod::MinT { covariance: cov })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown method '{}', expected 'ols', 'wls', or 'mint'",
            other
        ))),
    }
}

fn sheaf_err(e: sheaf::Error) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Reconciliation
// ---------------------------------------------------------------------------

/// Reconcile base forecasts using the structural summing matrix S.
///
/// method: "ols", "wls", or "mint".
/// weights: required for "wls" (m-dimensional diagonal weights).
/// covariance: required for "mint" (m x m covariance matrix).
#[pyfunction]
#[pyo3(signature = (summing_matrix, base_forecasts, method = "ols", weights = None, covariance = None))]
fn reconcile(
    summing_matrix: Vec<Vec<f64>>,
    base_forecasts: Vec<f64>,
    method: &str,
    weights: Option<Vec<f64>>,
    covariance: Option<Vec<Vec<f64>>>,
) -> PyResult<Vec<f64>> {
    let s_mat = list2d_to_mat(summing_matrix)?;
    let s = SummingMatrix::new(s_mat);
    let m = s.m();
    let bf = list1d_to_col(base_forecasts);
    let rm = parse_method(method, m, weights, covariance)?;
    let result = sheaf::reconciliation::reconcile(&s, &bf, rm).map_err(sheaf_err)?;
    Ok(col_to_list(&result))
}

/// Generate the summing matrix for a simple 2-level hierarchy (root + n leaves).
#[pyfunction]
fn simple_star_matrix(n_leaves: usize) -> Vec<Vec<f64>> {
    let s = SummingMatrix::simple_star(n_leaves);
    let mat_ref = s.as_ref();
    (0..mat_ref.nrows())
        .map(|i| (0..mat_ref.ncols()).map(|j| mat_ref[(i, j)]).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Conformal prediction
// ---------------------------------------------------------------------------

/// Hierarchical conformal predictor.
#[pyclass]
struct HierarchicalConformal {
    inner: sheaf::HierarchicalConformal,
}

#[pymethods]
impl HierarchicalConformal {
    #[new]
    #[pyo3(signature = (summing_matrix, method = "ols"))]
    fn new(summing_matrix: Vec<Vec<f64>>, method: &str) -> PyResult<Self> {
        let s_mat = list2d_to_mat(summing_matrix)?;
        let s = SummingMatrix::new(s_mat);
        let m = s.m();
        let rm = parse_method(method, m, None, None)?;
        Ok(Self {
            inner: sheaf::HierarchicalConformal::new(s, rm),
        })
    }

    /// Calibrate using a calibration set.
    ///
    /// y_calib: list[list[float]] -- true values, shape (m, n_calib)
    /// y_hat_calib: list[list[float]] -- base forecasts, shape (m, n_calib)
    /// alpha: float -- miscoverage level (e.g. 0.1 for 90% coverage)
    fn calibrate(
        &mut self,
        y_calib: Vec<Vec<f64>>,
        y_hat_calib: Vec<Vec<f64>>,
        alpha: f64,
    ) -> PyResult<()> {
        let yc = list2d_to_mat(y_calib)?;
        let yhc = list2d_to_mat(y_hat_calib)?;
        self.inner.calibrate(&yc, &yhc, alpha).map_err(sheaf_err)
    }

    /// Predict intervals for new base forecasts.
    ///
    /// y_hat: list[float] -- base forecast vector (m-dimensional)
    /// Returns (lower, upper) bounds as lists.
    fn predict(&self, y_hat: Vec<f64>) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let yh = list1d_to_col(y_hat);
        let (lower, upper) = self.inner.predict_intervals(&yh).map_err(sheaf_err)?;
        Ok((col_to_list(&lower), col_to_list(&upper)))
    }
}

// ---------------------------------------------------------------------------
// Community detection
// ---------------------------------------------------------------------------

/// Leiden community detection.
///
/// edges: list of (source, target, weight) tuples.
/// n_nodes: total number of nodes in the graph.
/// resolution: modularity resolution parameter (default 1.0).
/// seed: optional random seed.
#[pyfunction]
#[pyo3(signature = (edges, n_nodes, resolution = 1.0, seed = None))]
fn leiden(
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
    resolution: f64,
    seed: Option<u64>,
) -> PyResult<Vec<usize>> {
    let graph = edges_to_ungraph(edges, n_nodes);
    let mut det = Leiden::new().with_resolution(resolution);
    if let Some(s) = seed {
        det = det.with_seed(s);
    }
    det.detect_weighted(&graph).map_err(sheaf_err)
}

/// Louvain community detection.
///
/// edges: list of (source, target, weight) tuples.
/// n_nodes: total number of nodes in the graph.
/// resolution: modularity resolution parameter (default 1.0).
#[pyfunction]
#[pyo3(signature = (edges, n_nodes, resolution = 1.0))]
fn louvain(edges: Vec<(usize, usize, f64)>, n_nodes: usize, resolution: f64) -> PyResult<Vec<usize>> {
    let graph = edges_to_ungraph(edges, n_nodes);
    let det = Louvain::new().with_resolution(resolution);
    det.detect(&graph).map_err(sheaf_err)
}

/// Label propagation community detection.
///
/// edges: list of (source, target, weight) tuples.
/// n_nodes: total number of nodes in the graph.
#[pyfunction]
#[pyo3(signature = (edges, n_nodes))]
fn label_propagation(edges: Vec<(usize, usize, f64)>, n_nodes: usize) -> PyResult<Vec<usize>> {
    let graph = edges_to_ungraph(edges, n_nodes);
    let det = LabelPropagation::new();
    det.detect(&graph).map_err(sheaf_err)
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Normalized Mutual Information between two clusterings.
#[pyfunction]
fn nmi(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::nmi(&labels_a, &labels_b)
}

/// Adjusted Rand Index between two clusterings.
#[pyfunction]
fn ari(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::ari(&labels_a, &labels_b)
}

/// V-Measure: harmonic mean of homogeneity and completeness.
#[pyfunction]
fn v_measure(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::v_measure(&labels_a, &labels_b)
}

/// Purity of clustering with respect to ground truth.
#[pyfunction]
fn purity(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::purity(&labels_a, &labels_b)
}

/// Homogeneity: each cluster contains only members of a single class.
#[pyfunction]
fn homogeneity(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::homogeneity(&labels_a, &labels_b)
}

/// Completeness: all members of a given class are assigned to the same cluster.
#[pyfunction]
fn completeness(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::completeness(&labels_a, &labels_b)
}

/// Fowlkes-Mallows Index: geometric mean of pairwise precision and recall.
#[pyfunction]
fn fowlkes_mallows(labels_a: Vec<usize>, labels_b: Vec<usize>) -> f64 {
    sheaf::metrics::fowlkes_mallows(&labels_a, &labels_b)
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn cohera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Reconciliation
    m.add_function(wrap_pyfunction!(reconcile, m)?)?;
    m.add_function(wrap_pyfunction!(simple_star_matrix, m)?)?;

    // Conformal
    m.add_class::<HierarchicalConformal>()?;

    // Community detection
    m.add_function(wrap_pyfunction!(leiden, m)?)?;
    m.add_function(wrap_pyfunction!(louvain, m)?)?;
    m.add_function(wrap_pyfunction!(label_propagation, m)?)?;

    // Metrics
    m.add_function(wrap_pyfunction!(nmi, m)?)?;
    m.add_function(wrap_pyfunction!(ari, m)?)?;
    m.add_function(wrap_pyfunction!(v_measure, m)?)?;
    m.add_function(wrap_pyfunction!(purity, m)?)?;
    m.add_function(wrap_pyfunction!(homogeneity, m)?)?;
    m.add_function(wrap_pyfunction!(completeness, m)?)?;
    m.add_function(wrap_pyfunction!(fowlkes_mallows, m)?)?;

    Ok(())
}
