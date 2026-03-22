use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use faer::Mat;
use petgraph::graph::UnGraph;
use sheaf::community::{CommunityDetection, LabelPropagation, Leiden, Louvain};
use sheaf::reconciliation::{ReconciliationMethod, SummingMatrix};

// ---------------------------------------------------------------------------
// Helpers: Python list / numpy <-> faer::Mat
// ---------------------------------------------------------------------------

/// Convert a Python 2D list (list[list[float]]) to a faer Mat<f64>.
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

/// Convert a numpy 2D array to a faer Mat<f64>.
fn numpy2d_to_mat(arr: PyReadonlyArray2<'_, f64>) -> Mat<f64> {
    let shape = arr.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    let mut mat = Mat::<f64>::zeros(nrows, ncols);
    let arr_ref = arr.as_array();
    for i in 0..nrows {
        for j in 0..ncols {
            mat[(i, j)] = arr_ref[[i, j]];
        }
    }
    mat
}

/// Convert a numpy 1D array to a faer column vector (m x 1).
fn numpy1d_to_col(arr: PyReadonlyArray1<'_, f64>) -> Mat<f64> {
    let arr_ref = arr.as_array();
    let m = arr_ref.len();
    let mut mat = Mat::<f64>::zeros(m, 1);
    for i in 0..m {
        mat[(i, 0)] = arr_ref[i];
    }
    mat
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

/// Convert a faer Mat<f64> column vector (m x 1) to a Vec.
fn col_to_vec(mat: &Mat<f64>) -> Vec<f64> {
    (0..mat.nrows()).map(|i| mat[(i, 0)]).collect()
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
    covariance: Option<Mat<f64>>,
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
            let cov = covariance.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("covariance required for mint")
            })?;
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

/// Extract a Vec<usize> from either a Python list or a numpy int array.
fn extract_labels(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<usize>> {
    // Try numpy first
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i64>>() {
        return Ok(arr.as_array().iter().map(|&v| v as usize).collect());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
        return Ok(arr.as_array().iter().map(|&v| v as usize).collect());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, u64>>() {
        return Ok(arr.as_array().iter().map(|&v| v as usize).collect());
    }
    // Fall back to list
    obj.extract::<Vec<usize>>()
}

/// Extract a Mat<f64> from either a 2D list or numpy array.
fn extract_mat2d(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Mat<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        return Ok(numpy2d_to_mat(arr));
    }
    let rows: Vec<Vec<f64>> = obj.extract()?;
    list2d_to_mat(rows)
}

/// Extract a column vector from either a 1D list or numpy array.
fn extract_vec1d(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Mat<f64>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        return Ok(numpy1d_to_col(arr));
    }
    let vals: Vec<f64> = obj.extract()?;
    Ok(list1d_to_col(vals))
}

// ---------------------------------------------------------------------------
// Reconciliation
// ---------------------------------------------------------------------------

/// Reconcile base forecasts to satisfy hierarchical constraints.
///
/// Given a summing matrix S and incoherent base forecasts, produces
/// coherent forecasts where aggregates equal the sum of their children.
///
/// Args:
///     summing_matrix: 2D array (m x n) defining the hierarchy. Accepts
///         numpy ndarray or list[list[float]].
///     base_forecasts: 1D array (m,) of base forecasts. Accepts numpy
///         ndarray or list[float].
///     method: Reconciliation method -- "ols", "wls", or "mint".
///     weights: Diagonal weights for "wls" (m,). Required if method="wls".
///     covariance: Covariance matrix for "mint" (m x m). Accepts numpy
///         ndarray or list[list[float]]. Required if method="mint".
///
/// Returns:
///     numpy.ndarray: Reconciled forecasts (m,).
#[pyfunction]
#[pyo3(signature = (summing_matrix, base_forecasts, method = "ols", weights = None, covariance = None))]
fn reconcile<'py>(
    py: Python<'py>,
    summing_matrix: &Bound<'py, pyo3::PyAny>,
    base_forecasts: &Bound<'py, pyo3::PyAny>,
    method: &str,
    weights: Option<Vec<f64>>,
    covariance: Option<&Bound<'py, pyo3::PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s_mat = extract_mat2d(summing_matrix)?;
    let s = SummingMatrix::new(s_mat);
    let m = s.m();
    let bf = extract_vec1d(base_forecasts)?;
    let cov = match covariance {
        Some(obj) => Some(extract_mat2d(obj)?),
        None => None,
    };
    let rm = parse_method(method, m, weights, cov)?;
    let result = sheaf::reconciliation::reconcile(&s, &bf, rm).map_err(sheaf_err)?;
    Ok(col_to_vec(&result).into_pyarray(py))
}

/// Build a summing matrix for a simple 2-level hierarchy (1 root + n leaves).
///
/// The resulting matrix has shape (n_leaves + 1, n_leaves): the first row
/// is all ones (root = sum of leaves), rows 1..n are the identity (each
/// leaf maps to itself).
///
/// Args:
///     n_leaves: Number of leaf nodes.
///
/// Returns:
///     numpy.ndarray: Summing matrix with shape (n_leaves + 1, n_leaves).
#[pyfunction]
fn simple_star_matrix(py: Python<'_>, n_leaves: usize) -> Bound<'_, PyArray2<f64>> {
    let s = SummingMatrix::simple_star(n_leaves);
    let mat_ref = s.as_ref();
    let nrows = mat_ref.nrows();
    let ncols = mat_ref.ncols();
    numpy::PyArray2::from_vec2(
        py,
        &(0..nrows)
            .map(|i| (0..ncols).map(|j| mat_ref[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Conformal prediction
// ---------------------------------------------------------------------------

/// Hierarchical conformal predictor.
///
/// Wraps forecast reconciliation with split-conformal calibration to
/// produce prediction intervals that respect the hierarchy.
///
/// Args:
///     summing_matrix: 2D array (m x n) defining the hierarchy. Accepts
///         numpy ndarray or list[list[float]].
///     method: Reconciliation method -- "ols", "wls", or "mint".
#[pyclass]
struct HierarchicalConformal {
    inner: sheaf::HierarchicalConformal,
    method_name: String,
    calibrated: bool,
}

#[pymethods]
impl HierarchicalConformal {
    #[new]
    #[pyo3(signature = (summing_matrix, method = "ols"))]
    fn new(summing_matrix: &Bound<'_, pyo3::PyAny>, method: &str) -> PyResult<Self> {
        let s_mat = extract_mat2d(summing_matrix)?;
        let s = SummingMatrix::new(s_mat);
        let m = s.m();
        let rm = parse_method(method, m, None, None)?;
        Ok(Self {
            inner: sheaf::HierarchicalConformal::new(s, rm),
            method_name: method.to_owned(),
            calibrated: false,
        })
    }

    /// Calibrate using held-out data to determine conformal quantiles.
    ///
    /// Args:
    ///     y_calib: True values, shape (m, n_calib). Accepts numpy ndarray
    ///         or list[list[float]].
    ///     y_hat_calib: Base forecasts, shape (m, n_calib). Accepts numpy
    ///         ndarray or list[list[float]].
    ///     alpha: Miscoverage level (e.g. 0.1 for 90% coverage).
    fn calibrate(
        &mut self,
        y_calib: &Bound<'_, pyo3::PyAny>,
        y_hat_calib: &Bound<'_, pyo3::PyAny>,
        alpha: f64,
    ) -> PyResult<()> {
        let yc = extract_mat2d(y_calib)?;
        let yhc = extract_mat2d(y_hat_calib)?;
        self.inner.calibrate(&yc, &yhc, alpha).map_err(sheaf_err)?;
        self.calibrated = true;
        Ok(())
    }

    /// Predict reconciled intervals for new base forecasts.
    ///
    /// Args:
    ///     y_hat: Base forecast vector (m,). Accepts numpy ndarray or
    ///         list[float].
    ///
    /// Returns:
    ///     tuple[numpy.ndarray, numpy.ndarray]: (lower, upper) bounds,
    ///         each shape (m,).
    fn predict<'py>(
        &self,
        py: Python<'py>,
        y_hat: &Bound<'py, pyo3::PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let yh = extract_vec1d(y_hat)?;
        let (lower, upper) = self.inner.predict_intervals(&yh).map_err(sheaf_err)?;
        Ok((
            col_to_vec(&lower).into_pyarray(py),
            col_to_vec(&upper).into_pyarray(py),
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "HierarchicalConformal(method='{}', calibrated={})",
            self.method_name, self.calibrated
        )
    }
}

// ---------------------------------------------------------------------------
// Community detection
// ---------------------------------------------------------------------------

/// Leiden community detection on a weighted undirected graph.
///
/// Args:
///     edges: Edge list as [(source, target, weight), ...].
///     n_nodes: Total number of nodes.
///     resolution: Modularity resolution (default 1.0). Higher values
///         produce more communities.
///     seed: Random seed for reproducibility.
///
/// Returns:
///     numpy.ndarray: Community assignment for each node, dtype int64.
#[pyfunction]
#[pyo3(signature = (edges, n_nodes, resolution = 1.0, seed = None))]
fn leiden(
    py: Python<'_>,
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
    resolution: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'_, PyArray1<i64>>> {
    let graph = edges_to_ungraph(edges, n_nodes);
    let mut det = Leiden::new().with_resolution(resolution);
    if let Some(s) = seed {
        det = det.with_seed(s);
    }
    let result = det.detect_weighted(&graph).map_err(sheaf_err)?;
    let i64_result: Vec<i64> = result.into_iter().map(|v| v as i64).collect();
    Ok(i64_result.into_pyarray(py))
}

/// Louvain community detection on a weighted undirected graph.
///
/// Args:
///     edges: Edge list as [(source, target, weight), ...].
///     n_nodes: Total number of nodes.
///     resolution: Modularity resolution (default 1.0). Higher values
///         produce more communities.
///
/// Returns:
///     numpy.ndarray: Community assignment for each node, dtype int64.
#[pyfunction]
#[pyo3(signature = (edges, n_nodes, resolution = 1.0))]
fn louvain(
    py: Python<'_>,
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
    resolution: f64,
) -> PyResult<Bound<'_, PyArray1<i64>>> {
    let graph = edges_to_ungraph(edges, n_nodes);
    let det = Louvain::new().with_resolution(resolution);
    let result = det.detect(&graph).map_err(sheaf_err)?;
    let i64_result: Vec<i64> = result.into_iter().map(|v| v as i64).collect();
    Ok(i64_result.into_pyarray(py))
}

/// Label propagation community detection on a weighted undirected graph.
///
/// Args:
///     edges: Edge list as [(source, target, weight), ...].
///     n_nodes: Total number of nodes.
///
/// Returns:
///     numpy.ndarray: Community assignment for each node, dtype int64.
#[pyfunction]
#[pyo3(signature = (edges, n_nodes))]
fn label_propagation(
    py: Python<'_>,
    edges: Vec<(usize, usize, f64)>,
    n_nodes: usize,
) -> PyResult<Bound<'_, PyArray1<i64>>> {
    let graph = edges_to_ungraph(edges, n_nodes);
    let det = LabelPropagation::new();
    let result = det.detect(&graph).map_err(sheaf_err)?;
    let i64_result: Vec<i64> = result.into_iter().map(|v| v as i64).collect();
    Ok(i64_result.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Normalized Mutual Information between two clusterings.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: NMI in [0, 1]. 1.0 means identical clusterings.
#[pyfunction]
fn nmi(labels_true: &Bound<'_, pyo3::PyAny>, labels_pred: &Bound<'_, pyo3::PyAny>) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::nmi(&a, &b))
}

/// Adjusted Rand Index between two clusterings.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: ARI in [-1, 1]. 1.0 means identical, 0.0 means random.
#[pyfunction]
fn ari(labels_true: &Bound<'_, pyo3::PyAny>, labels_pred: &Bound<'_, pyo3::PyAny>) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::ari(&a, &b))
}

/// V-Measure: harmonic mean of homogeneity and completeness.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: V-measure in [0, 1].
#[pyfunction]
fn v_measure(
    labels_true: &Bound<'_, pyo3::PyAny>,
    labels_pred: &Bound<'_, pyo3::PyAny>,
) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::v_measure(&a, &b))
}

/// Purity of clustering with respect to ground truth.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: Purity in [0, 1]. 1.0 means each cluster is pure.
#[pyfunction]
fn purity(
    labels_true: &Bound<'_, pyo3::PyAny>,
    labels_pred: &Bound<'_, pyo3::PyAny>,
) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::purity(&a, &b))
}

/// Homogeneity: each cluster contains only members of a single class.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: Homogeneity in [0, 1].
#[pyfunction]
fn homogeneity(
    labels_true: &Bound<'_, pyo3::PyAny>,
    labels_pred: &Bound<'_, pyo3::PyAny>,
) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::homogeneity(&a, &b))
}

/// Completeness: all members of a class are assigned to the same cluster.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: Completeness in [0, 1].
#[pyfunction]
fn completeness(
    labels_true: &Bound<'_, pyo3::PyAny>,
    labels_pred: &Bound<'_, pyo3::PyAny>,
) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::completeness(&a, &b))
}

/// Fowlkes-Mallows Index: geometric mean of pairwise precision and recall.
///
/// Args:
///     labels_true: Cluster assignments. Accepts numpy int array or list[int].
///     labels_pred: Cluster assignments. Accepts numpy int array or list[int].
///
/// Returns:
///     float: FMI in [0, 1]. 1.0 means identical clusterings.
#[pyfunction]
fn fowlkes_mallows(
    labels_true: &Bound<'_, pyo3::PyAny>,
    labels_pred: &Bound<'_, pyo3::PyAny>,
) -> PyResult<f64> {
    let a = extract_labels(labels_true)?;
    let b = extract_labels(labels_pred)?;
    Ok(sheaf::metrics::fowlkes_mallows(&a, &b))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn cohera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

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
