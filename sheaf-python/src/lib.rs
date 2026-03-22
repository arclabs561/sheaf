use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use faer::Mat;
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use sheaf::community::{
    knn_graph_from_embeddings, CommunityDetection, LabelPropagation, Leiden, Louvain,
};
use sheaf::hierarchy::{Dendrogram as RustDendrogram, HierarchyTree};
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
fn edges_to_ungraph(edges: Vec<(usize, usize, f64)>, n_nodes: usize) -> PyResult<UnGraph<(), f32>> {
    for &(i, j, _) in edges.iter() {
        if i >= n_nodes || j >= n_nodes {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("edge ({}, {}) exceeds n_nodes={}", i, j, n_nodes)
            ));
        }
    }
    let mut graph = UnGraph::<(), f32>::new_undirected();
    for _ in 0..n_nodes {
        graph.add_node(());
    }
    for (i, j, w) in edges {
        let ni = petgraph::graph::NodeIndex::new(i);
        let nj = petgraph::graph::NodeIndex::new(j);
        graph.add_edge(ni, nj, w as f32);
    }
    Ok(graph)
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
        let vals = arr.as_array().to_vec();
        for &v in vals.iter() {
            if v < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("negative label {} not supported in metrics (filter noise labels first)", v)
                ));
            }
        }
        return Ok(vals.iter().map(|&v| v as usize).collect());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<'_, i32>>() {
        let vals = arr.as_array().to_vec();
        for &v in vals.iter() {
            if v < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("negative label {} not supported in metrics (filter noise labels first)", v)
                ));
            }
        }
        return Ok(vals.iter().map(|&v| v as usize).collect());
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
        if !self.calibrated {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "must call calibrate() before predict()"
            ));
        }
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
    let graph = edges_to_ungraph(edges, n_nodes)?;
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
    let graph = edges_to_ungraph(edges, n_nodes)?;
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
    let graph = edges_to_ungraph(edges, n_nodes)?;
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
// Dendrogram
// ---------------------------------------------------------------------------

/// Dendrogram from hierarchical clustering merge history.
///
/// Represents nested cluster structure from agglomerative clustering.
/// Supports cutting at a distance threshold or to a target number of clusters.
///
/// Args:
///     merges: List of (cluster_a, cluster_b, distance, size) merge operations.
///         Merges should be in ascending distance order.
///     n_items: Number of original (leaf) items.
#[pyclass]
struct Dendrogram {
    inner: RustDendrogram,
}

#[pymethods]
impl Dendrogram {
    #[new]
    fn new(merges: Vec<(usize, usize, f64, usize)>, n_items: usize) -> Self {
        let mut dendro = RustDendrogram::new(n_items);
        for (a, b, dist, size) in merges {
            dendro.add_merge(a, b, dist, size);
        }
        Self { inner: dendro }
    }

    /// Cut the dendrogram at a distance threshold.
    ///
    /// All merges with distance > threshold are severed, producing
    /// separate clusters.
    ///
    /// Args:
    ///     threshold: Distance threshold for cutting.
    ///
    /// Returns:
    ///     numpy.ndarray: Cluster label for each item, dtype int64.
    fn cut_at_distance<'py>(
        &self,
        py: Python<'py>,
        threshold: f64,
    ) -> Bound<'py, PyArray1<i64>> {
        let labels = self.inner.cut_at_distance(threshold);
        let i64_labels: Vec<i64> = labels.into_iter().map(|v| v as i64).collect();
        i64_labels.into_pyarray(py)
    }

    /// Cut the dendrogram to produce exactly k clusters.
    ///
    /// Args:
    ///     k: Desired number of clusters.
    ///
    /// Returns:
    ///     numpy.ndarray: Cluster label for each item, dtype int64.
    fn cut_to_k<'py>(
        &self,
        py: Python<'py>,
        k: usize,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = self.inner.cut_to_k(k).map_err(sheaf_err)?;
        let i64_labels: Vec<i64> = labels.into_iter().map(|v| v as i64).collect();
        Ok(i64_labels.into_pyarray(py))
    }

    /// Number of original (leaf) items.
    #[getter]
    fn n_items(&self) -> usize {
        self.inner.n_items()
    }

    /// Number of merge operations recorded.
    #[getter]
    fn n_merges(&self) -> usize {
        self.inner.n_merges()
    }

    /// Merge distances in order.
    ///
    /// Returns:
    ///     numpy.ndarray: Distance at each merge step, dtype float64.
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.distances().into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.n_merges()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dendrogram(n_items={}, n_merges={})",
            self.inner.n_items(),
            self.inner.n_merges()
        )
    }
}

// ---------------------------------------------------------------------------
// kNN graph
// ---------------------------------------------------------------------------

/// Build a k-nearest-neighbor graph from embeddings.
///
/// Uses HNSW for approximate nearest neighbor search. The resulting edge
/// list can be passed directly to leiden/louvain/label_propagation.
///
/// Args:
///     embeddings: 2D array of shape (n, dim), dtype float32. Each row is
///         an embedding vector. Accepts numpy ndarray or list[list[float]].
///     k: Number of neighbors per node (default 10).
///
/// Returns:
///     list[tuple[int, int, float]]: Edge list as (source, target, weight).
///         Weights are similarity scores (higher = more similar).
#[pyfunction]
#[pyo3(signature = (embeddings, k = 10))]
fn knn_graph(
    embeddings: &Bound<'_, pyo3::PyAny>,
    k: usize,
) -> PyResult<Vec<(usize, usize, f64)>> {
    // Accept numpy f32 2D array or list[list[float]]
    let vecs: Vec<Vec<f32>> = if let Ok(arr) = embeddings.extract::<PyReadonlyArray2<'_, f32>>() {
        let shape = arr.shape();
        let nrows = shape[0];
        let ncols = shape[1];
        let arr_ref = arr.as_array();
        (0..nrows)
            .map(|i| (0..ncols).map(|j| arr_ref[[i, j]]).collect())
            .collect()
    } else if let Ok(arr) = embeddings.extract::<PyReadonlyArray2<'_, f64>>() {
        // Accept f64 and downcast
        let shape = arr.shape();
        let nrows = shape[0];
        let ncols = shape[1];
        let arr_ref = arr.as_array();
        (0..nrows)
            .map(|i| (0..ncols).map(|j| arr_ref[[i, j]] as f32).collect())
            .collect()
    } else {
        // list[list[float]]
        let rows: Vec<Vec<f64>> = embeddings.extract()?;
        rows.into_iter()
            .map(|row| row.into_iter().map(|v| v as f32).collect())
            .collect()
    };

    let graph = knn_graph_from_embeddings(&vecs, k).map_err(sheaf_err)?;

    // Convert petgraph edges to tuples
    let edges: Vec<(usize, usize, f64)> = graph
        .edge_references()
        .map(|e| {
            (
                e.source().index(),
                e.target().index(),
                *e.weight() as f64,
            )
        })
        .collect();

    Ok(edges)
}

// ---------------------------------------------------------------------------
// SummingMatrix from tree
// ---------------------------------------------------------------------------

/// Build a summing matrix from a tree defined by parent-child merge history.
///
/// Constructs a HierarchyTree from dendrogram-style merges, then extracts
/// the structural summing matrix S. Each merge (a, b, height, size) creates
/// an internal node whose children are clusters a and b.
///
/// Args:
///     merges: List of (cluster_a, cluster_b, height, size) merge operations.
///     n_leaves: Number of leaf (bottom-level) nodes.
///
/// Returns:
///     numpy.ndarray: Summing matrix with shape (n_total, n_leaves), where
///         n_total = n_leaves + len(merges).
#[pyfunction]
fn summing_matrix_from_tree(
    py: Python<'_>,
    merges: Vec<(usize, usize, f64, usize)>,
    n_leaves: usize,
) -> Bound<'_, PyArray2<f64>> {
    let tree = HierarchyTree::from_merges(&merges, n_leaves);
    let s = tree.summing_matrix();
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
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn cohera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;

    // Reconciliation
    m.add_function(wrap_pyfunction!(reconcile, m)?)?;
    m.add_function(wrap_pyfunction!(simple_star_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(summing_matrix_from_tree, m)?)?;

    // Hierarchy
    m.add_class::<Dendrogram>()?;

    // Conformal
    m.add_class::<HierarchicalConformal>()?;

    // Community detection
    m.add_function(wrap_pyfunction!(leiden, m)?)?;
    m.add_function(wrap_pyfunction!(louvain, m)?)?;
    m.add_function(wrap_pyfunction!(label_propagation, m)?)?;
    m.add_function(wrap_pyfunction!(knn_graph, m)?)?;

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
