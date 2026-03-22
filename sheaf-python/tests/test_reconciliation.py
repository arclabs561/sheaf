import numpy as np
import pytest

import cohera


# ---------------------------------------------------------------------------
# Reconciliation -- list inputs (existing tests, adapted for numpy returns)
# ---------------------------------------------------------------------------


def test_reconcile_ols_simple_star():
    """OLS reconciliation on a 2-level hierarchy (root + 3 leaves)."""
    s = cohera.simple_star_matrix(3)
    # Base forecasts: root=12, leaves=3,4,5 (incoherent: 12 != 3+4+5)
    base = [12.0, 3.0, 4.0, 5.0]
    result = cohera.reconcile(s, base, method="ols")
    assert isinstance(result, np.ndarray)
    assert len(result) == 4
    # After reconciliation, root should equal sum of leaves
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


def test_reconcile_wls():
    """WLS reconciliation with diagonal weights."""
    s = cohera.simple_star_matrix(3)
    base = [12.0, 3.0, 4.0, 5.0]
    weights = [1.0, 1.0, 1.0, 1.0]
    result = cohera.reconcile(s, base, method="wls", weights=weights)
    assert isinstance(result, np.ndarray)
    assert len(result) == 4
    # Coherence check
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


def test_reconcile_mint():
    """MinTrace reconciliation with identity covariance."""
    s = cohera.simple_star_matrix(3)
    base = [12.0, 3.0, 4.0, 5.0]
    cov = np.eye(4).tolist()
    result = cohera.reconcile(s, base, method="mint", covariance=cov)
    assert isinstance(result, np.ndarray)
    assert len(result) == 4
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


def test_reconcile_mint_numpy_covariance():
    """MinTrace with numpy covariance matrix."""
    s = cohera.simple_star_matrix(3)
    base = np.array([12.0, 3.0, 4.0, 5.0])
    cov = np.eye(4)
    result = cohera.reconcile(s, base, method="mint", covariance=cov)
    assert isinstance(result, np.ndarray)
    assert len(result) == 4
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


# ---------------------------------------------------------------------------
# Reconciliation -- numpy inputs
# ---------------------------------------------------------------------------


def test_reconcile_numpy_inputs():
    """reconcile() accepts numpy arrays for both matrix and forecasts."""
    s = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    base = np.array([12.0, 3.0, 4.0, 5.0])
    result = cohera.reconcile(s, base, method="ols")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


def test_reconcile_mixed_inputs():
    """reconcile() accepts list matrix + numpy forecasts."""
    s = [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    base = np.array([12.0, 3.0, 4.0, 5.0])
    result = cohera.reconcile(s, base)
    assert isinstance(result, np.ndarray)
    assert abs(result[0] - sum(result[1:])) < 1e-8


# ---------------------------------------------------------------------------
# simple_star_matrix
# ---------------------------------------------------------------------------


def test_simple_star_matrix():
    """Verify shape and structure of the simple star summing matrix."""
    s = cohera.simple_star_matrix(3)
    assert isinstance(s, np.ndarray)
    assert s.shape == (4, 3)
    # Row 0 (root) should be all 1s
    np.testing.assert_array_equal(s[0], [1.0, 1.0, 1.0])
    # Rows 1-3 should be identity
    np.testing.assert_array_equal(s[1:], np.eye(3))


# ---------------------------------------------------------------------------
# HierarchicalConformal
# ---------------------------------------------------------------------------


def test_conformal_repr_uncalibrated():
    """__repr__ before calibration."""
    s = cohera.simple_star_matrix(3)
    hc = cohera.HierarchicalConformal(s, method="ols")
    r = repr(hc)
    assert "ols" in r
    assert "calibrated=false" in r


def test_conformal_calibrate_predict():
    """Full calibrate-then-predict workflow."""
    s = cohera.simple_star_matrix(2)
    hc = cohera.HierarchicalConformal(s, method="ols")

    # Calibration data: m=3 (root + 2 leaves), n_calib=10
    rng = np.random.default_rng(42)
    leaves = rng.standard_normal((2, 10))
    y_calib = np.vstack([leaves.sum(axis=0, keepdims=True), leaves])
    y_hat_calib = y_calib + rng.standard_normal((3, 10)) * 0.1

    hc.calibrate(y_calib, y_hat_calib, alpha=0.1)
    assert "calibrated=true" in repr(hc)

    # Predict on a new point
    y_hat = np.array([5.0, 2.0, 3.0])
    lower, upper = hc.predict(y_hat)
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    # Lower should be <= upper
    assert np.all(lower <= upper + 1e-12)


def test_conformal_numpy_constructor():
    """HierarchicalConformal accepts numpy summing matrix."""
    s = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    hc = cohera.HierarchicalConformal(s)
    assert "ols" in repr(hc)


# ---------------------------------------------------------------------------
# Metrics -- list inputs
# ---------------------------------------------------------------------------


def test_nmi_perfect_match():
    """Identical labels should give NMI = 1.0."""
    labels = [0, 0, 1, 1, 2, 2]
    assert abs(cohera.nmi(labels, labels) - 1.0) < 0.01


def test_ari_random_labels():
    """Unrelated labels should give ARI near 0."""
    a = [0, 0, 0, 0, 1, 1, 1, 1]
    b = [0, 1, 0, 1, 0, 1, 0, 1]
    score = cohera.ari(a, b)
    assert abs(score) < 0.3


# ---------------------------------------------------------------------------
# Metrics -- numpy inputs
# ---------------------------------------------------------------------------


def test_nmi_numpy():
    """nmi() accepts numpy int arrays."""
    a = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    b = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert abs(cohera.nmi(a, b) - 1.0) < 0.01


def test_ari_numpy_int32():
    """ari() accepts int32 numpy arrays."""
    a = np.array([0, 0, 1, 1], dtype=np.int32)
    b = np.array([0, 0, 1, 1], dtype=np.int32)
    assert abs(cohera.ari(a, b) - 1.0) < 0.01


def test_v_measure_numpy():
    """v_measure() with numpy arrays."""
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 1])
    assert cohera.v_measure(a, b) > 0.99


def test_purity_numpy():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 1])
    assert cohera.purity(a, b) > 0.99


def test_homogeneity_completeness_numpy():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 1])
    assert cohera.homogeneity(a, b) > 0.99
    assert cohera.completeness(a, b) > 0.99


def test_fowlkes_mallows_numpy():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 0, 1, 1])
    assert cohera.fowlkes_mallows(a, b) > 0.99


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------


def test_leiden_basic():
    """Leiden on a triangle should put all nodes in one community."""
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    communities = cohera.leiden(edges, n_nodes=3, seed=42)
    assert isinstance(communities, np.ndarray)
    assert communities.dtype == np.int64
    assert len(communities) == 3
    assert communities[0] == communities[1] == communities[2]


def test_louvain_basic():
    """Louvain on a triangle."""
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    communities = cohera.louvain(edges, n_nodes=3)
    assert isinstance(communities, np.ndarray)
    assert len(communities) == 3
    assert communities[0] == communities[1] == communities[2]


def test_label_propagation_basic():
    """Label propagation on a triangle."""
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    communities = cohera.label_propagation(edges, n_nodes=3)
    assert isinstance(communities, np.ndarray)
    assert len(communities) == 3
    assert communities[0] == communities[1] == communities[2]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_reconcile_bad_method():
    s = cohera.simple_star_matrix(2)
    with pytest.raises(ValueError, match="unknown method"):
        cohera.reconcile(s, [3.0, 1.0, 2.0], method="bad")


def test_reconcile_wls_missing_weights():
    s = cohera.simple_star_matrix(2)
    with pytest.raises(ValueError, match="weights required"):
        cohera.reconcile(s, [3.0, 1.0, 2.0], method="wls")


def test_reconcile_mint_missing_covariance():
    s = cohera.simple_star_matrix(2)
    with pytest.raises(ValueError, match="covariance required"):
        cohera.reconcile(s, [3.0, 1.0, 2.0], method="mint")


# ---------------------------------------------------------------------------
# Dendrogram
# ---------------------------------------------------------------------------


def test_dendrogram_basic():
    """Build a 4-leaf dendrogram and verify structure."""
    # 4 leaves: merge (0,1) at 0.5, (2,3) at 0.7, then (4,5) at 1.0
    merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
    dendro = cohera.Dendrogram(merges, n_items=4)
    assert dendro.n_items == 4
    assert dendro.n_merges == 3
    assert len(dendro) == 3


def test_dendrogram_repr():
    merges = [(0, 1, 0.5, 2)]
    dendro = cohera.Dendrogram(merges, n_items=3)
    r = repr(dendro)
    assert "n_items=3" in r
    assert "n_merges=1" in r


def test_dendrogram_distances():
    merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
    dendro = cohera.Dendrogram(merges, n_items=4)
    dists = dendro.distances()
    assert isinstance(dists, np.ndarray)
    assert dists.dtype == np.float64
    np.testing.assert_allclose(dists, [0.5, 0.7, 1.0])


def test_dendrogram_cut_at_distance():
    """Cut at various thresholds."""
    merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
    dendro = cohera.Dendrogram(merges, n_items=4)

    # Cut below first merge: 4 clusters
    labels = dendro.cut_at_distance(0.3)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 4
    assert len(set(labels.tolist())) == 4

    # Cut between first and second merge: 3 clusters (0,1 merged)
    labels = dendro.cut_at_distance(0.6)
    assert len(set(labels.tolist())) == 3
    assert labels[0] == labels[1]  # 0 and 1 merged

    # Cut above all: 1 cluster
    labels = dendro.cut_at_distance(2.0)
    assert len(set(labels.tolist())) == 1


def test_dendrogram_cut_to_k():
    """Cut to k clusters."""
    merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
    dendro = cohera.Dendrogram(merges, n_items=4)

    labels_4 = dendro.cut_to_k(4)
    assert len(set(labels_4.tolist())) == 4

    labels_2 = dendro.cut_to_k(2)
    assert len(set(labels_2.tolist())) == 2

    labels_1 = dendro.cut_to_k(1)
    assert len(set(labels_1.tolist())) == 1


# ---------------------------------------------------------------------------
# kNN graph
# ---------------------------------------------------------------------------


def _normalize(vecs):
    """L2-normalize rows (vicinity HNSW requires unit vectors)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def test_knn_graph_basic():
    """Build kNN graph from clustered embeddings."""
    # Two clusters in 3D, L2-normalized
    embeddings = _normalize(np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.95, 0.05, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.05, 0.95, 0.0],
    ], dtype=np.float32))
    edges = cohera.knn_graph(embeddings, k=2)
    assert isinstance(edges, list)
    assert len(edges) > 0
    for src, tgt, wt in edges:
        assert isinstance(src, int)
        assert isinstance(tgt, int)
        assert isinstance(wt, float)
        assert wt > 0


def test_knn_graph_f64_input():
    """knn_graph accepts float64 arrays (downcast internally)."""
    embeddings = _normalize(np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ], dtype=np.float64))
    edges = cohera.knn_graph(embeddings, k=1)
    assert len(edges) > 0


def test_knn_graph_list_input():
    """knn_graph accepts list[list[float]]."""
    raw = _normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    embeddings = raw.tolist()
    edges = cohera.knn_graph(embeddings, k=1)
    assert len(edges) > 0


def test_knn_graph_to_leiden_pipeline():
    """End-to-end: embeddings -> knn graph -> leiden."""
    embeddings = _normalize(np.array([
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.95, 0.05, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.05, 0.95, 0.0],
    ], dtype=np.float32))
    edges = cohera.knn_graph(embeddings, k=2)
    labels = cohera.leiden(edges, n_nodes=len(embeddings), seed=42)
    assert len(labels) == 6
    # Items 0-2 should be in one cluster, 3-5 in another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


# ---------------------------------------------------------------------------
# summing_matrix_from_tree
# ---------------------------------------------------------------------------


def test_summing_matrix_from_tree_basic():
    """Build summing matrix from a merge tree."""
    # 4 leaves, 3 merges -> 7 total nodes
    merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
    s = cohera.summing_matrix_from_tree(merges, n_leaves=4)
    assert isinstance(s, np.ndarray)
    assert s.shape == (7, 4)
    # Leaf rows should be identity
    np.testing.assert_array_equal(s[:4], np.eye(4))
    # Internal node 4 (merge of 0,1): row 4 should be [1,1,0,0]
    np.testing.assert_array_equal(s[4], [1, 1, 0, 0])
    # Internal node 5 (merge of 2,3): row 5 should be [0,0,1,1]
    np.testing.assert_array_equal(s[5], [0, 0, 1, 1])
    # Root node 6 (merge of 4,5): row 6 should be [1,1,1,1]
    np.testing.assert_array_equal(s[6], [1, 1, 1, 1])


def test_summing_matrix_from_tree_reconcile():
    """Summing matrix from tree can be used for reconciliation."""
    merges = [(0, 1, 0.5, 2), (2, 3, 0.7, 2), (4, 5, 1.0, 4)]
    s = cohera.summing_matrix_from_tree(merges, n_leaves=4)
    # 7 base forecasts (4 leaves + 3 internal)
    base = [3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 20.0]
    result = cohera.reconcile(s, base, method="ols")
    assert len(result) == 7
    # Coherence: internal nodes should equal sum of their children
    assert abs(result[4] - (result[0] + result[1])) < 1e-8
    assert abs(result[5] - (result[2] + result[3])) < 1e-8
    assert abs(result[6] - sum(result[:4])) < 1e-8
