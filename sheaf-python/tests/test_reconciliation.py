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
