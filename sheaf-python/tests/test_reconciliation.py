import cohera


def test_reconcile_ols_simple_star():
    """OLS reconciliation on a 2-level hierarchy (root + 3 leaves)."""
    s = cohera.simple_star_matrix(3)
    # Base forecasts: root=12, leaves=3,4,5 (incoherent: 12 != 3+4+5)
    base = [12.0, 3.0, 4.0, 5.0]
    result = cohera.reconcile(s, base, method="ols")
    assert len(result) == 4
    # After reconciliation, root should equal sum of leaves
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


def test_reconcile_wls():
    """WLS reconciliation with diagonal weights."""
    s = cohera.simple_star_matrix(3)
    base = [12.0, 3.0, 4.0, 5.0]
    weights = [1.0, 1.0, 1.0, 1.0]
    result = cohera.reconcile(s, base, method="wls", weights=weights)
    assert len(result) == 4
    # Coherence check
    assert abs(result[0] - (result[1] + result[2] + result[3])) < 1e-8


def test_simple_star_matrix():
    """Verify shape and structure of the simple star summing matrix."""
    s = cohera.simple_star_matrix(3)
    # Should be (n_leaves+1) x n_leaves = 4 x 3
    assert len(s) == 4
    assert all(len(row) == 3 for row in s)
    # Row 0 (root) should be all 1s
    assert s[0] == [1.0, 1.0, 1.0]
    # Rows 1-3 should be identity
    assert s[1] == [1.0, 0.0, 0.0]
    assert s[2] == [0.0, 1.0, 0.0]
    assert s[3] == [0.0, 0.0, 1.0]


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


def test_leiden_basic():
    """Leiden on a triangle should put all nodes in one community."""
    # Triangle: 0-1, 1-2, 0-2
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    communities = cohera.leiden(edges, n_nodes=3, seed=42)
    assert len(communities) == 3
    assert communities[0] == communities[1] == communities[2]
