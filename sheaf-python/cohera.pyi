from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

def reconcile(
    summing_matrix: npt.NDArray[np.float64] | list[list[float]],
    base_forecasts: npt.NDArray[np.float64] | list[float],
    method: Literal["ols", "wls", "mint"] = "ols",
    weights: list[float] | None = None,
    covariance: npt.NDArray[np.float64] | list[list[float]] | None = None,
) -> npt.NDArray[np.float64]:
    """Reconcile base forecasts to satisfy hierarchical constraints.

    Args:
        summing_matrix: 2D array (m x n) defining the hierarchy.
        base_forecasts: 1D array (m,) of incoherent base forecasts.
        method: "ols", "wls", or "mint".
        weights: Diagonal weights for "wls" (m,).
        covariance: Covariance matrix for "mint" (m x m).

    Returns:
        Reconciled forecasts as a 1D numpy array (m,).
    """
    ...

def simple_star_matrix(n_leaves: int) -> npt.NDArray[np.float64]:
    """Build a summing matrix for a 2-level hierarchy (1 root + n leaves).

    Args:
        n_leaves: Number of leaf nodes.

    Returns:
        Summing matrix with shape (n_leaves + 1, n_leaves).
    """
    ...

def summing_matrix_from_tree(
    merges: list[tuple[int, int, float, int]],
    n_leaves: int,
) -> npt.NDArray[np.float64]:
    """Build a summing matrix from a dendrogram-style merge history.

    Each merge (cluster_a, cluster_b, height, size) creates an internal
    node. The resulting matrix has shape (n_leaves + len(merges), n_leaves).

    Args:
        merges: List of (cluster_a, cluster_b, height, size) merge operations.
        n_leaves: Number of leaf (bottom-level) nodes.

    Returns:
        Summing matrix as a 2D numpy array.
    """
    ...

class Dendrogram:
    """Dendrogram from hierarchical clustering merge history.

    Supports cutting at a distance threshold or to a target number of clusters.
    """

    def __init__(
        self,
        merges: list[tuple[int, int, float, int]],
        n_items: int,
    ) -> None: ...

    def cut_at_distance(self, threshold: float) -> npt.NDArray[np.int64]:
        """Cut at a distance threshold.

        Args:
            threshold: Distance threshold for cutting.

        Returns:
            Cluster label for each item, dtype int64.
        """
        ...

    def cut_to_k(self, k: int) -> npt.NDArray[np.int64]:
        """Cut to produce exactly k clusters.

        Args:
            k: Desired number of clusters.

        Returns:
            Cluster label for each item, dtype int64.
        """
        ...

    def distances(self) -> npt.NDArray[np.float64]:
        """Merge distances in order.

        Returns:
            Distance at each merge step, dtype float64.
        """
        ...

    @property
    def n_items(self) -> int:
        """Number of original (leaf) items."""
        ...

    @property
    def n_merges(self) -> int:
        """Number of merge operations recorded."""
        ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class HierarchicalConformal:
    """Hierarchical conformal predictor.

    Wraps reconciliation with split-conformal calibration to produce
    prediction intervals that respect the hierarchy.
    """

    def __init__(
        self,
        summing_matrix: npt.NDArray[np.float64] | list[list[float]],
        method: Literal["ols", "wls", "mint"] = "ols",
    ) -> None: ...
    def calibrate(
        self,
        y_calib: npt.NDArray[np.float64] | list[list[float]],
        y_hat_calib: npt.NDArray[np.float64] | list[list[float]],
        alpha: float,
    ) -> None:
        """Calibrate using held-out data.

        Args:
            y_calib: True values, shape (m, n_calib).
            y_hat_calib: Base forecasts, shape (m, n_calib).
            alpha: Miscoverage level (e.g. 0.1 for 90% coverage).
        """
        ...
    def predict(
        self,
        y_hat: npt.NDArray[np.float64] | list[float],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Predict reconciled intervals for new base forecasts.

        Args:
            y_hat: Base forecast vector (m,).

        Returns:
            (lower, upper) bounds, each a 1D numpy array (m,).
        """
        ...
    def __repr__(self) -> str: ...

def leiden(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    resolution: float = 1.0,
    seed: int | None = None,
) -> npt.NDArray[np.int64]:
    """Leiden community detection on a weighted undirected graph.

    Args:
        edges: Edge list as [(source, target, weight), ...].
        n_nodes: Total number of nodes.
        resolution: Modularity resolution (default 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Community assignment for each node, dtype int64.
    """
    ...

def louvain(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    resolution: float = 1.0,
) -> npt.NDArray[np.int64]:
    """Louvain community detection on a weighted undirected graph.

    Args:
        edges: Edge list as [(source, target, weight), ...].
        n_nodes: Total number of nodes.
        resolution: Modularity resolution (default 1.0).

    Returns:
        Community assignment for each node, dtype int64.
    """
    ...

def label_propagation(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
) -> npt.NDArray[np.int64]:
    """Label propagation community detection.

    Args:
        edges: Edge list as [(source, target, weight), ...].
        n_nodes: Total number of nodes.

    Returns:
        Community assignment for each node, dtype int64.
    """
    ...

def knn_graph(
    embeddings: npt.NDArray[np.float32] | npt.NDArray[np.float64] | list[list[float]],
    k: int = 10,
) -> list[tuple[int, int, float]]:
    """Build a k-nearest-neighbor graph from embeddings.

    Uses HNSW for approximate nearest neighbor search. The resulting edge
    list can be passed directly to leiden/louvain/label_propagation.

    Args:
        embeddings: 2D array of shape (n, dim). Each row is an embedding.
        k: Number of neighbors per node (default 10).

    Returns:
        Edge list as [(source, target, weight), ...]. Weights are similarity
        scores (higher = more similar).
    """
    ...

def nmi(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """Normalized Mutual Information between two clusterings.

    Returns:
        NMI in [0, 1]. 1.0 means identical clusterings.
    """
    ...

def ari(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """Adjusted Rand Index between two clusterings.

    Returns:
        ARI in [-1, 1]. 1.0 means identical, 0.0 means random.
    """
    ...

def v_measure(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """V-Measure: harmonic mean of homogeneity and completeness.

    Returns:
        V-measure in [0, 1].
    """
    ...

def purity(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """Purity of clustering with respect to ground truth.

    Returns:
        Purity in [0, 1].
    """
    ...

def homogeneity(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """Homogeneity: each cluster contains only members of a single class.

    Returns:
        Homogeneity in [0, 1].
    """
    ...

def completeness(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """Completeness: all members of a class are in the same cluster.

    Returns:
        Completeness in [0, 1].
    """
    ...

def fowlkes_mallows(
    labels_true: npt.NDArray[np.integer] | Sequence[int],
    labels_pred: npt.NDArray[np.integer] | Sequence[int],
) -> float:
    """Fowlkes-Mallows Index: geometric mean of pairwise precision and recall.

    Returns:
        FMI in [0, 1]. 1.0 means identical clusterings.
    """
    ...
