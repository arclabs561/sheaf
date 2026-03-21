def reconcile(
    summing_matrix: list[list[float]],
    base_forecasts: list[float],
    method: str = "ols",
    weights: list[float] | None = None,
    covariance: list[list[float]] | None = None,
) -> list[float]: ...

def simple_star_matrix(n_leaves: int) -> list[list[float]]: ...

class HierarchicalConformal:
    def __init__(
        self,
        summing_matrix: list[list[float]],
        method: str = "ols",
    ) -> None: ...
    def calibrate(
        self,
        y_calib: list[list[float]],
        y_hat_calib: list[list[float]],
        alpha: float,
    ) -> None: ...
    def predict(
        self, y_hat: list[float]
    ) -> tuple[list[float], list[float]]: ...

def leiden(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    resolution: float = 1.0,
    seed: int | None = None,
) -> list[int]: ...

def louvain(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
    resolution: float = 1.0,
) -> list[int]: ...

def label_propagation(
    edges: list[tuple[int, int, float]],
    n_nodes: int,
) -> list[int]: ...

def nmi(labels_a: list[int], labels_b: list[int]) -> float: ...
def ari(labels_a: list[int], labels_b: list[int]) -> float: ...
def v_measure(labels_a: list[int], labels_b: list[int]) -> float: ...
def purity(labels_a: list[int], labels_b: list[int]) -> float: ...
def homogeneity(labels_a: list[int], labels_b: list[int]) -> float: ...
def completeness(labels_a: list[int], labels_b: list[int]) -> float: ...
def fowlkes_mallows(labels_a: list[int], labels_b: list[int]) -> float: ...
