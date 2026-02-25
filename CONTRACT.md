## parti — stable contract (draft v0)

`parti` is a **general-purpose hierarchy + clustering primitives** crate.
It is not Anno-specific. Anno-specific "tiering product" logic should live elsewhere.

This contract defines the stable surface: what `parti` promises to callers.

### Scope (in)

- **Clustering primitives** (dense / graph-backed):
  - hard clustering (k-means, hierarchical, DBSCAN, spectral, etc.)
  - soft clustering (GMM) when enabled
- **Community detection** on graphs:
  - Louvain / Leiden / label propagation
  - optional kNN graph construction from embeddings (feature-gated)
- **Hierarchy primitives**:
  - dendrogram/tree representations
  - reconciliation and hierarchical conformal prediction utilities
- **Metrics**:
  - clustering evaluation metrics (NMI/ARI/etc.) as utilities

### Non-scope (out)

- No domain semantics (no "NER/coref" notions, no document types).
- No IO formats as the primary interface (serde is optional).
- No "product CLI" by default (examples are fine; stable CLI lives elsewhere).

### Inputs / outputs (shapes)

- **Clustering**:
  - input: dense points `&[&[f32]]` (or feature-gated ndarray types)
  - output: assignments `Vec<usize>` and/or centroids, plus algorithm-specific diagnostics
- **Community detection**:
  - input: graph (feature-gated types) or kNN graph built from embeddings
  - output: partition / community labels
- **Hierarchy / reconciliation / conformal**:
  - input: tree structure + base predictions
  - output: reconciled predictions and/or prediction intervals with coherence guarantees (when assumptions are met)

### Invariants (must always hold)

- **Determinism**: for fixed inputs + fixed RNG seeds, results are deterministic.
- **Probability-like values** (e.g. conformal confidence/alpha):
  - finite, within expected bounds, and validated by tests.
- **Numeric honesty**:
  - no silent NaN propagation in public APIs; either reject or document behavior.

### Compatibility promises

- Public structs and enums evolve by **adding fields/variants**, not changing meanings.
- Features stay meaning-stable: a feature name shouldn't change what it "means."

### Relation to the stack

- `parti` is a **substrate** library (Tekne L3 "Structures" in your internal map).
- Downstream crates may wrap it into domain-specific products (e.g. an "anno-tiering" crate),
  but `parti` itself should remain domain-agnostic.
