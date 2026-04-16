//! Louvain algorithm for community detection.
//!
//! Fast modularity optimization through local node moves and graph aggregation.
//!
//! ## The Algorithm (Blondel et al. 2008)
//!
//! Louvain is a multi-level, greedy modularity optimization algorithm:
//!
//! 1. **Phase 1 (Local Moving)**: Start with each node in its own community.
//!    Repeatedly move nodes to neighboring community with highest modularity
//!    gain until no improvement.
//!
//! 2. **Phase 2 (Aggregation)**: Build a meta-graph where communities become
//!    single nodes. Edge weights are sums of edges between communities.
//!    Self-loops represent internal community edges.
//!
//! 3. **Iterate**: Repeat phases 1-2 on the meta-graph until modularity
//!    stops improving.
//!
//! ## Multi-Level Benefits
//!
//! - Finds hierarchical community structure at different resolutions
//! - Often achieves higher modularity than single-level
//! - Faster convergence due to coarsening
//!
//! ## References
//!
//! Blondel et al. (2008). "Fast unfolding of communities in large networks."
//! Journal of Statistical Mechanics: Theory and Experiment, P10008.

use super::traits::CommunityDetection;
use crate::error::{Error, Result};
use graphops::{louvain_seeded, louvain_weighted_seeded, Graph, PetgraphRef, WeightedGraph};
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;

/// Louvain community detection algorithm.
#[derive(Debug, Clone)]
pub struct Louvain {
    /// Resolution parameter (gamma).
    resolution: f64,
    /// Maximum iterations per level (stored for API compatibility; graphops runs to convergence).
    #[allow(dead_code)]
    max_iter: usize,
    /// Maximum levels of aggregation (stored for API compatibility; graphops runs to convergence).
    #[allow(dead_code)]
    max_levels: usize,
    /// Minimum modularity improvement to continue (stored for API compatibility).
    #[allow(dead_code)]
    min_modularity_gain: f64,
}

impl Louvain {
    /// Create a new Louvain detector with default settings.
    pub fn new() -> Self {
        Self {
            resolution: 1.0,
            max_iter: 100,
            max_levels: 10,
            min_modularity_gain: 1e-7,
        }
    }

    /// Set resolution parameter.
    ///
    /// Higher values produce smaller communities.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set maximum iterations per level.
    ///
    /// Note: the graphops backend runs until convergence; this value is retained
    /// for API compatibility but is not forwarded to the solver.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set maximum aggregation levels.
    ///
    /// Note: the graphops backend runs until convergence; this value is retained
    /// for API compatibility but is not forwarded to the solver.
    pub fn with_max_levels(mut self, levels: usize) -> Self {
        self.max_levels = levels;
        self
    }

    /// Detect communities in a weighted graph with `f32` edge weights.
    ///
    /// Edge weights are used directly in the modularity computation.
    pub fn detect_weighted<N>(&self, graph: &UnGraph<N, f32>) -> Result<Vec<usize>> {
        let n = graph.node_count();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if graph.edge_count() == 0 {
            return Ok((0..n).collect());
        }

        let adapter = F32WeightedAdapter::from_graph(graph);
        Ok(louvain_weighted_seeded(&adapter, self.resolution, 0))
    }
}

impl Default for Louvain {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunityDetection for Louvain {
    fn detect<N, E>(&self, graph: &UnGraph<N, E>) -> Result<Vec<usize>> {
        let n = graph.node_count();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        if graph.edge_count() == 0 {
            return Ok((0..n).collect());
        }

        let adapter = PetgraphRef::from_graph(graph);
        Ok(louvain_seeded(&adapter, self.resolution, 0))
    }

    fn resolution(&self) -> f64 {
        self.resolution
    }
}

/// Adapter that exposes a `petgraph::UnGraph<N, f32>` as a `WeightedGraph`.
///
/// graphops's `WeightedGraph` impl for petgraph covers `f64` edge weights only;
/// this adapter bridges the `f32` → `f64` conversion at the boundary.
struct F32WeightedAdapter {
    adj: Vec<Vec<(usize, f64)>>,
}

impl F32WeightedAdapter {
    fn from_graph<N>(graph: &UnGraph<N, f32>) -> Self {
        let n = graph.node_count();
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for edge in graph.edge_references() {
            let i = edge.source().index();
            let j = edge.target().index();
            let w = *edge.weight() as f64;
            adj[i].push((j, w));
            adj[j].push((i, w));
        }
        Self { adj }
    }
}

impl Graph for F32WeightedAdapter {
    fn node_count(&self) -> usize {
        self.adj.len()
    }

    fn neighbors(&self, node: usize) -> Vec<usize> {
        self.adj[node].iter().map(|(v, _)| *v).collect()
    }
}

impl WeightedGraph for F32WeightedAdapter {
    fn edge_weight(&self, source: usize, target: usize) -> f64 {
        self.adj[source]
            .iter()
            .find(|(v, _)| *v == target)
            .map(|(_, w)| *w)
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::UnGraph;

    #[test]
    fn test_louvain_triangle() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        let _ = graph.add_edge(n0, n1, ());
        let _ = graph.add_edge(n1, n2, ());
        let _ = graph.add_edge(n0, n2, ());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 3);
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);
    }

    #[test]
    fn test_louvain_two_cliques() {
        let mut graph = UnGraph::<(), ()>::new_undirected();

        // First clique
        let a0 = graph.add_node(());
        let a1 = graph.add_node(());
        let a2 = graph.add_node(());
        let _ = graph.add_edge(a0, a1, ());
        let _ = graph.add_edge(a1, a2, ());
        let _ = graph.add_edge(a0, a2, ());

        // Second clique
        let b0 = graph.add_node(());
        let b1 = graph.add_node(());
        let b2 = graph.add_node(());
        let _ = graph.add_edge(b0, b1, ());
        let _ = graph.add_edge(b1, b2, ());
        let _ = graph.add_edge(b0, b2, ());

        // Bridge
        let _ = graph.add_edge(a2, b0, ());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 6);

        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);

        assert_eq!(communities[3], communities[4]);
        assert_eq!(communities[4], communities[5]);

        assert_ne!(communities[0], communities[3]);
    }

    #[test]
    fn test_louvain_empty_graph() {
        let graph = UnGraph::<(), ()>::new_undirected();
        let louvain = Louvain::new();
        let result = louvain.detect(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_louvain_single_node() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let _ = graph.add_node(());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0], 0);
    }

    #[test]
    fn test_louvain_disconnected() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let _ = graph.add_node(());
        let _ = graph.add_node(());

        let louvain = Louvain::new();
        let communities = louvain.detect(&graph).unwrap();

        assert_eq!(communities.len(), 2);
        assert_ne!(communities[0], communities[1]);
    }
}
