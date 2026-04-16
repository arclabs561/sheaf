//! Leiden algorithm for community detection.
//!
//! An improvement over Louvain that guarantees well-connected communities.
//!
//! ## The Leiden Algorithm (Traag et al. 2019)
//!
//! Leiden fixes Louvain's fundamental flaw: Louvain can create disconnected
//! communities because it never re-examines decisions within a community.
//!
//! ### Three Phases
//!
//! 1. **Local Moving**: Like Louvain, greedily move nodes to best community.
//!
//! 2. **Refinement**: The key innovation. Within each community from phase 1:
//!    - Reset all nodes to singletons
//!    - Merge only within the community's boundary
//!    - Check that each merge maintains connectivity
//!
//! 3. **Aggregation**: Build meta-graph and recurse.
//!
//! ### Why Refinement Matters
//!
//! ```text
//! Louvain can produce:        Leiden guarantees:
//!     A---B                       A---B
//!         |                           |
//!     C   D                       C   D
//!                                 (C in separate community)
//! [A,B,C,D] all in one         [A,B,D] connected, [C] alone
//! community despite C
//! being disconnected!
//! ```
//!
//! ## Complexity
//!
//! - Time: O(m) per iteration (m = edges), typically O(m log n) total
//! - Space: O(n + m)
//!
//! ## References
//!
//! Traag, Waltman, van Eck (2019). "From Louvain to Leiden: guaranteeing
//! well-connected communities." Scientific Reports 9, 5233.

use super::traits::CommunityDetection;
use crate::error::{Error, Result};
use graphops::{leiden_seeded, leiden_weighted_seeded, Graph, PetgraphRef, WeightedGraph};
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;

/// Leiden community detection algorithm.
///
/// Guarantees well-connected communities through a refinement phase
/// that Louvain lacks.
#[derive(Debug, Clone)]
pub struct Leiden {
    /// Resolution parameter (gamma). Higher = smaller communities.
    resolution: f64,
    /// Maximum iterations per phase (stored but not forwarded; graphops runs to convergence).
    #[allow(dead_code)]
    max_iter: usize,
    /// Minimum modularity gain to continue (stored but not forwarded; graphops runs to convergence).
    #[allow(dead_code)]
    min_gain: f64,
    /// Random seed for tie-breaking.
    seed: u64,
}

impl Leiden {
    /// Create a new Leiden detector.
    pub fn new() -> Self {
        Self {
            resolution: 1.0,
            max_iter: 100,
            min_gain: 1e-7,
            seed: 42,
        }
    }

    /// Set resolution parameter.
    ///
    /// Higher values produce smaller communities.
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Set maximum iterations.
    ///
    /// Note: the graphops backend runs until convergence; this value is retained
    /// for API compatibility but is not forwarded to the solver.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set minimum modularity gain threshold.
    ///
    /// Note: the graphops backend runs until convergence; this value is retained
    /// for API compatibility but is not forwarded to the solver.
    pub fn with_min_gain(mut self, min_gain: f64) -> Self {
        self.min_gain = min_gain;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Deprecated: use `with_max_iter` instead.
    #[deprecated(since = "0.2.0", note = "Use with_max_iter instead")]
    pub fn with_refinement(self, n: usize) -> Self {
        self.with_max_iter(n)
    }

    /// Detect communities in a weighted graph with `f32` edge weights.
    ///
    /// Edge weights are used directly in the modularity computation. Higher
    /// weights indicate stronger connections. This is the recommended method
    /// when you have meaningful edge weights (e.g., similarity scores).
    pub fn detect_weighted<N>(&self, graph: &UnGraph<N, f32>) -> Result<Vec<usize>> {
        let n = graph.node_count();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if graph.edge_count() == 0 {
            return Ok((0..n).collect());
        }

        let adapter = F32WeightedAdapter::from_graph(graph);
        Ok(leiden_weighted_seeded(&adapter, self.resolution, self.seed))
    }
}

impl Default for Leiden {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunityDetection for Leiden {
    /// Detect communities in an unweighted graph (all edges have weight 1.0).
    fn detect<N, E>(&self, graph: &UnGraph<N, E>) -> Result<Vec<usize>> {
        let n = graph.node_count();
        if n == 0 {
            return Err(Error::EmptyInput);
        }
        if graph.edge_count() == 0 {
            return Ok((0..n).collect());
        }

        let adapter = PetgraphRef::from_graph(graph);
        Ok(leiden_seeded(&adapter, self.resolution, self.seed))
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

    #[test]
    fn test_leiden_basic() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        let _ = graph.add_edge(n0, n1, ());
        let _ = graph.add_edge(n1, n2, ());
        let _ = graph.add_edge(n0, n2, ());

        let leiden = Leiden::new();
        let communities = leiden.detect(&graph).unwrap();

        // All in one community (triangle)
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);
    }

    #[test]
    fn test_leiden_two_cliques() {
        // Two triangles connected by a single edge
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

        let leiden = Leiden::new();
        let communities = leiden.detect(&graph).unwrap();

        assert_eq!(communities.len(), 6);

        // First clique should be in same community
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);

        // Second clique should be in same community
        assert_eq!(communities[3], communities[4]);
        assert_eq!(communities[4], communities[5]);

        // Two cliques should be in different communities
        assert_ne!(communities[0], communities[3]);
    }

    #[test]
    fn test_leiden_disconnected_within_community() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());
        let e = graph.add_node(());

        let _ = graph.add_edge(a, b, ());
        let _ = graph.add_edge(b, c, ());
        let _ = graph.add_edge(d, e, ());

        let leiden = Leiden::new();
        let communities = leiden.detect(&graph).unwrap();

        // A, B, C should be in one community (connected)
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[1], communities[2]);

        // D, E should be in another community (connected)
        assert_eq!(communities[3], communities[4]);

        // The two groups should be in different communities
        assert_ne!(communities[0], communities[3]);
    }

    #[test]
    fn test_leiden_empty_graph() {
        let graph = UnGraph::<(), ()>::new_undirected();
        let leiden = Leiden::new();
        let result = leiden.detect(&graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_leiden_single_node() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let _ = graph.add_node(());

        let leiden = Leiden::new();
        let communities = leiden.detect(&graph).unwrap();

        assert_eq!(communities.len(), 1);
        assert_eq!(communities[0], 0);
    }

    #[test]
    fn test_leiden_resolution_parameter() {
        let mut graph = UnGraph::<(), ()>::new_undirected();

        for _ in 0..10 {
            let _ = graph.add_node(());
        }
        for i in 0..9 {
            let n1 = petgraph::graph::NodeIndex::new(i);
            let n2 = petgraph::graph::NodeIndex::new(i + 1);
            let _ = graph.add_edge(n1, n2, ());
        }

        let low_res = Leiden::new().with_resolution(0.5);
        let high_res = Leiden::new().with_resolution(2.0);

        let comms_low = low_res.detect(&graph).unwrap();
        let comms_high = high_res.detect(&graph).unwrap();

        assert_eq!(comms_low.len(), 10);
        assert_eq!(comms_high.len(), 10);

        assert!(!comms_low.is_empty());
        assert!(!comms_high.is_empty());
    }

    #[test]
    fn test_leiden_connectivity_guarantee() {
        use std::collections::{HashMap, HashSet, VecDeque};

        let mut graph = UnGraph::<(), ()>::new_undirected();

        for _ in 0..20 {
            let _ = graph.add_node(());
        }
        for i in 0..15 {
            let n1 = petgraph::graph::NodeIndex::new(i);
            let n2 = petgraph::graph::NodeIndex::new(i + 1);
            let _ = graph.add_edge(n1, n2, ());
        }
        let _ = graph.add_edge(
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(5),
            (),
        );
        let _ = graph.add_edge(
            petgraph::graph::NodeIndex::new(10),
            petgraph::graph::NodeIndex::new(15),
            (),
        );

        let leiden = Leiden::new();
        let communities = leiden.detect(&graph).unwrap();

        let mut by_community: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node, &comm) in communities.iter().enumerate() {
            by_community.entry(comm).or_default().push(node);
        }

        for (_comm, nodes) in by_community {
            if nodes.len() <= 1 {
                continue;
            }

            let node_set: HashSet<usize> = nodes.iter().copied().collect();

            let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
            for edge in graph.edge_references() {
                let i = edge.source().index();
                let j = edge.target().index();
                if node_set.contains(&i) && node_set.contains(&j) {
                    adj.entry(i).or_default().push(j);
                    adj.entry(j).or_default().push(i);
                }
            }

            let start = nodes[0];
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                if !visited.insert(node) {
                    continue;
                }
                if let Some(neighbors) = adj.get(&node) {
                    for &n in neighbors {
                        if !visited.contains(&n) {
                            queue.push_back(n);
                        }
                    }
                }
            }

            assert_eq!(
                visited.len(),
                nodes.len(),
                "Community is not fully connected!"
            );
        }
    }
}
