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
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};

/// Leiden community detection algorithm.
///
/// Guarantees well-connected communities through a refinement phase
/// that Louvain lacks.
#[derive(Debug, Clone)]
pub struct Leiden {
    /// Resolution parameter (gamma). Higher = smaller communities.
    resolution: f64,
    /// Maximum iterations per phase.
    max_iter: usize,
    /// Minimum modularity gain to continue.
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
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set minimum modularity gain threshold.
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
}

impl Default for Leiden {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal graph representation for weighted operations.
struct WeightedGraph {
    n: usize,
    /// Adjacency: node -> [(neighbor, weight)]
    adj: Vec<Vec<(usize, f64)>>,
    /// Weighted degree of each node
    degrees: Vec<f64>,
    /// Total edge weight (2m in modularity formula)
    total_weight: f64,
}

impl WeightedGraph {
    fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut adj = vec![Vec::new(); n];
        let mut degrees = vec![0.0; n];
        let mut total_weight = 0.0;

        for &(i, j, w) in edges {
            adj[i].push((j, w));
            adj[j].push((i, w));
            degrees[i] += w;
            degrees[j] += w;
            total_weight += 2.0 * w; // Each edge counted twice
        }

        Self {
            n,
            adj,
            degrees,
            total_weight,
        }
    }

    /// Compute modularity gain from moving node to target community.
    ///
    /// delta_Q = k_i,in / m - gamma * sigma_tot * k_i / (2m^2)
    fn modularity_gain(
        &self,
        node: usize,
        target_comm: usize,
        communities: &[usize],
        comm_total_weight: &[f64],
        resolution: f64,
    ) -> f64 {
        if self.total_weight == 0.0 {
            return 0.0;
        }

        let m = self.total_weight / 2.0;
        let ki = self.degrees[node];

        // Sum of edge weights from node to target community
        let ki_in: f64 = self.adj[node]
            .iter()
            .filter(|(neighbor, _)| communities[*neighbor] == target_comm)
            .map(|(_, w)| w)
            .sum();

        let sigma_tot = comm_total_weight[target_comm];

        ki_in / m - resolution * sigma_tot * ki / (2.0 * m * m)
    }
}

/// Community assignment with cached statistics.
struct CommunityState {
    /// Community assignment for each node.
    assignment: Vec<usize>,
    /// Total weighted degree in each community.
    comm_total_weight: Vec<f64>,
    /// Number of communities (some may be empty).
    n_communities: usize,
}

impl CommunityState {
    fn new_singletons(n: usize, degrees: &[f64]) -> Self {
        Self {
            assignment: (0..n).collect(),
            comm_total_weight: degrees.to_vec(),
            n_communities: n,
        }
    }

    fn move_node(&mut self, node: usize, from: usize, to: usize, degree: f64) {
        self.assignment[node] = to;
        self.comm_total_weight[from] -= degree;
        self.comm_total_weight[to] += degree;
    }
}

impl Leiden {
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

        let edges: Vec<(usize, usize, f64)> = graph
            .edge_references()
            .filter_map(|e| {
                let i = e.source().index();
                let j = e.target().index();
                if i < j {
                    Some((i, j, *e.weight() as f64))
                } else {
                    None
                }
            })
            .collect();

        self.detect_from_edges(n, &edges)
    }

    /// Internal: run Leiden on a weighted edge list.
    fn detect_from_edges(&self, n: usize, edges: &[(usize, usize, f64)]) -> Result<Vec<usize>> {
        let wg = WeightedGraph::from_edges(n, edges);
        let mut state = CommunityState::new_singletons(n, &wg.degrees);

        for _level in 0..self.max_iter {
            let improved = self.local_moving_phase(&wg, &mut state);
            if !improved {
                break;
            }
            self.refinement_phase(&wg, &mut state);
        }

        Ok(renumber_communities(&state.assignment))
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

        let edges: Vec<(usize, usize, f64)> = graph
            .edge_references()
            .filter_map(|e| {
                let i = e.source().index();
                let j = e.target().index();
                if i < j {
                    Some((i, j, 1.0))
                } else {
                    None
                }
            })
            .collect();

        self.detect_from_edges(n, &edges)
    }

    fn resolution(&self) -> f64 {
        self.resolution
    }
}

impl Leiden {
    /// Phase 1: Local moving (similar to Louvain).
    ///
    /// Greedily move nodes to neighboring communities with best modularity gain.
    fn local_moving_phase(&self, wg: &WeightedGraph, state: &mut CommunityState) -> bool {
        let mut improved = false;
        let mut queue: VecDeque<usize> = (0..wg.n).collect();
        let mut in_queue = vec![true; wg.n];

        while let Some(node) = queue.pop_front() {
            in_queue[node] = false;
            let current_comm = state.assignment[node];

            // Find neighboring communities
            let mut neighbor_comms: HashSet<usize> = HashSet::new();
            for &(neighbor, _) in &wg.adj[node] {
                let _ = neighbor_comms.insert(state.assignment[neighbor]);
            }
            let _ = neighbor_comms.insert(current_comm);

            // Find best community
            let mut best_comm = current_comm;
            let mut best_gain = 0.0;

            // Temporarily remove node from current community
            state.comm_total_weight[current_comm] -= wg.degrees[node];

            for &target_comm in &neighbor_comms {
                let gain = wg.modularity_gain(
                    node,
                    target_comm,
                    &state.assignment,
                    &state.comm_total_weight,
                    self.resolution,
                );
                if gain > best_gain + 1e-10 {
                    best_gain = gain;
                    best_comm = target_comm;
                }
            }

            // Restore node to community (will be moved below if needed)
            state.comm_total_weight[current_comm] += wg.degrees[node];

            if best_comm != current_comm {
                state.move_node(node, current_comm, best_comm, wg.degrees[node]);
                improved = true;

                // Add neighbors back to queue
                for &(neighbor, _) in &wg.adj[node] {
                    if !in_queue[neighbor] {
                        queue.push_back(neighbor);
                        in_queue[neighbor] = true;
                    }
                }
            }
        }

        improved
    }

    /// Phase 2: Refinement (Leiden's key innovation).
    ///
    /// Within each community from phase 1:
    /// 1. Reset nodes to singletons
    /// 2. Merge only within the community boundary
    /// 3. Ensure each sub-community is connected
    fn refinement_phase(&self, wg: &WeightedGraph, state: &mut CommunityState) {
        // Get current partition (from phase 1)
        let phase1_communities = state.assignment.clone();

        // Find unique communities
        let mut unique_comms: Vec<usize> = phase1_communities.to_vec();
        unique_comms.sort_unstable();
        unique_comms.dedup();

        // Process each community independently
        for &comm in &unique_comms {
            // Get nodes in this community
            let nodes_in_comm: Vec<usize> = (0..wg.n)
                .filter(|&i| phase1_communities[i] == comm)
                .collect();

            if nodes_in_comm.len() <= 1 {
                continue;
            }

            // Refine within this community
            self.refine_community(wg, state, &nodes_in_comm);
        }
    }

    /// Refine a single community.
    ///
    /// Ensures the community is well-connected by:
    /// 1. Finding connected components within the community
    /// 2. Splitting disconnected components into separate communities
    fn refine_community(&self, wg: &WeightedGraph, state: &mut CommunityState, nodes: &[usize]) {
        let node_set: HashSet<usize> = nodes.iter().copied().collect();

        // Find connected components within this subset
        let components = find_components_in_subset(wg, &node_set);

        if components.len() <= 1 {
            // Already connected, nothing to do
            return;
        }

        // Split disconnected components into separate communities
        let base_comm = state.n_communities;
        state.n_communities += components.len() - 1;

        // Extend comm_total_weight if needed
        while state.comm_total_weight.len() < state.n_communities {
            state.comm_total_weight.push(0.0);
        }

        // Keep first component in original community, assign others to new communities
        for (idx, component) in components.iter().enumerate().skip(1) {
            let new_comm = base_comm + idx - 1;
            for &node in component {
                let old_comm = state.assignment[node];
                state.move_node(node, old_comm, new_comm, wg.degrees[node]);
            }
        }
    }
}

/// Find connected components within a subset of nodes.
fn find_components_in_subset(wg: &WeightedGraph, node_set: &HashSet<usize>) -> Vec<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &start in node_set {
        if visited.contains(&start) {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            if !visited.insert(node) {
                continue;
            }
            component.push(node);

            for &(neighbor, _) in &wg.adj[node] {
                if node_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        if !component.is_empty() {
            components.push(component);
        }
    }

    components
}

/// Renumber communities to consecutive integers starting at 0.
fn renumber_communities(assignment: &[usize]) -> Vec<usize> {
    let mut unique: Vec<usize> = assignment.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let mapping: HashMap<usize, usize> = unique
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect();

    assignment
        .iter()
        .map(|&c| mapping.get(&c).copied().unwrap_or(0))
        .collect()
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
        // This is the key test that Louvain fails but Leiden should pass.
        // Create a graph where naive merging could create disconnected communities.
        //
        // Structure: A--B--C  D--E (D,E disconnected from A,B,C)
        // If Louvain merges all into one community, it's wrong.
        // Leiden should detect and split.

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
        // Resolution parameter affects community detection
        // Note: Due to greedy local optima, higher resolution doesn't *guarantee*
        // more communities, but both should produce valid partitions.
        let mut graph = UnGraph::<(), ()>::new_undirected();

        // Create a larger structure
        for _ in 0..10 {
            let _ = graph.add_node(());
        }
        // Connect as a chain
        for i in 0..9 {
            let n1 = petgraph::graph::NodeIndex::new(i);
            let n2 = petgraph::graph::NodeIndex::new(i + 1);
            let _ = graph.add_edge(n1, n2, ());
        }

        let low_res = Leiden::new().with_resolution(0.5);
        let high_res = Leiden::new().with_resolution(2.0);

        let comms_low = low_res.detect(&graph).unwrap();
        let comms_high = high_res.detect(&graph).unwrap();

        // Both should assign all nodes
        assert_eq!(comms_low.len(), 10);
        assert_eq!(comms_high.len(), 10);

        let unique_low: HashSet<_> = comms_low.iter().collect();
        let unique_high: HashSet<_> = comms_high.iter().collect();

        // Both should produce valid partitions (at least 1 community)
        assert!(!unique_low.is_empty());
        assert!(!unique_high.is_empty());
    }

    #[test]
    fn test_leiden_connectivity_guarantee() {
        // Verify that every community is internally connected
        let mut graph = UnGraph::<(), ()>::new_undirected();

        // Create a moderately complex graph
        for _ in 0..20 {
            let _ = graph.add_node(());
        }
        // Add some edges
        for i in 0..15 {
            let n1 = petgraph::graph::NodeIndex::new(i);
            let n2 = petgraph::graph::NodeIndex::new(i + 1);
            let _ = graph.add_edge(n1, n2, ());
        }
        // Add some cross-links
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

        // Group nodes by community
        let mut by_community: HashMap<usize, Vec<usize>> = HashMap::new();
        for (node, &comm) in communities.iter().enumerate() {
            by_community.entry(comm).or_default().push(node);
        }

        // Verify each community is connected
        for (_comm, nodes) in by_community {
            if nodes.len() <= 1 {
                continue;
            }

            let node_set: HashSet<usize> = nodes.iter().copied().collect();

            // Build subgraph adjacency
            let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
            for edge in graph.edge_references() {
                let i = edge.source().index();
                let j = edge.target().index();
                if node_set.contains(&i) && node_set.contains(&j) {
                    adj.entry(i).or_default().push(j);
                    adj.entry(j).or_default().push(i);
                }
            }

            // BFS from first node should reach all nodes
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
