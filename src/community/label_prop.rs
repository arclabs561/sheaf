//! Label propagation for community detection.
//!
//! Very fast O(E) algorithm where nodes adopt the most common
//! label among their neighbors.

use super::traits::CommunityDetection;
use crate::error::{Error, Result};
use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;
use rand::prelude::*;

/// Label propagation community detection.
#[derive(Debug, Clone)]
pub struct LabelPropagation {
    /// Maximum iterations.
    max_iter: usize,
    /// Random seed.
    seed: Option<u64>,
}

impl LabelPropagation {
    /// Create a new label propagation detector.
    pub fn new() -> Self {
        Self {
            max_iter: 100,
            seed: None,
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for LabelPropagation {
    fn default() -> Self {
        Self::new()
    }
}

impl CommunityDetection for LabelPropagation {
    fn detect<N, E>(&self, graph: &UnGraph<N, E>) -> Result<Vec<usize>> {
        let n = graph.node_count();
        if n == 0 {
            return Err(Error::EmptyInput);
        }

        // Initialize: each node has its own label
        let mut labels: Vec<usize> = (0..n).collect();

        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(rand::rng()),
        };

        // Iterate until convergence
        for _iter in 0..self.max_iter {
            let mut changed = false;

            // Random order
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(&mut rng);

            for &node in &order {
                let node_idx = petgraph::graph::NodeIndex::new(node);

                // Count neighbor labels. A BTreeMap keeps the tie candidates in
                // label order, so a fixed seed draws the same label every run.
                let mut label_counts = std::collections::BTreeMap::new();
                for edge in graph.edges(node_idx) {
                    let neighbor = edge.target().index();
                    *label_counts.entry(labels[neighbor]).or_insert(0) += 1;
                }

                if label_counts.is_empty() {
                    continue;
                }

                // Find most common label (ties broken randomly)
                // SAFETY: label_counts is non-empty (checked above)
                let max_count = label_counts.values().max().copied().unwrap_or(0);
                let candidates: Vec<usize> = label_counts
                    .iter()
                    .filter(|(_, &count)| count == max_count)
                    .map(|(&label, _)| label)
                    .collect();

                let new_label = if candidates.len() == 1 {
                    candidates[0]
                } else {
                    candidates[rng.random_range(0..candidates.len())]
                };

                if labels[node] != new_label {
                    labels[node] = new_label;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Renumber to consecutive integers
        let mut unique: Vec<usize> = labels.to_vec();
        unique.sort_unstable();
        unique.dedup();

        Ok(labels
            .iter()
            .map(|&l| unique.iter().position(|&u| u == l).unwrap_or(0))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_propagation_basic() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        // Two disconnected edges
        let _ = graph.add_edge(n0, n1, ());
        let _ = graph.add_edge(n2, n3, ());

        let lp = LabelPropagation::new().with_seed(42);
        let communities = lp.detect(&graph).unwrap();

        // Should find 2 communities
        assert_eq!(communities[0], communities[1]);
        assert_eq!(communities[2], communities[3]);
        assert_ne!(communities[0], communities[2]);
    }
}
