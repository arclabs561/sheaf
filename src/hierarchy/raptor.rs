//! RAPTOR-style hierarchical tree for recursive summarization.
//!
//! RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
//! builds a tree by:
//! 1. Clustering items at each level
//! 2. Summarizing each cluster
//! 3. Recursively building higher levels from summaries
//!
//! ## References
//!
//! Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for
//! Tree-Organized Retrieval." ICLR 2024.

use super::node::{Node, NodeMetadata};
use crate::error::{Error, Result};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Configuration for building a RAPTOR tree.
#[derive(Debug, Clone)]
pub struct TreeConfig {
    /// Maximum depth of the tree (number of summary levels).
    pub max_depth: usize,
    /// Target number of items per cluster (fanout).
    pub fanout: usize,
    /// Minimum items needed to form a cluster.
    pub min_cluster_size: usize,
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 4,
            fanout: 6,
            min_cluster_size: 2,
        }
    }
}

impl TreeConfig {
    /// Create a new tree configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum depth.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Set target fanout (items per cluster).
    pub fn with_fanout(mut self, fanout: usize) -> Self {
        self.fanout = fanout;
        self
    }

    /// Set minimum cluster size.
    pub fn with_min_cluster_size(mut self, size: usize) -> Self {
        self.min_cluster_size = size;
        self
    }
}

/// A RAPTOR-style hierarchical tree.
///
/// The tree has multiple levels:
/// - Level 0: Leaf nodes (original items)
/// - Level 1+: Summary nodes (aggregations of children)
///
/// ## Type Parameters
///
/// - `T`: Type of leaf items
/// - `S`: Type of summaries (often same as T, or a summary struct)
#[derive(Debug, Clone)]
pub struct RaptorTree<T, S = T> {
    /// All nodes in the tree, indexed by ID.
    nodes: Vec<Node<T, S>>,
    /// Node IDs at each level (level 0 = leaves).
    levels: Vec<Vec<usize>>,
    /// Configuration used to build the tree.
    /// TODO: Use config.max_depth in build() for early termination.
    #[allow(dead_code)]
    config: TreeConfig,
}

impl<T, S> RaptorTree<T, S> {
    /// Create a new empty tree.
    pub fn new(config: TreeConfig) -> Self {
        Self {
            nodes: Vec::new(),
            levels: Vec::new(),
            config,
        }
    }

    /// Build a tree from items using provided clustering and summarization functions.
    ///
    /// ## Arguments
    ///
    /// - `items`: The leaf items to build the tree from
    /// - `cluster_fn`: Function that groups items into clusters
    /// - `summarize_fn`: Function that summarizes the leaf items under a cluster;
    ///   at level 2 and above it receives the leaves of the member summaries
    ///
    /// ## Example
    ///
    /// ```rust,ignore
    /// let tree = RaptorTree::build(
    ///     chunks,
    ///     TreeConfig::default(),
    ///     |items| cluster_by_embedding(items, 6),
    ///     |group| llm_summarize(group),
    /// )?;
    /// ```
    pub fn build<F, G>(
        items: Vec<T>,
        config: TreeConfig,
        cluster_fn: F,
        summarize_fn: G,
    ) -> Result<Self>
    where
        F: Fn(&[usize], usize) -> Vec<Vec<usize>>,
        G: Fn(&[&T]) -> S,
        T: Clone,
        S: Clone,
    {
        if items.is_empty() {
            return Err(Error::EmptyInput);
        }

        let mut tree = Self::new(config.clone());

        // Level 0: Create leaf nodes
        let leaf_ids: Vec<usize> = items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                let node = Node::leaf(i, item);
                tree.nodes.push(node);
                i
            })
            .collect();

        tree.levels.push(leaf_ids.clone());

        // Build higher levels recursively
        let mut current_ids = leaf_ids;
        let mut next_id = tree.nodes.len();

        for level in 1..=config.max_depth {
            if current_ids.len() <= config.min_cluster_size {
                break;
            }

            // Cluster current level's nodes
            let clusters = cluster_fn(&current_ids, config.fanout);

            if clusters.is_empty() {
                break;
            }

            let mut level_ids = Vec::new();

            for cluster in clusters {
                if cluster.is_empty() {
                    continue;
                }

                // Collect the leaf items under this cluster. From level 2 on
                // the members are summary nodes, and `summarize_fn` takes
                // leaf items, so descend to their leaves.
                let items_to_summarize: Vec<&T> = cluster
                    .iter()
                    .flat_map(|&id| tree.leaf_descendants(id))
                    .filter_map(|id| tree.get_leaf_content(id))
                    .collect();

                if items_to_summarize.is_empty() {
                    continue;
                }

                // Create summary
                let summary = summarize_fn(&items_to_summarize);

                // Create summary node
                let metadata = NodeMetadata {
                    leaf_count: items_to_summarize.len(),
                    cluster_method: Some("provided".into()),
                    summary_method: Some("provided".into()),
                };

                let node = Node::internal(next_id, summary, level, cluster).with_metadata(metadata);

                tree.nodes.push(node);
                level_ids.push(next_id);
                next_id += 1;
            }

            if level_ids.is_empty() {
                break;
            }

            tree.levels.push(level_ids.clone());
            current_ids = level_ids;
        }

        Ok(tree)
    }

    /// IDs of the leaves under node `id` (the node itself if it is a leaf).
    fn leaf_descendants(&self, id: usize) -> Vec<usize> {
        let mut leaves = Vec::new();
        let mut stack = vec![id];
        while let Some(cur) = stack.pop() {
            match self.nodes.get(cur) {
                Some(node) if node.is_leaf() => leaves.push(cur),
                Some(node) => stack.extend(node.children.iter().rev()),
                None => {}
            }
        }
        leaves
    }

    /// Get content of a leaf node by ID.
    fn get_leaf_content(&self, id: usize) -> Option<&T> {
        self.nodes.get(id).and_then(|n| n.as_leaf())
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&Node<T, S>> {
        self.nodes.get(id)
    }

    /// Get all nodes at a specific level.
    pub fn get_level(&self, level: usize) -> Option<Vec<&Node<T, S>>> {
        self.levels
            .get(level)
            .map(|ids| ids.iter().filter_map(|&id| self.nodes.get(id)).collect())
    }

    /// Get the number of levels in the tree.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// Get all leaf nodes.
    pub fn leaves(&self) -> Vec<&Node<T, S>> {
        self.get_level(0).unwrap_or_default()
    }

    /// Get the root nodes (highest level).
    pub fn roots(&self) -> Vec<&Node<T, S>> {
        self.get_level(self.depth().saturating_sub(1))
            .unwrap_or_default()
    }

    /// Total number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Iterate over all nodes.
    pub fn iter(&self) -> impl Iterator<Item = &Node<T, S>> {
        self.nodes.iter()
    }

    /// Get nodes for "collapsed tree" retrieval (all nodes flattened).
    ///
    /// This is the retrieval method from RAPTOR where all nodes at all
    /// levels are considered as retrieval candidates.
    pub fn collapsed(&self) -> Vec<&Node<T, S>> {
        self.nodes.iter().collect()
    }

    /// View the tree at a specific level of abstraction.
    ///
    /// - Level 0: Original items (finest granularity)
    /// - Higher levels: Increasingly coarse summaries
    pub fn view_at_level(&self, level: usize) -> Vec<&Node<T, S>> {
        let target = level.min(self.depth().saturating_sub(1));
        self.get_level(target).unwrap_or_default()
    }
}

impl<T, S> Default for RaptorTree<T, S> {
    fn default() -> Self {
        Self::new(TreeConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_config_default() {
        let config = TreeConfig::default();
        assert_eq!(config.max_depth, 4);
        assert_eq!(config.fanout, 6);
    }

    #[test]
    fn test_empty_tree() {
        let tree: RaptorTree<String> = RaptorTree::default();
        assert!(tree.is_empty());
        assert_eq!(tree.depth(), 0);
    }
}
