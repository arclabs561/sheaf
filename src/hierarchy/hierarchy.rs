//! General hierarchical retrieval patterns.
//!
//! # The Multi-Scale Problem
//!
//! Flat retrieval returns chunks at a single granularity. But information
//! exists at multiple scales:
//!
//! ```text
//! Scale       │ Example                    │ Use Case
//! ────────────┼────────────────────────────┼──────────────────────────
//! Document    │ "Paper discusses X"        │ Classification
//! Section     │ "Methods section"          │ Topic navigation
//! Paragraph   │ "Experiment 3 results"     │ Detailed facts
//! Sentence    │ "Accuracy was 94.2%"       │ Specific claims
//! ```
//!
//! Different questions need different scales. Hierarchical retrieval lets
//! you navigate and retrieve at any granularity.
//!
//! # Research Landscape
//!
//! Several papers address multi-scale retrieval with different approaches:
//!
//! | Paper | Key Idea | Structure |
//! |-------|----------|-----------|
//! | **RAPTOR** (2024) | Recursive summarization | K-ary tree with LLM summaries |
//! | **Recursive Chunking** | Hierarchical splitting | Binary tree from document structure |
//! | **Multi-Vector** | Multiple embeddings | Flat but multi-scale representations |
//! | **Matryoshka** | Nested dimensions | Truncatable embeddings |
//! | **Coarse-to-Fine** | Two-phase search | Separate indices at each scale |
//!
//! This module provides abstractions that work across these patterns.
//!
//! # Core Abstraction: Resolution
//!
//! We model multi-scale retrieval as operating at different **resolutions**.
//! Higher resolution = finer detail, lower resolution = more aggregated.
//!
//! ```text
//! Resolution  │ Content
//! ────────────┼──────────────────────────
//! 0.0 (low)   │ Document-level summary
//! 0.5 (mid)   │ Section summaries
//! 1.0 (high)  │ Original chunks
//! ```
//!
//! Retrieval can:
//! - Target a specific resolution
//! - Search across all resolutions (collapsed)
//! - Traverse from coarse to fine (tree search)
//! - Use coarse for filtering, fine for ranking

/// A resolution level in a hierarchical index.
///
/// Resolution 0.0 = coarsest (most aggregated)
/// Resolution 1.0 = finest (original items)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolution(pub f32);

impl Resolution {
    /// Coarsest resolution (document-level).
    pub const COARSE: Self = Self(0.0);
    /// Medium resolution (section-level).
    pub const MEDIUM: Self = Self(0.5);
    /// Finest resolution (chunk-level).
    pub const FINE: Self = Self(1.0);

    /// Create a resolution from a 0-1 value.
    pub fn new(level: f32) -> Self {
        Self(level.clamp(0.0, 1.0))
    }

    /// Get the underlying level.
    pub fn level(&self) -> f32 {
        self.0
    }

    /// Map to discrete tree level.
    pub fn to_tree_level(&self, max_depth: usize) -> usize {
        let inverted = 1.0 - self.0; // Fine = 0, Coarse = max
        let level = (inverted * max_depth as f32).round() as usize;
        level.min(max_depth)
    }
}

impl Default for Resolution {
    fn default() -> Self {
        Self::FINE
    }
}

/// Trait for multi-resolution retrieval.
///
/// Implementors provide access to content at different granularities.
pub trait MultiResolution {
    /// Item type at finest resolution.
    type Item;
    /// Summary/aggregation type at coarser resolutions.
    type Summary;

    /// Get items at a specific resolution.
    ///
    /// Resolution 1.0 returns original items.
    /// Resolution 0.0 returns most aggregated form.
    fn at_resolution(
        &self,
        resolution: Resolution,
    ) -> Vec<ResolutionItem<Self::Item, Self::Summary>>;

    /// Get all items across all resolutions (collapsed view).
    fn collapsed(&self) -> Vec<ResolutionItem<Self::Item, Self::Summary>>;

    /// Available resolution levels.
    fn available_resolutions(&self) -> Vec<Resolution>;
}

/// An item at a particular resolution.
#[derive(Debug, Clone)]
pub struct ResolutionItem<T, S> {
    /// Unique identifier.
    pub id: usize,
    /// Resolution level.
    pub resolution: Resolution,
    /// Content (either original item or summary).
    pub content: ResolutionContent<T, S>,
    /// IDs of child items at finer resolution.
    pub children: Vec<usize>,
    /// ID of parent at coarser resolution (if any).
    pub parent: Option<usize>,
}

/// Content at a resolution level.
#[derive(Debug, Clone)]
pub enum ResolutionContent<T, S> {
    /// Original item (finest resolution).
    Item(T),
    /// Summary/aggregation (coarser resolutions).
    Summary(S),
}

impl<T, S> ResolutionContent<T, S> {
    /// Is this original content?
    pub fn is_item(&self) -> bool {
        matches!(self, Self::Item(_))
    }

    /// Is this a summary?
    pub fn is_summary(&self) -> bool {
        matches!(self, Self::Summary(_))
    }

    /// Get as item reference.
    pub fn as_item(&self) -> Option<&T> {
        match self {
            Self::Item(t) => Some(t),
            Self::Summary(_) => None,
        }
    }

    /// Get as summary reference.
    pub fn as_summary(&self) -> Option<&S> {
        match self {
            Self::Item(_) => None,
            Self::Summary(s) => Some(s),
        }
    }
}

/// Strategy for navigating hierarchical structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraversalStrategy {
    /// Search all levels simultaneously (RAPTOR collapsed).
    Collapsed,
    /// Start at coarse, descend to fine (tree traversal).
    TopDown,
    /// Start at fine, ascend to coarse.
    BottomUp,
    /// Search at a fixed resolution.
    FixedResolution,
    /// Coarse filter, fine rank (two-phase).
    CoarseToFine,
}

/// Configuration for hierarchical retrieval.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Traversal strategy.
    pub strategy: TraversalStrategy,
    /// Target resolution (for FixedResolution strategy).
    pub target_resolution: Resolution,
    /// Number of candidates at coarse level (for CoarseToFine).
    pub coarse_candidates: usize,
    /// Final k to return.
    pub k: usize,
    /// Whether to include parent/child links in results.
    pub include_context: bool,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            strategy: TraversalStrategy::Collapsed,
            target_resolution: Resolution::FINE,
            coarse_candidates: 50,
            k: 10,
            include_context: false,
        }
    }
}

impl HierarchicalConfig {
    /// Configure for collapsed retrieval (RAPTOR-style).
    pub fn collapsed(k: usize) -> Self {
        Self {
            strategy: TraversalStrategy::Collapsed,
            k,
            ..Default::default()
        }
    }

    /// Configure for tree traversal.
    pub fn tree_traversal(k: usize) -> Self {
        Self {
            strategy: TraversalStrategy::TopDown,
            k,
            include_context: true,
            ..Default::default()
        }
    }

    /// Configure for coarse-to-fine (two-phase).
    pub fn coarse_to_fine(k: usize, coarse_candidates: usize) -> Self {
        Self {
            strategy: TraversalStrategy::CoarseToFine,
            coarse_candidates,
            k,
            ..Default::default()
        }
    }

    /// Configure for fixed resolution.
    pub fn at_resolution(resolution: Resolution, k: usize) -> Self {
        Self {
            strategy: TraversalStrategy::FixedResolution,
            target_resolution: resolution,
            k,
            ..Default::default()
        }
    }
}

/// Statistics about a hierarchical index.
#[derive(Debug, Clone)]
pub struct HierarchyStats {
    /// Number of levels.
    pub num_levels: usize,
    /// Items per level.
    pub level_sizes: Vec<usize>,
    /// Total nodes.
    pub total_nodes: usize,
    /// Average fanout.
    pub avg_fanout: f32,
    /// Max fanout.
    pub max_fanout: usize,
}

impl HierarchyStats {
    /// Compression ratio (leaf_nodes / total_nodes).
    pub fn compression_ratio(&self) -> f32 {
        if self.total_nodes == 0 {
            return 1.0;
        }
        let leaf_count = self.level_sizes.first().copied().unwrap_or(0);
        leaf_count as f32 / self.total_nodes as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_levels() {
        assert_eq!(Resolution::COARSE.level(), 0.0);
        assert_eq!(Resolution::FINE.level(), 1.0);

        let mid = Resolution::new(0.5);
        assert_eq!(mid.to_tree_level(4), 2);
    }

    #[test]
    fn test_resolution_clamping() {
        let above = Resolution::new(1.5);
        assert_eq!(above.level(), 1.0);

        let below = Resolution::new(-0.5);
        assert_eq!(below.level(), 0.0);
    }

    #[test]
    fn test_config_builders() {
        let collapsed = HierarchicalConfig::collapsed(10);
        assert_eq!(collapsed.strategy, TraversalStrategy::Collapsed);

        let c2f = HierarchicalConfig::coarse_to_fine(10, 50);
        assert_eq!(c2f.strategy, TraversalStrategy::CoarseToFine);
        assert_eq!(c2f.coarse_candidates, 50);
    }
}
