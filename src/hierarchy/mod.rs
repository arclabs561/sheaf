//! Hierarchical structures for multi-resolution retrieval.
//!
//! # The Core Insight
//!
//! Information exists at multiple scales. A single question might need:
//!
//! ```text
//! Question                        │ Scale Needed
//! ────────────────────────────────┼──────────────────────
//! "What year was X founded?"      │ Single sentence
//! "Summarize the main themes"     │ Document-level
//! "How does A relate to B?"       │ Cross-section synthesis
//! ```
//!
//! **Hierarchical retrieval** lets you navigate and retrieve at any granularity,
//! from fine-grained chunks to high-level summaries.
//!
//! # Research Landscape
//!
//! Several papers address multi-scale retrieval with different trade-offs:
//!
//! | Paper | Structure | Key Insight |
//! |-------|-----------|-------------|
//! | **RAPTOR** | K-ary tree | Recursive LLM summarization |
//! | **Hierarchical Chunking** | Binary tree | Exploit document structure |
//! | **Matryoshka** | Flat embeddings | Truncatable dimensions |
//! | **Coarse-to-Fine** | Two indices | Fast filter, accurate rank |
//! | **Multi-Vector** | Multiple embeddings | One doc → many representations |
//!
//! This module provides abstractions that work across these patterns:
//!
//! - [`hierarchy`]: General multi-resolution traits and types
//! - [`RaptorTree`]: RAPTOR-style recursive summarization
//! - [`Dendrogram`]: Agglomerative clustering tree
//!
//! # Module Overview
//!
//! ## [`hierarchy`] - General Multi-Resolution Abstractions
//!
//! The [`hierarchy::Resolution`] type models granularity on a 0-1 scale:
//!
//! ```text
//! Resolution  │ Meaning
//! ────────────┼──────────────
//! 0.0 (coarse)│ Document-level
//! 0.5 (medium)│ Section-level
//! 1.0 (fine)  │ Chunk-level
//! ```
//!
//! The [`hierarchy::MultiResolution`] trait abstracts over different
//! hierarchical structures, enabling generic retrieval code.
//!
//! ## [`RaptorTree`] - Recursive Summarization
//!
//! RAPTOR (Sarthi et al., ICLR 2024) builds trees by:
//! 1. **Cluster** similar items at each level
//! 2. **Summarize** each cluster (via LLM or extractive method)
//! 3. **Recurse** until reaching root or max depth
//!
//! ```text
//! Level 3:        [Root Summary]
//!                 /             \
//! Level 2:  [Summary A]      [Summary B]
//!           /    |    \      /    |    \
//! Level 1: [s1] [s2] [s3]  [s4] [s5] [s6]
//!          /|\   |     |\   |     |   /|\
//! Level 0: ... original chunks ...
//! ```
//!
//! ## [`Dendrogram`] - Agglomerative Clustering
//!
//! Records complete merge history from hierarchical clustering:
//!
//! ```text
//!         6 (height=1.0)
//!        / \
//!       4   5 (height=0.7)
//!      / \ / \
//!     0  1 2  3 (leaves)
//! ```
//!
//! Key property: "cut" at any height to get any number of clusters.
//!
//! # Choosing a Structure
//!
//! | Use Case | Recommended |
//! |----------|-------------|
//! | RAG with summaries | [`RaptorTree`] |
//! | Cluster exploration | [`Dendrogram`] |
//! | Generic multi-scale | [`hierarchy`] traits |
//! | Two-phase retrieval | [`hierarchy::HierarchicalConfig::coarse_to_fine`] |
//!
//! # References
//!
//! - Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for
//!   Tree-Organized Retrieval." ICLR.
//! - Kusupati et al. (2022). "Matryoshka Representation Learning." NeurIPS.
//! - Wei et al. (2022). "ADSampling." VLDB.

pub mod conformal;
mod dendrogram;
pub mod foundations;
#[allow(clippy::module_inception)]
pub mod hierarchy;
mod node;
mod raptor;
pub mod tree;
mod validate;

pub use conformal::HierarchicalConformal;
pub use dendrogram::Dendrogram;
pub use foundations::{
    embedding_distortion, gromov_hyperbolicity, is_ultrametric, subdominant_ultrametric,
    Ultrametric, UltrametricTree,
};
pub use hierarchy::{
    HierarchicalConfig, HierarchyStats, MultiResolution, Resolution, ResolutionContent,
    ResolutionItem, TraversalStrategy,
};
pub use node::Node;
pub use raptor::{RaptorTree, TreeConfig};
pub use tree::HierarchyTree;
pub use validate::{
    validate_tree_structure, HealthCheck, HealthReport, Severity, ValidationIssue, ValidationReport,
};
