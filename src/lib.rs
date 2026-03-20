#![warn(missing_docs)]
//\! # sheaf
//!
//! Hierarchical abstraction: tree structures + reconciliation / conformal primitives for multi-resolution views.
//!
//! **Default build** is hierarchy-first (minimal dependencies). Algorithmic clustering and community detection
//! are opt-in via feature flags.

#[cfg(feature = "cluster")]
pub mod cluster;
#[cfg(feature = "community")]
pub mod community;
/// Error types used across `sheaf`.
pub mod error;
pub mod hierarchy;
pub mod metrics;
pub mod reconciliation;
#[cfg(feature = "summarize")]
pub mod summarize;

#[cfg(any(feature = "rkhs", feature = "wass"))]
pub mod distribution_distance;

#[cfg(test)]
mod reconciliation_tests;

#[cfg(feature = "cluster")]
pub use crate::cluster::ItDendrogram;
pub use crate::hierarchy::{HierarchicalConformal, HierarchyTree, ReconciliationScore};
pub use crate::reconciliation::{reconcile, ReconciliationMethod, SummingMatrix};

// Re-exports from Strata merge
pub use error::{Error, Result};
pub use metrics::{ari, completeness, fowlkes_mallows, homogeneity, nmi, purity, v_measure};

#[cfg(any(feature = "rkhs", feature = "wass"))]
pub use distribution_distance::{DistributionDistance, DistributionDistanceConfig};

#[cfg(feature = "cluster")]
pub use cluster::{
    Clustering, CompositeDistance, Constraint, CopKmeans, CorrelationClustering, CorrelationResult,
    CosineDistance, DataRef, DenStream, DistanceMetric, Euclidean, FlatRef, Gmm,
    HierarchicalClustering, InnerProductDistance, Kmeans, Linkage, MiniBatchKmeans, SignedEdge,
    SoftClustering, SquaredEuclidean,
};

#[cfg(feature = "community")]
pub use community::{CommunityDetection, LabelPropagation, Leiden, Louvain};

#[cfg(feature = "knn-graph")]
pub use community::{
    knn_graph_from_embeddings, knn_graph_with_config, KnnGraphConfig, WeightFunction,
};

#[cfg(feature = "summarize")]
pub use summarize::Summarizer;

pub use hierarchy::{Dendrogram, RaptorTree, TreeConfig};
