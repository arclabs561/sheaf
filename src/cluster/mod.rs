//! Clustering algorithms for grouping similar items.
//!
//! This module provides clustering algorithms used to group items before
//! summarization in hierarchical trees.
//!
//! ## Hard vs Soft Clustering
//!
//! **Hard clustering** assigns each item to exactly one cluster. Simple, but
//! loses information when items genuinely span multiple groups.
//!
//! **Soft clustering** gives each item a probability distribution over clusters.
//! A text chunk might be 60% about "machine learning", 30% about "statistics",
//! 10% about "software". This reflects reality better than forcing a choice.
//!
//! For RAPTOR-style hierarchical summarization, **soft clustering (GMM) is
//! recommended** because text chunks often span multiple topics.
//!
//! ## Algorithms
//!
//! ### Gaussian Mixture Model (GMM)
//!
//! Models data as a mixture of k Gaussian distributions:
//!
//! ```text
//! P(x) = Σ π_k × N(x | μ_k, Σ_k)
//! ```
//!
//! Where π_k is the mixture weight (probability of cluster k), and N is the
//! Gaussian density with mean μ_k and covariance Σ_k.
//!
//! **EM Algorithm**:
//! 1. **E-step**: Compute P(cluster k | point x) for each point
//! 2. **M-step**: Update μ, Σ, π to maximize likelihood
//! 3. Repeat until convergence
//!
//! **When to use**: When you want soft assignments, or when clusters have
//! different shapes/sizes (unlike K-means which assumes spherical clusters).
//!
//! ### K-means
//!
//! The classic algorithm: assign each point to the nearest centroid, then
//! update centroids to the mean of their points. Repeat.
//!
//! **Objective**: Minimize within-cluster sum of squares:
//!
//! ```text
//! J = Σ_k Σ_{x ∈ C_k} ||x - μ_k||²
//! ```
//!
//! **Assumptions**:
//! - Clusters are roughly spherical
//! - Clusters have similar sizes
//! - You know k in advance
//!
//! **When to use**: Fast initial exploration, or when you need hard assignments
//! and can accept the spherical assumption.
//!
//! ### Hierarchical (Agglomerative) Clustering
//!
//! Bottom-up: start with each point as its own cluster, repeatedly merge
//! the two closest clusters until one remains. The merge history forms a
//! **dendrogram**—a binary tree you can cut at any height to get k clusters.
//!
//! **Linkage methods** determine "distance between clusters":
//!
//! | Linkage | Distance | Effect |
//! |---------|----------|--------|
//! | Single | min(pairwise) | Chaining; elongated clusters |
//! | Complete | max(pairwise) | Compact, spherical clusters |
//! | Average | mean(pairwise) | Balanced compromise |
//! | Ward | Variance increase | Minimizes within-cluster variance |
//!
//! **When to use**: When you want to explore cluster structure at multiple
//! granularities, or don't know k in advance.
//!
//! ## Usage
//!
//! ```rust
//! use sheaf::cluster::{Kmeans, Gmm, Clustering, SoftClustering};
//!
//! let data = vec![
//!     vec![0.0, 0.0],
//!     vec![0.1, 0.1],
//!     vec![10.0, 10.0],
//!     vec![10.1, 10.1],
//! ];
//!
//! // Hard clustering with K-means
//! let labels = Kmeans::new(2).fit_predict(&data).unwrap();
//! assert_eq!(labels[0], labels[1]);  // First two together
//! assert_ne!(labels[0], labels[2]);  // Separate from last two
//!
//! // Soft clustering with GMM
//! let probs = Gmm::new()
//!     .with_n_components(2)
//!     .fit_predict_proba(&data)
//!     .unwrap();
//! // probs[i][k] = P(point i belongs to cluster k)
//! ```

mod dbscan;
mod gmm;
mod hierarchical;
mod it_dendrogram;
mod kmeans;
mod kmeans_elkan;
mod traits;

#[cfg(feature = "spectral")]
pub mod spectral;

pub use dbscan::{Dbscan, DbscanExt, NOISE};
pub use gmm::Gmm;
pub use hierarchical::{HierarchicalClustering, Linkage};
pub use it_dendrogram::ItDendrogram;
pub use kmeans::Kmeans;
pub use kmeans_elkan::{ElkanStats, KmeansElkan};
pub use traits::{Clustering, SoftClustering};

// Re-export distance metric types from clump so users don't need a direct
// clump dependency to select a metric.
pub use clump::cluster::distance::{
    CompositeDistance, CosineDistance, DistanceMetric, Euclidean, InnerProductDistance,
    SquaredEuclidean,
};

// Constrained clustering (COP-Kmeans, Wagstaff et al. 2001).
pub use clump::cluster::constrained::{Constraint, CopKmeans};

// Streaming/online clustering (Mini-Batch K-means, Sculley 2010; DenStream, Cao et al. 2006).
pub use clump::cluster::denstream::DenStream;
pub use clump::cluster::streaming::MiniBatchKmeans;

// Correlation clustering (Bansal et al. 2004) -- no k required.
pub use clump::cluster::correlation::{CorrelationClustering, CorrelationResult, SignedEdge};

// Data input abstractions (zero-copy flat buffer, ndarray views).
pub use clump::cluster::flat::{DataRef, FlatRef};

#[cfg(feature = "spectral")]
pub use spectral::{AffinityType, SpectralClustering};
