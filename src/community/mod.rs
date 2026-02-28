//! Community detection algorithms for graphs.
//!
//! Given a graph, find natural groupings where nodes within groups are
//! densely connected, and connections between groups are sparse.
//!
//! ## The Modularity Objective
//!
//! Most algorithms optimize **modularity** Q, which compares the actual
//! number of edges within communities to the expected number in a random
//! graph with the same degree sequence:
//!
//! ```text
//! Q = (1/2m) × Σ[A_ij - γ(k_i × k_j)/(2m)] × δ(c_i, c_j)
//! ```
//!
//! Where:
//! - m = total edge weight (sum of all edges)
//! - A_ij = edge weight between i and j
//! - k_i = degree of node i
//! - γ = resolution parameter
//! - δ(c_i, c_j) = 1 if i and j are in same community
//!
//! **Intuition**: For each pair in the same community, we add (actual edges) -
//! (expected edges). A good sheaftion has Q > 0, meaning more internal edges
//! than expected by chance.
//!
//! ## The Resolution Parameter γ
//!
//! The resolution parameter controls granularity:
//!
//! - **γ = 1**: Standard modularity (default)
//! - **γ > 1**: Smaller communities (higher penalty for merging)
//! - **γ < 1**: Larger communities (lower penalty for merging)
//!
//! This is crucial because modularity has a **resolution limit**—it can't
//! detect communities smaller than √(2m). Increasing γ helps find fine-grained
//! structure.
//!
//! ## Algorithms
//!
//! ### Leiden (Recommended)
//!
//! The Leiden algorithm ([Traag et al. 2019](https://arxiv.org/abs/1810.08473))
//! improves on Louvain with a critical guarantee: **communities are always
//! well-connected**.
//!
//! **Three phases**:
//! 1. **Local moving**: Greedily move nodes to improve modularity
//! 2. **Refinement**: Split communities that aren't internally connected
//! 3. **Aggregation**: Contract graph, repeat
//!
//! The refinement phase is the key innovation. Louvain skips this, which can
//! produce pathological results—nodes labeled as "same community" with no
//! path between them.
//!
//! ### Louvain
//!
//! The original fast modularity algorithm ([Blondel et al. 2008](https://arxiv.org/abs/0803.0476)).
//! Still useful as a baseline, but **can produce disconnected communities**.
//!
//! In experiments, Louvain produced disconnected communities in up to 16% of
//! cases when run iteratively. Use Leiden instead for production.
//!
//! ### Label Propagation
//!
//! O(E) algorithm that spreads labels through the network. Each node adopts
//! the most common label among its neighbors. Fast but approximate.
//!
//! ## Usage
//!
//! ```rust
//! use petgraph::graph::UnGraph;
//! use sheaf::community::{Leiden, CommunityDetection};
//!
//! // Build a graph
//! let mut graph = UnGraph::<(), ()>::new_undirected();
//! let a = graph.add_node(());
//! let b = graph.add_node(());
//! let c = graph.add_node(());
//! graph.add_edge(a, b, ());
//! graph.add_edge(b, c, ());
//!
//! // Detect communities
//! let leiden = Leiden::new();
//! let communities = leiden.detect(&graph).unwrap();
//! // communities[i] = community ID for node i
//! ```
//!
//! ## References
//!
//! - Traag, Waltman, van Eck (2019). "From Louvain to Leiden: guaranteeing
//!   well-connected communities." Scientific Reports 9, 5233.
//! - Blondel et al. (2008). "Fast unfolding of communities in large networks."
//! - Newman & Girvan (2004). "Finding and evaluating community structure in networks."

mod label_prop;
mod leiden;
mod louvain;
mod traits;

#[cfg(feature = "knn-graph")]
mod knn_graph;

pub use label_prop::LabelPropagation;
pub use leiden::Leiden;
pub use louvain::Louvain;
pub use traits::CommunityDetection;

#[cfg(feature = "knn-graph")]
pub use knn_graph::{
    knn_graph_from_embeddings, knn_graph_with_config, KnnGraphConfig, WeightFunction,
};
