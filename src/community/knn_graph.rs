//! kNN graph construction from embeddings using vicinity.
//!
//! This module bridges vector embeddings to community detection algorithms.
//! Given a set of embeddings, it constructs a k-nearest neighbor graph where:
//! - Each embedding becomes a node
//! - Edges connect each node to its k nearest neighbors
//! - Edge weights are similarity scores (higher = more similar)
//!
//! The resulting graph can be passed directly to Leiden/Louvain for clustering.
//!
//! # The Connection: From Embeddings to Communities
//!
//! ```text
//! Embeddings → vicinity (HNSW) → kNN Graph → Leiden → Communities
//! ```
//!
//! This is the algorithmic pipeline behind many modern clustering approaches:
//! - **Spectral clustering**: Uses kNN graph + spectral decomposition
//! - **GraphRAG**: Uses kNN + community detection for document organization
//! - **Sentence-BERT clustering**: kNN graph on sentence embeddings
//!
//! # Mathematical Background
//!
//! A kNN graph G = (V, E) is constructed as:
//! - V = {v_1, ..., v_n} where each v_i corresponds to embedding e_i
//! - E = {(v_i, v_j) : v_j ∈ kNN(v_i)} where kNN(v_i) are k nearest neighbors
//!
//! Edge weights can be:
//! - **Similarity**: w_ij = sim(e_i, e_j), higher = more connected
//! - **Distance**: w_ij = 1 / (1 + dist(e_i, e_j)), transforms distance to similarity
//!
//! # Example
//!
//! ```rust,ignore
//! use parti::community::{knn_graph_from_embeddings, Leiden, CommunityDetection};
//!
//! // Your embeddings (e.g., from sentence-transformers)
//! let embeddings: Vec<Vec<f32>> = /* ... */;
//!
//! // Build kNN graph (k=10 neighbors)
//! let graph = knn_graph_from_embeddings(&embeddings, 10)?;
//!
//! // Detect communities
//! let leiden = Leiden::new();
//! let communities = leiden.detect(&graph)?;
//! ```
//!
//! # Performance Considerations
//!
//! - For n embeddings, brute-force kNN is O(n²)
//! - Using vicinity's HNSW, we get O(n log n) approximate kNN
//! - The approximation has minimal impact on community quality
//!
//! # References
//!
//! - HNSW: Malkov & Yashunin (2018). "Efficient and robust approximate nearest neighbor search"
//! - GraphRAG: Microsoft (2024). "From Local to Global: A Graph RAG Approach"

use crate::{Error, Result};
use vicinity::hnsw::{HNSWIndex, HNSWParams};
use petgraph::graph::UnGraph;

/// Configuration for kNN graph construction.
#[derive(Debug, Clone)]
pub struct KnnGraphConfig {
    /// Number of neighbors per node (default: 10)
    pub k: usize,
    /// HNSW construction parameter M (default: 16)
    pub hnsw_m: usize,
    /// HNSW construction parameter ef_construction (default: 100)
    pub hnsw_ef_construction: usize,
    /// Whether to make the graph symmetric (default: true)
    /// If true, if A is in kNN(B), then B→A edge is also added
    pub symmetric: bool,
    /// Edge weight function (default: similarity)
    pub weight_fn: WeightFunction,
}

impl Default for KnnGraphConfig {
    fn default() -> Self {
        Self {
            k: 10,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            symmetric: true,
            weight_fn: WeightFunction::Similarity,
        }
    }
}

/// How to compute edge weights from distances.
#[derive(Debug, Clone, Copy)]
pub enum WeightFunction {
    /// w = 1 - distance (for cosine distance in [0, 2])
    Similarity,
    /// w = 1 / (1 + distance)
    InverseDistance,
    /// w = exp(-distance² / σ²) where σ is the median distance
    GaussianKernel,
    /// w = 1.0 (unweighted)
    Uniform,
}

/// Build a kNN graph from embeddings using HNSW for efficient neighbor search.
///
/// # Arguments
///
/// * `embeddings` - Vector embeddings, each of the same dimension
/// * `k` - Number of neighbors per node
///
/// # Returns
///
/// An undirected graph suitable for community detection.
///
/// # Errors
///
/// Returns an error if embeddings are empty or have inconsistent dimensions.
pub fn knn_graph_from_embeddings(embeddings: &[Vec<f32>], k: usize) -> Result<UnGraph<(), f32>> {
    knn_graph_with_config(
        embeddings,
        &KnnGraphConfig {
            k,
            ..Default::default()
        },
    )
}

/// Build a kNN graph with full configuration options.
pub fn knn_graph_with_config(
    embeddings: &[Vec<f32>],
    config: &KnnGraphConfig,
) -> Result<UnGraph<(), f32>> {
    if embeddings.is_empty() {
        return Err(Error::EmptyInput);
    }

    let dim = embeddings[0].len();
    if let Some((_, e)) = embeddings.iter().enumerate().find(|(_, e)| e.len() != dim) {
        return Err(Error::DimensionMismatch {
            expected: dim,
            found: e.len(),
        });
    }

    let n = embeddings.len();
    let k = config.k.min(n - 1); // Can't have more neighbors than n-1

    // Build HNSW index
    let params = HNSWParams {
        m: config.hnsw_m,
        ef_construction: config.hnsw_ef_construction,
        ..Default::default()
    };
    let mut hnsw = HNSWIndex::with_params(dim, params).map_err(|e| Error::Other(e.to_string()))?;

    for (i, embedding) in embeddings.iter().enumerate() {
        hnsw.add(i as u32, embedding.clone())
            .map_err(|e| Error::Other(e.to_string()))?;
    }

    // Build the index for search
    hnsw.build().map_err(|e| Error::Other(e.to_string()))?;

    // Create graph with n nodes
    let mut graph = UnGraph::<(), f32>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    // ef for search (higher = more accurate, slower)
    let ef_search = (config.k * 2).max(config.hnsw_ef_construction / 2);

    // Compute distances for Gaussian kernel if needed
    let sigma = if matches!(config.weight_fn, WeightFunction::GaussianKernel) {
        // Estimate sigma from sample distances
        let sample_size = (n / 10).clamp(10, 100);
        let mut distances = Vec::with_capacity(sample_size * k);
        for i in (0..n).step_by((n / sample_size).max(1)).take(sample_size) {
            if let Ok(neighbors) = hnsw.search(&embeddings[i], k + 1, ef_search) {
                for (_, dist) in neighbors.iter().skip(1) {
                    // skip self
                    distances.push(*dist);
                }
            }
        }
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        distances.get(distances.len() / 2).copied().unwrap_or(1.0)
    } else {
        1.0
    };

    // Find kNN for each point and add edges
    for (i, embedding) in embeddings.iter().enumerate() {
        let neighbors = hnsw
            .search(embedding, k + 1, ef_search)
            .map_err(|e| Error::Other(e.to_string()))?;

        for (neighbor_idx, distance) in neighbors {
            let neighbor_idx = neighbor_idx as usize;
            if neighbor_idx == i {
                continue; // Skip self
            }

            let weight = match config.weight_fn {
                // For L2 distance: convert to similarity.
                // Clamp to ensure positive weights (distance=0 -> weight=1)
                WeightFunction::Similarity => (1.0 - distance).max(0.001),
                WeightFunction::InverseDistance => 1.0 / (1.0 + distance),
                WeightFunction::GaussianKernel => (-distance * distance / (sigma * sigma)).exp(),
                WeightFunction::Uniform => 1.0,
            };

            // Add edge (petgraph deduplicates)
            if config.symmetric || i < neighbor_idx {
                let _ = graph.add_edge(nodes[i], nodes[neighbor_idx], weight);
            }
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_graph_basic() {
        // Create clustered embeddings
        let embeddings: Vec<Vec<f32>> = vec![
            // Cluster 1
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.95, 0.05, 0.0],
            // Cluster 2
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.05, 0.95, 0.0],
        ];

        let graph = knn_graph_from_embeddings(&embeddings, 2).unwrap();

        // Should have 6 nodes
        assert_eq!(graph.node_count(), 6);

        // Should have edges
        assert!(graph.edge_count() > 0);
    }

    #[test]
    fn test_knn_graph_empty() {
        let embeddings: Vec<Vec<f32>> = vec![];
        let result = knn_graph_from_embeddings(&embeddings, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_knn_graph_single_point() {
        let embeddings = vec![vec![1.0, 0.0]];
        let graph = knn_graph_from_embeddings(&embeddings, 5).unwrap();
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_knn_graph_weight_functions() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];

        for weight_fn in [
            WeightFunction::Similarity,
            WeightFunction::InverseDistance,
            WeightFunction::GaussianKernel,
            WeightFunction::Uniform,
        ] {
            let config = KnnGraphConfig {
                k: 2,
                weight_fn,
                ..Default::default()
            };
            let graph = knn_graph_with_config(&embeddings, &config).unwrap();

            // All weights should be positive
            for edge in graph.edge_references() {
                assert!(
                    *edge.weight() > 0.0,
                    "Weight should be positive for {:?}",
                    weight_fn
                );
            }
        }
    }
}
