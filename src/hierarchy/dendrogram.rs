//! Dendrogram for hierarchical clustering visualization.
//!
//! A dendrogram represents the nested structure of clusters produced
//! by agglomerative (bottom-up) clustering.

use crate::error::Result;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A dendrogram representing hierarchical cluster merges.
///
/// Each merge combines two clusters into one, recording:
/// - Which clusters were merged
/// - The distance at which they merged
/// - The size of the resulting cluster
#[derive(Debug, Clone)]
pub struct Dendrogram {
    /// Merge history: (cluster_a, cluster_b, distance, new_size)
    merges: Vec<Merge>,
    /// Number of original items.
    n_items: usize,
}

/// A single merge operation in the dendrogram.
#[derive(Debug, Clone, Copy)]
pub struct Merge {
    /// First cluster being merged (index).
    pub cluster_a: usize,
    /// Second cluster being merged (index).
    pub cluster_b: usize,
    /// Distance/dissimilarity at which merge occurred.
    pub distance: f64,
    /// Size of resulting cluster.
    pub size: usize,
}

impl Dendrogram {
    /// Create a new dendrogram for n items.
    pub fn new(n_items: usize) -> Self {
        Self {
            merges: Vec::with_capacity(n_items.saturating_sub(1)),
            n_items,
        }
    }

    /// Record a merge operation.
    pub fn add_merge(&mut self, cluster_a: usize, cluster_b: usize, distance: f64, size: usize) {
        self.merges.push(Merge {
            cluster_a,
            cluster_b,
            distance,
            size,
        });
    }

    /// Get cluster assignments at a given distance threshold.
    ///
    /// All merges with distance > threshold are "cut", producing
    /// separate clusters.
    pub fn cut_at_distance(&self, threshold: f64) -> Vec<usize> {
        let n_applied = self
            .merges
            .iter()
            .position(|m| m.distance > threshold)
            .unwrap_or(self.merges.len());
        self.labels_after_merges(n_applied)
    }

    /// Cluster labels after applying the first `n_applied` merges.
    fn labels_after_merges(&self, n_applied: usize) -> Vec<usize> {
        // Track which cluster each original item belongs to.
        // cluster_map[i] = current cluster ID for item i
        let mut cluster_map: Vec<usize> = (0..self.n_items).collect();

        // Track which cluster IDs map to which "merged" cluster
        // When we merge clusters A and B, all items in A and B get the same new ID
        let mut cluster_id_map: Vec<usize> = (0..(2 * self.n_items)).collect();

        for (i, merge) in self.merges.iter().take(n_applied).enumerate() {
            // New cluster ID for this merge
            let new_cluster_id = self.n_items + i;

            // Find what the actual current cluster IDs are for merge.cluster_a and merge.cluster_b
            // by following the id_map chain
            let mut id_a = merge.cluster_a;
            while cluster_id_map[id_a] != id_a && id_a < cluster_id_map.len() {
                id_a = cluster_id_map[id_a];
            }

            let mut id_b = merge.cluster_b;
            while cluster_id_map[id_b] != id_b && id_b < cluster_id_map.len() {
                id_b = cluster_id_map[id_b];
            }

            // Expand cluster_id_map if needed
            while cluster_id_map.len() <= new_cluster_id {
                cluster_id_map.push(cluster_id_map.len());
            }

            // Map both cluster_a and cluster_b to the new cluster
            cluster_id_map[id_a] = new_cluster_id;
            cluster_id_map[id_b] = new_cluster_id;
            cluster_id_map[merge.cluster_a] = new_cluster_id;
            cluster_id_map[merge.cluster_b] = new_cluster_id;
        }

        // Resolve final cluster IDs for each original item
        for slot in cluster_map.iter_mut().take(self.n_items) {
            let mut cid = *slot;
            while cid < cluster_id_map.len() && cluster_id_map[cid] != cid {
                cid = cluster_id_map[cid];
            }
            *slot = cid;
        }

        // Renumber to consecutive integers
        let mut unique: Vec<usize> = cluster_map.to_vec();
        unique.sort_unstable();
        unique.dedup();

        cluster_map
            .iter()
            .map(|&l| unique.iter().position(|&u| u == l).unwrap_or(0))
            .collect()
    }

    /// Get cluster assignments for k clusters.
    pub fn cut_to_k(&self, k: usize) -> Result<Vec<usize>> {
        if k == 0 || k > self.n_items {
            return Ok((0..self.n_items).collect());
        }

        // Number of merges needed to get k clusters from n items
        let n_merges = self.n_items.saturating_sub(k);

        if n_merges >= self.merges.len() {
            // Use all merges
            return Ok(self.cut_at_distance(f64::MAX));
        }

        // Apply exactly `n_merges` merges. Cutting at the last merge's height
        // would also apply later merges tied at that height and return fewer
        // than k clusters.
        Ok(self.labels_after_merges(n_merges))
    }

    /// Number of original items.
    pub fn n_items(&self) -> usize {
        self.n_items
    }

    /// Number of merges recorded.
    pub fn n_merges(&self) -> usize {
        self.merges.len()
    }

    /// Iterate over merges.
    pub fn merges(&self) -> impl Iterator<Item = &Merge> {
        self.merges.iter()
    }

    /// Get the merge distances (for visualization).
    pub fn distances(&self) -> Vec<f64> {
        self.merges.iter().map(|m| m.distance).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dendrogram_creation() {
        let dendro = Dendrogram::new(5);
        assert_eq!(dendro.n_items(), 5);
        assert_eq!(dendro.n_merges(), 0);
    }

    #[test]
    fn test_dendrogram_merge() {
        let mut dendro = Dendrogram::new(4);
        dendro.add_merge(0, 1, 0.5, 2);
        dendro.add_merge(2, 3, 0.7, 2);
        dendro.add_merge(4, 5, 1.0, 4); // clusters from previous merges

        assert_eq!(dendro.n_merges(), 3);
    }
}
