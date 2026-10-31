use super::traits::Clustering;
use crate::error::{Error, Result};

/// Internal node for In-Tree representation.
#[allow(dead_code)] // Work-in-progress implementation
#[derive(Debug, Clone)]
struct ITNode {
    id: usize,
    parent: Option<usize>,
    distance: f32, // Distance to parent
    density: f32,  // Local density estimate
}

/// In-tree clustering after Qiu & Li (2015), "IT-Dendrogram".
///
/// Steps:
/// 1. Estimate the density of each point as the inverse distance to its
///    k-th nearest neighbour.
/// 2. Link each point to its nearest point (over the whole data set) with
///    strictly higher density. The edge length is that distance.
/// 3. Points with no strictly denser point are roots.
/// 4. If [`with_n_clusters`](Self::with_n_clusters) is set, remove the
///    longest in-tree edges until that many trees remain.
/// 5. Each tree is one cluster.
///
/// Without `with_n_clusters` no edges are removed, so the result has one
/// cluster per root. Usually only the global density maximum is a root,
/// which gives a single cluster even for well-separated groups; ties in
/// density can add roots. Set `with_n_clusters(c)` to get `c` clusters: with
/// one root this cuts the `c - 1` longest edges.
#[derive(Debug, Clone)]
pub struct ItDendrogram {
    k: usize, // k-NN for density estimation
    n_clusters: Option<usize>,
}

impl ItDendrogram {
    /// Create a new IT-Dendrogram clusterer.
    ///
    /// - \(k\) controls the neighborhood size used for density estimation.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            n_clusters: None,
        }
    }

    /// Cut the longest in-tree edges until `n_clusters` trees remain.
    ///
    /// With a single root this removes the `n_clusters - 1` longest edges.
    /// If the in-tree already has more roots than `n_clusters`, nothing is
    /// cut and the result has one cluster per root. Edges of equal length
    /// are cut in point-index order. `fit_predict` returns
    /// `Error::InvalidClusterCount` when `n_clusters` is 0 or exceeds the
    /// number of points.
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = Some(n_clusters);
        self
    }

    fn density_estimation(&self, data: &[Vec<f32>]) -> Vec<f32> {
        // Simplified density: 1 / distance to k-th neighbor
        // Real implementations might use more robust kernel density.
        let n = data.len();
        let mut densities = vec![0.0; n];

        for i in 0..n {
            let mut distances = Vec::with_capacity(n);
            for j in 0..n {
                if i != j {
                    distances.push(self.dist(&data[i], &data[j]));
                }
            }
            distances.sort_by(|a, b| a.total_cmp(b));
            let d_k = distances.get(self.k - 1).copied().unwrap_or(1e-6);
            densities[i] = 1.0 / (d_k + 1e-9);
        }
        densities
    }

    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

impl Clustering for ItDendrogram {
    fn fit_predict(&self, data: &[Vec<f32>]) -> Result<Vec<usize>> {
        if data.is_empty() {
            return Err(Error::EmptyInput);
        }
        if self.k == 0 {
            return Err(Error::InvalidParameter {
                name: "k",
                message: "must be > 0",
            });
        }
        let n = data.len();
        if let Some(c) = self.n_clusters {
            if c == 0 || c > n {
                return Err(Error::InvalidClusterCount {
                    requested: c,
                    n_items: n,
                });
            }
        }

        // 1. Estimate densities
        let densities = self.density_estimation(data);

        // 2. Build In-Tree: link to nearest neighbor with higher density
        let mut parent = vec![None; n];
        let mut roots = Vec::new();
        let mut edge_len = vec![0.0f32; n];

        for i in 0..n {
            let mut best_dist = f32::INFINITY;
            let mut best_parent = None;

            for j in 0..n {
                if i == j {
                    continue;
                }
                if densities[j] > densities[i] {
                    let d = self.dist(&data[i], &data[j]);
                    if d < best_dist {
                        best_dist = d;
                        best_parent = Some(j);
                    }
                }
            }

            if let Some(p) = best_parent {
                parent[i] = Some(p);
                edge_len[i] = best_dist;
            } else {
                roots.push(i); // Local density maximum
            }
        }

        // Optional pruning: remove the longest edges until `c` trees remain.
        if let Some(c) = self.n_clusters {
            let n_cuts = c.saturating_sub(roots.len());
            let mut edges: Vec<usize> = (0..n).filter(|&i| parent[i].is_some()).collect();
            // Stable sort keeps index order among equal lengths.
            edges.sort_by(|&a, &b| edge_len[b].total_cmp(&edge_len[a]));
            for &i in edges.iter().take(n_cuts) {
                parent[i] = None;
            }
        }

        // 3. Assign labels based on connected components (roots)
        // Each root defines a cluster basin.
        // We trace paths up to roots.
        let mut labels = vec![0; n];
        for (i, label) in labels.iter_mut().enumerate().take(n) {
            let mut curr = i;
            // Path compression / traversal
            while let Some(p) = parent[curr] {
                curr = p;
            }
            // `curr` is now a root. Map root ID to a cluster label 0..k
            *label = curr;
        }

        // Remap labels to contiguous range 0..num_clusters
        let mut unique_labels = labels.clone();
        unique_labels.sort();
        unique_labels.dedup();

        let label_map: std::collections::HashMap<_, _> = unique_labels
            .iter()
            .enumerate()
            .map(|(i, &l)| (l, i))
            .collect();

        Ok(labels.iter().map(|l| label_map[l]).collect())
    }

    fn n_clusters(&self) -> usize {
        // 0 means the count is determined by the number of roots.
        self.n_clusters.unwrap_or(0)
    }
}
