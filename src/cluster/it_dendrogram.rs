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

/// IT-Dendrogram clustering (Qiu & Li, 2015).
///
/// Constructs an "effective in-tree" structure that reveals underlying cluster
/// geometry even in high-dimensional or non-Euclidean spaces.
///
/// Steps:
/// 1. Estimate local density for each point (k-NN distance).
/// 2. Link each point to its nearest neighbor with *higher* density.
/// 3. Points with no higher-density neighbor are local maxima (roots).
/// 4. Prune "undesired edges" (low-density bridges) to form clusters.
#[derive(Debug, Clone)]
pub struct ItDendrogram {
    k: usize, // k-NN for density estimation
}

impl ItDendrogram {
    /// Create a new IT-Dendrogram clusterer.
    ///
    /// - \(k\) controls the neighborhood size used for density estimation.
    pub fn new(k: usize) -> Self {
        Self { k }
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
        let n = data.len();

        // 1. Estimate densities
        let densities = self.density_estimation(data);

        // 2. Build In-Tree: link to nearest neighbor with higher density
        let mut parent = vec![None; n];
        let mut roots = Vec::new();

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
            } else {
                roots.push(i); // Local density maximum
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
        0 // Determined dynamically
    }
}
