//! Clustering evaluation metrics.
//!
//! Measures for assessing clustering quality by comparing predicted clusters
//! to ground truth labels.
//!
//! # Metrics Overview
//!
//! | Metric | Range | Best | Properties |
//! |--------|-------|------|------------|
//! | [`nmi`] | [0, 1] | 1 | Normalized, chance-corrected |
//! | AMI | [-1, 1] | 1 | Adjusted for chance (not implemented yet) |
//! | [`ari`] | [-1, 1] | 1 | Adjusted Rand Index |
//! | [`purity`] | [0, 1] | 1 | Simple, biased toward many clusters |
//! | [`homogeneity`] | [0, 1] | 1 | Each cluster has one class |
//! | [`completeness`] | [0, 1] | 1 | Each class in one cluster |
//! | [`v_measure`] | [0, 1] | 1 | Harmonic mean of above two |
//!
//! # When to Use Which
//!
//! - **NMI**: General-purpose, widely used, comparable across datasets
//! - **ARI**: When you want to penalize random clustering
//! - **Purity**: Simple interpretation, but favors over-clustering
//! - **V-Measure**: When you care about both homogeneity and completeness
//!
//! # Example
//!
//! ```rust
//! use parti::metrics::{nmi, ari, purity};
//!
//! let pred = [0, 0, 1, 1, 2, 2];
//! let truth = [0, 0, 0, 1, 1, 1];
//!
//! let nmi_score = nmi(&pred, &truth);
//! let ari_score = ari(&pred, &truth);
//! let purity_score = purity(&pred, &truth);
//! ```
//!
//! # References
//!
//! - Hubert & Arabie (1985). "Comparing partitions" (ARI)
//! - Strehl & Ghosh (2002). "Cluster ensembles" (NMI)
//! - Rosenberg & Hirschberg (2007). "V-Measure"
//! - Vinh et al. (2010). "Information theoretic measures for clusterings comparison"

use std::collections::HashMap;

/// Normalized Mutual Information between two clusterings.
///
/// NMI measures the agreement between two clusterings, normalized to [0, 1].
/// A value of 1 indicates perfect agreement.
///
/// ```text
/// NMI(U, V) = 2 * I(U; V) / (H(U) + H(V))
/// ```
///
/// where I(U; V) is mutual information and H is entropy.
///
/// # Arguments
///
/// * `pred` - Predicted cluster assignments
/// * `truth` - Ground truth cluster assignments
///
/// # Returns
///
/// NMI score in [0, 1]. Higher is better.
///
/// # Example
///
/// ```rust
/// use parti::metrics::nmi;
///
/// // Perfect clustering
/// let pred = [0, 0, 1, 1];
/// let truth = [0, 0, 1, 1];
/// assert!((nmi(&pred, &truth) - 1.0).abs() < 0.01);
///
/// // Random clustering has low NMI
/// let pred = [0, 1, 0, 1];
/// let truth = [0, 0, 1, 1];
/// assert!(nmi(&pred, &truth) < 0.5);
/// ```
pub fn nmi(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.is_empty() {
        return 0.0;
    }

    let (joint, n) = build_contingency_table(pred, truth);

    // Use surp for the actual computation
    #[cfg(feature = "metrics")]
    {
        let n_f = n as f64;

        // Build probability distributions
        let mut p_pred = HashMap::new();
        let mut p_truth = HashMap::new();

        for &p in pred {
            *p_pred.entry(p).or_insert(0) += 1;
        }
        for &t in truth {
            *p_truth.entry(t).or_insert(0) += 1;
        }

        // Entropy of predictions
        let h_pred: f64 = p_pred
            .values()
            .map(|&c| {
                let p = c as f64 / n_f;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // Entropy of truth
        let h_truth: f64 = p_truth
            .values()
            .map(|&c| {
                let p = c as f64 / n_f;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // Mutual information
        let mut mi = 0.0;
        for (&(p, t), &count) in &joint {
            if count > 0 {
                let p_joint = count as f64 / n_f;
                let p_p = *p_pred.get(&p).unwrap_or(&0) as f64 / n_f;
                let p_t = *p_truth.get(&t).unwrap_or(&0) as f64 / n_f;
                if p_p > 0.0 && p_t > 0.0 {
                    mi += p_joint * (p_joint / (p_p * p_t)).ln();
                }
            }
        }

        // Normalize
        let denom = h_pred + h_truth;
        if denom > 0.0 {
            2.0 * mi / denom
        } else {
            1.0 // Both are constant
        }
    }

    #[cfg(not(feature = "metrics"))]
    {
        // Fallback implementation without surp
        nmi_impl(pred, truth, &joint, n)
    }
}

#[cfg(not(feature = "metrics"))]
fn nmi_impl(
    pred: &[usize],
    truth: &[usize],
    joint: &HashMap<(usize, usize), usize>,
    n: usize,
) -> f64 {
    let _ = n; // suppress warning when unused
    let n_f = pred.len() as f64;

    let mut p_pred = HashMap::new();
    let mut p_truth = HashMap::new();

    for &p in pred {
        *p_pred.entry(p).or_insert(0) += 1;
    }
    for &t in truth {
        *p_truth.entry(t).or_insert(0) += 1;
    }

    let h_pred: f64 = p_pred
        .values()
        .map(|&c| {
            let p = c as f64 / n_f;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let h_truth: f64 = p_truth
        .values()
        .map(|&c| {
            let p = c as f64 / n_f;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    let mut mi = 0.0;
    for (&(p, t), &count) in joint {
        if count > 0 {
            let p_joint = count as f64 / n_f;
            let p_p = *p_pred.get(&p).unwrap_or(&0) as f64 / n_f;
            let p_t = *p_truth.get(&t).unwrap_or(&0) as f64 / n_f;
            if p_p > 0.0 && p_t > 0.0 {
                mi += p_joint * (p_joint / (p_p * p_t)).ln();
            }
        }
    }

    let denom = h_pred + h_truth;
    if denom > 0.0 {
        2.0 * mi / denom
    } else {
        1.0
    }
}

/// Adjusted Rand Index between two clusterings.
///
/// ARI is the corrected-for-chance version of the Rand Index.
/// A value of 0 indicates random clustering, 1 indicates perfect agreement.
///
/// # Arguments
///
/// * `pred` - Predicted cluster assignments
/// * `truth` - Ground truth cluster assignments
///
/// # Returns
///
/// ARI score in [-1, 1]. Higher is better. 0 = random, 1 = perfect.
///
/// # Example
///
/// ```rust
/// use parti::metrics::ari;
///
/// let pred = [0, 0, 1, 1];
/// let truth = [0, 0, 1, 1];
/// assert!((ari(&pred, &truth) - 1.0).abs() < 0.01);
/// ```
pub fn ari(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.is_empty() {
        return 0.0;
    }

    let (joint, n) = build_contingency_table(pred, truth);

    // Row sums (a_i) and column sums (b_j)
    let mut row_sums = HashMap::new();
    let mut col_sums = HashMap::new();

    for (&(p, t), &count) in &joint {
        *row_sums.entry(p).or_insert(0usize) += count;
        *col_sums.entry(t).or_insert(0usize) += count;
    }

    // Sum of C(n_ij, 2)
    let mut sum_comb_ij: f64 = 0.0;
    for &count in joint.values() {
        sum_comb_ij += comb2(count) as f64;
    }

    // Sum of C(a_i, 2) and C(b_j, 2)
    let sum_comb_a: f64 = row_sums.values().map(|&a| comb2(a) as f64).sum();
    let sum_comb_b: f64 = col_sums.values().map(|&b| comb2(b) as f64).sum();

    let comb_n = comb2(n) as f64;

    // ARI = (index - expected) / (max - expected)
    let expected = sum_comb_a * sum_comb_b / comb_n;
    let max_index = (sum_comb_a + sum_comb_b) / 2.0;

    let denom = max_index - expected;
    if denom.abs() < 1e-10 {
        return 1.0; // Perfect agreement when both clusterings are identical
    }

    (sum_comb_ij - expected) / denom
}

/// Purity of clustering with respect to ground truth.
///
/// For each cluster, find the most common ground truth label.
/// Purity is the fraction of correctly assigned points.
///
/// Note: Purity increases with more clusters and is 1.0 when each point
/// is its own cluster. Use with caution.
///
/// # Arguments
///
/// * `pred` - Predicted cluster assignments
/// * `truth` - Ground truth cluster assignments
///
/// # Returns
///
/// Purity score in [0, 1]. Higher is better.
pub fn purity(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.is_empty() {
        return 0.0;
    }

    let n = pred.len();
    let (joint, _) = build_contingency_table(pred, truth);

    // For each predicted cluster, find max overlap with any true class
    let mut cluster_maxes: HashMap<usize, usize> = HashMap::new();

    for (&(p, _), &count) in &joint {
        let current_max = cluster_maxes.entry(p).or_insert(0);
        *current_max = (*current_max).max(count);
    }

    let correct: usize = cluster_maxes.values().sum();
    correct as f64 / n as f64
}

/// Homogeneity: each cluster contains only members of a single class.
///
/// H = 1 - H(C|K) / H(C)
///
/// where C is classes (truth) and K is clusters (pred).
pub fn homogeneity(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.is_empty() {
        return 0.0;
    }

    let (h_c, h_c_given_k) = conditional_entropies(pred, truth);

    if h_c < 1e-10 {
        return 1.0; // All same class
    }

    1.0 - h_c_given_k / h_c
}

/// Completeness: all members of a given class are assigned to the same cluster.
///
/// C = 1 - H(K|C) / H(K)
///
/// where K is clusters (pred) and C is classes (truth).
pub fn completeness(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.is_empty() {
        return 0.0;
    }

    let (h_k, h_k_given_c) = conditional_entropies(truth, pred);

    if h_k < 1e-10 {
        return 1.0; // All same cluster
    }

    1.0 - h_k_given_c / h_k
}

/// V-Measure: harmonic mean of homogeneity and completeness.
///
/// V = 2 * (homogeneity * completeness) / (homogeneity + completeness)
pub fn v_measure(pred: &[usize], truth: &[usize]) -> f64 {
    let h = homogeneity(pred, truth);
    let c = completeness(pred, truth);

    if h + c < 1e-10 {
        return 0.0;
    }

    2.0 * h * c / (h + c)
}

/// Fowlkes-Mallows Index.
///
/// Geometric mean of precision and recall of pairwise cluster membership.
pub fn fowlkes_mallows(pred: &[usize], truth: &[usize]) -> f64 {
    if pred.len() != truth.len() || pred.len() < 2 {
        return 0.0;
    }

    let n = pred.len();
    let mut tp = 0usize; // Same cluster in both
    let mut fp = 0usize; // Same in pred, different in truth
    let mut fn_ = 0usize; // Different in pred, same in truth

    for i in 0..n {
        for j in (i + 1)..n {
            let same_pred = pred[i] == pred[j];
            let same_truth = truth[i] == truth[j];

            match (same_pred, same_truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };

    (precision * recall).sqrt()
}

// Helper functions

fn build_contingency_table(
    pred: &[usize],
    truth: &[usize],
) -> (HashMap<(usize, usize), usize>, usize) {
    let mut table = HashMap::new();
    for (&p, &t) in pred.iter().zip(truth.iter()) {
        *table.entry((p, t)).or_insert(0) += 1;
    }
    (table, pred.len())
}

fn comb2(n: usize) -> usize {
    if n < 2 {
        0
    } else {
        n * (n - 1) / 2
    }
}

fn conditional_entropies(a: &[usize], b: &[usize]) -> (f64, f64) {
    let n = a.len() as f64;

    // Count a values
    let mut count_a = HashMap::new();
    for &v in a {
        *count_a.entry(v).or_insert(0usize) += 1;
    }

    // H(A)
    let h_a: f64 = count_a
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            if p > 0.0 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();

    // H(A|B) = Σ_b P(b) H(A|B=b)
    let mut count_b = HashMap::new();
    let mut joint = HashMap::new();

    for (&va, &vb) in a.iter().zip(b.iter()) {
        *count_b.entry(vb).or_insert(0usize) += 1;
        *joint.entry((va, vb)).or_insert(0usize) += 1;
    }

    let mut h_a_given_b = 0.0;
    for (&vb, &nb) in &count_b {
        let p_b = nb as f64 / n;
        let mut h_a_in_b = 0.0;

        for &va in count_a.keys() {
            let n_ab = *joint.get(&(va, vb)).unwrap_or(&0);
            if n_ab > 0 && nb > 0 {
                let p_a_given_b = n_ab as f64 / nb as f64;
                h_a_in_b -= p_a_given_b * p_a_given_b.ln();
            }
        }

        h_a_given_b += p_b * h_a_in_b;
    }

    (h_a, h_a_given_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmi_perfect() {
        let pred = [0, 0, 1, 1, 2, 2];
        let truth = [0, 0, 1, 1, 2, 2];
        assert!((nmi(&pred, &truth) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_nmi_permuted() {
        // Same clustering, different labels
        let pred = [1, 1, 0, 0, 2, 2];
        let truth = [0, 0, 1, 1, 2, 2];
        assert!((nmi(&pred, &truth) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ari_perfect() {
        let pred = [0, 0, 1, 1];
        let truth = [0, 0, 1, 1];
        assert!((ari(&pred, &truth) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_purity_perfect() {
        let pred = [0, 0, 1, 1];
        let truth = [0, 0, 1, 1];
        assert!((purity(&pred, &truth) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_purity_overclustering() {
        // Each point is its own cluster
        let pred = [0, 1, 2, 3];
        let truth = [0, 0, 1, 1];
        // Purity should be 1.0 (each cluster is pure)
        assert!((purity(&pred, &truth) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_homogeneity_completeness() {
        let pred = [0, 0, 1, 1];
        let truth = [0, 0, 1, 1];
        assert!((homogeneity(&pred, &truth) - 1.0).abs() < 0.01);
        assert!((completeness(&pred, &truth) - 1.0).abs() < 0.01);
        assert!((v_measure(&pred, &truth) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_fowlkes_mallows_perfect() {
        let pred = [0, 0, 1, 1];
        let truth = [0, 0, 1, 1];
        assert!((fowlkes_mallows(&pred, &truth) - 1.0).abs() < 0.01);
    }
}
