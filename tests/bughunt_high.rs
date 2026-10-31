//! Regression tests for RAPTOR tree depth and Elkan k-means assignment.
#![allow(dead_code, clippy::unwrap_used, clippy::needless_range_loop)]

use sheaf::RaptorTree;
use sheaf::TreeConfig;

/// raptor.rs:171-176 collects `tree.get_leaf_content(id)` for every cluster
/// member; from level 2 on the members are summary nodes, as_leaf() is None,
/// the cluster is skipped and the loop breaks. Sarthi et al. 2024 §3 build
/// the tree recursively (summaries of summaries) until clustering stops.
/// 12 leaves, fanout 3: level 1 has 4 summaries, level 2 should have 2.
#[test]
fn raptor_builds_levels_above_one() {
    let items: Vec<String> = (0..12).map(|i| format!("chunk{i}")).collect();
    let tree: RaptorTree<String> = RaptorTree::build(
        items,
        TreeConfig::default().with_fanout(3).with_max_depth(4),
        |ids: &[usize], fanout: usize| ids.chunks(fanout).map(|c| c.to_vec()).collect(),
        |group: &[&String]| {
            group
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("+")
        },
    )
    .unwrap();
    let per_level: Vec<usize> = (0..tree.depth())
        .map(|l| tree.get_level(l).unwrap().len())
        .collect();
    assert!(tree.depth() >= 3, "levels {per_level:?}");
}

// ------------------------------------------------------------------ Elkan

/// Elkan (2003) must reproduce Lloyd's iterates from the same initial centres.
/// The reference below replays KmeansElkan's own k-means++ initialisation
/// (same StdRng stream, same f32 arithmetic) and then runs plain Lloyd with the
/// same stopping rule.
#[cfg(feature = "cluster")]
fn lloyd_replica(data: &[Vec<f32>], k: usize, seed: u64, max_iter: usize, tol: f32) -> Vec<usize> {
    use rand::prelude::*;
    let n = data.len();
    let d = data[0].len();
    let sq = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
    let mut rng: Box<dyn RngCore> = Box::new(StdRng::seed_from_u64(seed));
    let mut c: Vec<Vec<f32>> = vec![vec![0.0; d]; k];
    let first = rng.random_range(0..n);
    c[0] = data[first].clone();
    for i in 1..k {
        let dist: Vec<f32> = (0..n)
            .map(|j| (0..i).map(|m| sq(&data[j], &c[m])).fold(f32::MAX, f32::min))
            .collect();
        let total: f32 = dist.iter().sum();
        if total == 0.0 {
            let idx = rng.random_range(0..n);
            c[i] = data[idx].clone();
            continue;
        }
        let threshold = rng.random::<f32>() * total;
        let mut cum = 0.0;
        let mut sel = 0;
        for (j, &dj) in dist.iter().enumerate() {
            cum += dj;
            if cum >= threshold {
                sel = j;
                break;
            }
        }
        c[i] = data[sel].clone();
    }
    let assign = |c: &Vec<Vec<f32>>| -> Vec<usize> {
        (0..n)
            .map(|i| {
                let mut best = 0;
                let mut bd = f32::MAX;
                for j in 0..k {
                    let dd = sq(&data[i], &c[j]).sqrt();
                    if dd < bd {
                        bd = dd;
                        best = j;
                    }
                }
                best
            })
            .collect()
    };
    let mut labels = assign(&c);
    for _ in 0..max_iter {
        labels = assign(&c);
        let mut nc = vec![vec![0.0f32; d]; k];
        let mut cnt = vec![0usize; k];
        for i in 0..n {
            for j in 0..d {
                nc[labels[i]][j] += data[i][j];
            }
            cnt[labels[i]] += 1;
        }
        for m in 0..k {
            if cnt[m] > 0 {
                for j in 0..d {
                    nc[m][j] /= cnt[m] as f32;
                }
            } else {
                let idx = rng.random_range(0..n);
                nc[m] = data[idx].clone();
            }
        }
        let shift = (0..k)
            .map(|m| sq(&c[m], &nc[m]).sqrt())
            .fold(0.0f32, f32::max);
        c = nc;
        if shift < tol {
            break;
        }
    }
    labels
}

/// kmeans_elkan.rs:208-243: inside the per-point loop the old label `a_i` is
/// captured once; after the point moves to a closer centre j1, later j2
/// checks recompute d_a against the *old* centre, reset upper to it and move
/// the point to j2 whenever d(x, j2) < d(x, old), even if j1 was closer.
#[cfg(feature = "cluster")]
#[test]
fn elkan_matches_lloyd_from_same_init() {
    use rand::prelude::*;
    use sheaf::cluster::{Clustering, KmeansElkan};
    let mut rng = StdRng::seed_from_u64(99);
    let mut data = Vec::new();
    for c in 0..10 {
        let (cx, cy) = ((c % 5) as f32 * 2.0, (c / 5) as f32 * 2.0);
        for _ in 0..40 {
            data.push(vec![
                cx + rng.random::<f32>() * 2.5,
                cy + rng.random::<f32>() * 2.5,
            ]);
        }
    }
    let mut mismatched = Vec::new();
    for seed in 0..40u64 {
        for max_iter in [2usize, 3, 5, 100] {
            let e = KmeansElkan::new(10)
                .with_seed(seed)
                .with_max_iter(max_iter)
                .fit_predict(&data)
                .unwrap();
            let l = lloyd_replica(&data, 10, seed, max_iter, 1e-4);
            let diff = e.iter().zip(&l).filter(|(a, b)| a != b).count();
            if diff > 0 {
                mismatched.push((seed, max_iter, diff));
            }
        }
    }
    assert!(
        mismatched.is_empty(),
        "(seed, max_iter, #labels differing): {mismatched:?}"
    );
}
