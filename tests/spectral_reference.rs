//! Spectral clustering against known partitions and scikit-learn 1.7.2.
//!
//! Data in tests/spectral_data/*.csv: x, y, ground truth, scikit-learn
//! `SpectralClustering` label. The points come from `make_blobs`,
//! `make_moons` and `make_circles` with fixed seeds; scikit-learn reaches
//! ARI 1.000 against the truth on every file.
#![cfg(feature = "spectral")]
#![allow(clippy::unwrap_used)]

use ndarray::Array2;
use sheaf::cluster::spectral::{AffinityType, SpectralClustering};

fn ari(a: &[usize], b: &[usize]) -> f64 {
    let n = a.len();
    let ka = a.iter().max().unwrap() + 1;
    let kb = b.iter().max().unwrap() + 1;
    let mut table = vec![vec![0u64; kb]; ka];
    for (&x, &y) in a.iter().zip(b) {
        table[x][y] += 1;
    }
    let c2 = |m: u64| (m * m.saturating_sub(1)) as f64 / 2.0;
    let sum_ij: f64 = table.iter().flatten().map(|&m| c2(m)).sum();
    let sum_a: f64 = table.iter().map(|r| c2(r.iter().sum())).sum();
    let sum_b: f64 = (0..kb).map(|j| c2(table.iter().map(|r| r[j]).sum())).sum();
    let expected = sum_a * sum_b / c2(n as u64);
    let max = 0.5 * (sum_a + sum_b);
    if (max - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_ij - expected) / (max - expected)
}

fn load(name: &str) -> (Array2<f64>, Vec<usize>, Vec<usize>) {
    let path = format!(
        "{}/tests/spectral_data/{name}.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).unwrap();
    let mut xs = Vec::new();
    let mut truth = Vec::new();
    let mut sk = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split(',').collect();
        xs.push(f[0].parse::<f64>().unwrap());
        xs.push(f[1].parse::<f64>().unwrap());
        truth.push(f[2].parse().unwrap());
        sk.push(f[3].parse().unwrap());
    }
    let n = truth.len();
    (Array2::from_shape_vec((n, 2), xs).unwrap(), truth, sk)
}

/// Two parallel chains of `m` points, 0.5 apart along each chain and 50
/// apart between chains: with an RBF kernel of sigma 1 the affinity graph has
/// exactly two connected components (cross-chain weights underflow to 0).
fn two_chains(m: usize) -> (Array2<f64>, Vec<usize>) {
    let mut pts = Array2::zeros((2 * m, 2));
    let mut truth = vec![0; 2 * m];
    for i in 0..m {
        pts[[i, 0]] = 0.5 * i as f64;
        pts[[m + i, 0]] = 0.5 * i as f64;
        pts[[m + i, 1]] = 50.0;
        truth[m + i] = 1;
    }
    (pts, truth)
}

fn check(name: &str, labels: &[usize], truth: &[usize], sk: Option<&[usize]>) {
    let a = ari(labels, truth);
    if let Some(sk) = sk {
        eprintln!("{name}: ARI vs scikit-learn = {:.3}", ari(labels, sk));
    }
    assert!(
        a > 0.95,
        "{name}: ARI(sheaf, truth) = {a:.3}; scikit-learn reaches 1.000"
    );
}

/// n = 120 is above lapl's default Jacobi threshold (64).
#[test]
fn spectral_two_chains_n120() {
    let (pts, truth) = two_chains(60);
    let labels = SpectralClustering::new(2).sigma(1.0).fit(&pts).unwrap();
    check("two_chains_n120", &labels, &truth, None);
}

/// n = 600 is above lapl's `faer` dense threshold (512) as well.
#[test]
fn spectral_two_chains_n600() {
    let (pts, truth) = two_chains(300);
    let labels = SpectralClustering::new(2).sigma(1.0).fit(&pts).unwrap();
    check("two_chains_n600", &labels, &truth, None);
}

#[test]
fn spectral_blobs3_rbf_matches_sklearn() {
    let (x, truth, sk) = load("blobs3");
    let labels = SpectralClustering::new(3).sigma(1.0).fit(&x).unwrap();
    check("blobs3 (n=150, rbf)", &labels, &truth, Some(&sk));
}

#[test]
fn spectral_blobs3_n600_rbf_matches_sklearn() {
    let (x, truth, sk) = load("blobs3_n600");
    let labels = SpectralClustering::new(3).sigma(1.0).fit(&x).unwrap();
    check("blobs3_n600 (rbf)", &labels, &truth, Some(&sk));
}

#[test]
fn spectral_moons_knn_matches_sklearn() {
    let (x, truth, sk) = load("moons");
    let labels = SpectralClustering::new(2)
        .affinity(AffinityType::Knn)
        .n_neighbors(10)
        .fit(&x)
        .unwrap();
    check("moons (n=200, knn 10)", &labels, &truth, Some(&sk));
}

#[test]
fn spectral_circles_knn_matches_sklearn() {
    let (x, truth, sk) = load("circles");
    let labels = SpectralClustering::new(2)
        .affinity(AffinityType::Knn)
        .n_neighbors(10)
        .fit(&x)
        .unwrap();
    check("circles (n=200, knn 10)", &labels, &truth, Some(&sk));
}

#[test]
fn spectral_moons_n600_knn_matches_sklearn() {
    let (x, truth, sk) = load("moons_n600");
    let labels = SpectralClustering::new(2)
        .affinity(AffinityType::Knn)
        .n_neighbors(10)
        .fit(&x)
        .unwrap();
    check("moons_n600 (knn 10)", &labels, &truth, Some(&sk));
}

/// `fit_affinity` takes the same eigensolver path as `fit`.
#[test]
fn spectral_fit_affinity_two_components_n600() {
    let (pts, truth) = two_chains(300);
    let aff = lapl::gaussian_similarity(&pts, 1.0);
    let labels = SpectralClustering::new(2)
        .affinity(AffinityType::Precomputed)
        .fit_affinity(&aff)
        .unwrap();
    check("fit_affinity two_chains_n600", &labels, &truth, None);
}
