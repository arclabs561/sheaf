//! Small example: hierarchical conformal prediction over a toy tree.

use faer::Mat;
use parti::{HierarchicalConformal, HierarchyTree, ReconciliationMethod};
use parti::{RaptorTree, TreeConfig};

fn main() -> parti::Result<()> {
    // 1. Create a simple hierarchy (e.g., from RAPTOR)
    let chunks = vec![
        "The cat sat on the mat.".to_string(),
        "The dog barked at the cat.".to_string(),
        "The bird sang in the tree.".to_string(),
    ];

    // For this example, we'll build a tree with 1 root and 3 leaves
    let tree = RaptorTree::build(
        chunks,
        TreeConfig::new().with_max_depth(1).with_fanout(3),
        |_ids, _fanout| vec![vec![0, 1, 2]], // All in one cluster
        |group| format!("Summary of {} items", group.len()),
    )?;

    // 2. Convert to HierarchyTree for conformal prediction
    let h_tree = HierarchyTree::from_raptor(&tree);
    let s = h_tree.summing_matrix();

    // 3. Setup Conformal Predictor
    let mut cp = HierarchicalConformal::new(s, ReconciliationMethod::Ols);

    // 4. Calibrate (using synthetic data for demonstration)
    let m = h_tree.len(); // 1 root + 3 leaves = 4
    let n_calib = 20;
    let y_calib = Mat::<f64>::zeros(m, n_calib);
    let mut y_hat_calib = Mat::<f64>::zeros(m, n_calib);
    // Add some random noise/error
    for j in 0..n_calib {
        for i in 0..m {
            y_hat_calib[(i, j)] = (i as f64 + 1.0) * 0.1;
        }
    }

    cp.calibrate(&y_calib, &y_hat_calib, 0.1)?;
    println!("Calibrated quantile: {}", cp.quantile());

    // 5. Predict with Intervals
    let mut y_hat_test = Mat::<f64>::zeros(m, 1);
    y_hat_test[(0, 0)] = 3.0; // root
    y_hat_test[(1, 0)] = 1.0; // leaf 1
    y_hat_test[(2, 0)] = 1.0; // leaf 2
    y_hat_test[(3, 0)] = 1.0; // leaf 3

    let (lower, upper) = cp.predict_intervals(&y_hat_test)?;

    println!("Root interval: [{}, {}]", lower[(0, 0)], upper[(0, 0)]);
    Ok(())
}
