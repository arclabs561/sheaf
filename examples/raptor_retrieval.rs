//! RAPTOR-style hierarchical retrieval over synthetic documents.
//!
//! Builds a [`RaptorTree`] from 20 short documents (4 topics), represented
//! as bag-of-words vectors.  Clusters by greedy cosine grouping; summaries
//! are centroid vectors with concatenated text.  Searches at leaf vs. root
//! level to demonstrate narrow vs. broad retrieval.
//!
//! Run: `cargo run -p sheaf --example raptor_retrieval`

use sheaf::{RaptorTree, TreeConfig};

type Doc = (String, Vec<f32>);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = 15; // 4 topic groups: compilers(0-3), net(4-7), db(8-11), os(12-14)
    let docs: Vec<Doc> = vec![
        doc(
            "Recursive descent parsers build an AST directly.",
            dim,
            &[1, 2],
        ),
        doc(
            "Constant folding is a key compiler optimization.",
            dim,
            &[0, 3],
        ),
        doc(
            "LR parsers use tables generated from a grammar.",
            dim,
            &[1, 0],
        ),
        doc(
            "Loop unrolling reduces branch overhead in compilers.",
            dim,
            &[0, 3],
        ),
        doc(
            "Abstract syntax trees represent program structure.",
            dim,
            &[2, 0],
        ),
        doc(
            "TCP uses a three-way handshake to open connections.",
            dim,
            &[4, 7],
        ),
        doc(
            "Packet loss increases tail latency in data centers.",
            dim,
            &[5, 6],
        ),
        doc("Non-blocking sockets allow multiplexed I/O.", dim, &[7, 6]),
        doc("TCP congestion control adapts to bandwidth.", dim, &[4, 5]),
        doc(
            "Socket read timeouts prevent indefinite blocking.",
            dim,
            &[7, 6],
        ),
        doc("B-tree indexes speed up range queries.", dim, &[8, 9]),
        doc(
            "MVCC enables snapshot isolation for transactions.",
            dim,
            &[10, 9],
        ),
        doc(
            "Schema migrations must preserve compatibility.",
            dim,
            &[11, 10],
        ),
        doc(
            "Covering indexes avoid heap lookups for queries.",
            dim,
            &[8, 9],
        ),
        doc(
            "Two-phase commit coordinates distributed txns.",
            dim,
            &[10, 8],
        ),
        doc(
            "The kernel scheduler assigns threads to cores.",
            dim,
            &[12, 14],
        ),
        doc(
            "Syscalls transition from user to kernel space.",
            dim,
            &[13, 12],
        ),
        doc(
            "Priority inversion causes deadline misses in RTOS.",
            dim,
            &[14, 12],
        ),
        doc(
            "CFS scheduler uses a red-black tree of tasks.",
            dim,
            &[14, 12],
        ),
        doc(
            "Syscall overhead matters for high-frequency I/O.",
            dim,
            &[13, 6],
        ),
    ];
    let vecs: Vec<Vec<f32>> = docs.iter().map(|(_, v)| v.clone()).collect();

    println!("Building RAPTOR tree: {} docs, dim={}.", docs.len(), dim);
    let config = TreeConfig::new()
        .with_max_depth(3)
        .with_fanout(5)
        .with_min_cluster_size(2);
    let tree = RaptorTree::<Doc, Doc>::build(
        docs,
        config,
        |ids, fan| greedy_cluster(ids, fan, &vecs),
        |items: &[&Doc]| {
            let text = items
                .iter()
                .map(|(t, _)| t.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            let mut c = vec![0.0f32; dim];
            for (_, v) in items.iter() {
                for (i, &x) in v.iter().enumerate() {
                    c[i] += x;
                }
            }
            let n = items.len() as f32;
            c.iter_mut().for_each(|x| *x /= n);
            (text, c)
        },
    )?;

    println!(
        "Depth: {} levels, {} nodes total.",
        tree.depth(),
        tree.len()
    );
    for lv in 0..tree.depth() {
        println!(
            "  Level {}: {} nodes",
            lv,
            tree.get_level(lv).unwrap_or_default().len()
        );
    }
    println!();

    // Queries: (text, term-index weights)
    let queries: &[(&str, &[(usize, f32)])] = &[
        ("compiler optimization", &[(0, 1.0), (3, 1.0)]),
        ("network latency", &[(6, 1.0)]),
        ("database transactions", &[(9, 1.0), (10, 1.0)]),
        ("kernel scheduling", &[(12, 1.0), (14, 1.0)]),
    ];
    for &(qtext, qterms) in queries {
        let qv = bow(dim, qterms);
        println!("Query: \"{}\"", qtext);
        for lv in 0..tree.depth() {
            let mut scored: Vec<_> = tree
                .get_level(lv)
                .unwrap_or_default()
                .iter()
                .filter_map(|n| {
                    let (t, v) = n.as_leaf().or_else(|| n.as_summary())?;
                    Some((cosine(&qv, v), n.id, t.as_str()))
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(3);
            let tag = if lv == 0 {
                "leaf"
            } else if lv == tree.depth() - 1 {
                "root"
            } else {
                "mid"
            };
            println!("  Level {} ({}):", lv, tag);
            for (r, &(sim, id, text)) in scored.iter().enumerate() {
                let t = if text.len() > 60 {
                    format!("{}...", &text[..57])
                } else {
                    text.to_string()
                };
                println!("    #{}: id={:<3} sim={:.3}  {}", r + 1, id, sim, t);
            }
        }
        println!();
    }
    Ok(())
}

fn bow(dim: usize, terms: &[(usize, f32)]) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for &(i, w) in terms {
        if i < dim {
            v[i] = w;
        }
    }
    v
}

fn doc(text: &str, dim: usize, ids: &[usize]) -> Doc {
    (
        text.into(),
        bow(dim, &ids.iter().map(|&i| (i, 1.0)).collect::<Vec<_>>()),
    )
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum();
    let na: f64 = a.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let nb: f64 = b.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let d = na.sqrt() * nb.sqrt();
    if d < 1e-12 {
        0.0
    } else {
        dot / d
    }
}

/// Greedy cosine-similarity clustering.  Assigns each node to the best
/// existing cluster (or starts a new one below a similarity threshold).
fn greedy_cluster(ids: &[usize], fanout: usize, vecs: &[Vec<f32>]) -> Vec<Vec<usize>> {
    if ids.is_empty() {
        return vec![];
    }
    let _dim = vecs.first().map_or(0, Vec::len);
    let cap = ids.len().div_ceil(fanout).max(1) * 2;
    let mut cls: Vec<(Vec<f32>, usize, Vec<usize>)> = Vec::new(); // (sum, count, members)

    for &nid in ids {
        let v = match vecs.get(nid) {
            Some(v) => v,
            None => continue,
        };
        let (best_i, best_s) = cls
            .iter()
            .enumerate()
            .filter(|(_, (_, c, _))| *c > 0)
            .map(|(i, (s, c, _))| {
                let cent: Vec<f32> = s.iter().map(|&x| x / *c as f32).collect();
                (i, cosine(v, &cent))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, -1.0));

        let join = best_s > 0.15 && cls.get(best_i).is_some_and(|c| c.2.len() < fanout * 2);
        if join {
            for (i, &x) in v.iter().enumerate() {
                cls[best_i].0[i] += x;
            }
            cls[best_i].1 += 1;
            cls[best_i].2.push(nid);
        } else if cls.len() < cap {
            cls.push((v.clone(), 1, vec![nid]));
        } else if let Some(c) = cls.get_mut(best_i) {
            for (i, &x) in v.iter().enumerate() {
                c.0[i] += x;
            }
            c.1 += 1;
            c.2.push(nid);
        }
    }
    cls.into_iter()
        .map(|(_, _, m)| m)
        .filter(|m| !m.is_empty())
        .collect()
}
