//! Tree validation and health checking utilities.
//!
//! Provides tools to verify tree structure integrity and detect common issues:
//! - Orphaned nodes (no parent connection)
//! - Cycles in the tree
//! - Missing children references
//! - Inconsistent depth calculations
//!
//! # Example
//!
//! ```rust,ignore
//! use parti::tree::{RaptorTree, HealthCheck};
//!
//! let tree = RaptorTree::new(config);
//! // ... build tree ...
//!
//! let report = tree.health_check();
//! if !report.is_healthy() {
//!     for issue in report.issues {
//!         eprintln!("{}: {}", issue.severity, issue.message);
//!     }
//! }
//! ```

use std::collections::{HashMap, HashSet};

use super::RaptorTree;

/// Severity level for validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational, not a problem.
    Info,
    /// Something unusual but not necessarily wrong.
    Warning,
    /// A problem that should be fixed.
    Error,
    /// A critical issue that may cause failures.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A single validation issue found during health check.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue.
    pub severity: Severity,
    /// Human-readable description.
    pub message: String,
    /// Optional node ID involved.
    pub node_id: Option<usize>,
    /// Optional additional context.
    pub context: Option<String>,
}

impl ValidationIssue {
    /// Create a new validation issue.
    pub fn new(severity: Severity, message: impl Into<String>) -> Self {
        Self {
            severity,
            message: message.into(),
            node_id: None,
            context: None,
        }
    }

    /// Add a node ID to this issue.
    pub fn with_node(mut self, id: usize) -> Self {
        self.node_id = Some(id);
        self
    }

    /// Add context to this issue.
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.severity, self.message)?;
        if let Some(id) = self.node_id {
            write!(f, " (node {})", id)?;
        }
        if let Some(ctx) = &self.context {
            write!(f, " - {}", ctx)?;
        }
        Ok(())
    }
}

/// Report from a validation/health check.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Add an issue to the report.
    pub fn add(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    /// Add an info-level issue.
    pub fn info(&mut self, message: impl Into<String>) {
        self.add(ValidationIssue::new(Severity::Info, message));
    }

    /// Add a warning-level issue.
    pub fn warn(&mut self, message: impl Into<String>) {
        self.add(ValidationIssue::new(Severity::Warning, message));
    }

    /// Add an error-level issue.
    pub fn error(&mut self, message: impl Into<String>) {
        self.add(ValidationIssue::new(Severity::Error, message));
    }

    /// Add a critical-level issue.
    pub fn critical(&mut self, message: impl Into<String>) {
        self.add(ValidationIssue::new(Severity::Critical, message));
    }

    /// Check if the report contains no errors or critical issues.
    pub fn is_healthy(&self) -> bool {
        !self.issues.iter().any(|i| i.severity >= Severity::Error)
    }

    /// Check if there are any issues at all.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Get issues of a specific severity or higher.
    pub fn issues_at_level(&self, min_severity: Severity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity >= min_severity)
            .collect()
    }

    /// Count issues by severity.
    pub fn counts(&self) -> HashMap<Severity, usize> {
        let mut counts = HashMap::new();
        for issue in &self.issues {
            *counts.entry(issue.severity).or_default() += 1;
        }
        counts
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_clean() {
            return write!(f, "Validation passed: no issues found");
        }

        let counts = self.counts();
        write!(f, "Validation report: ")?;

        let parts: Vec<String> = [
            (Severity::Critical, "critical"),
            (Severity::Error, "errors"),
            (Severity::Warning, "warnings"),
            (Severity::Info, "info"),
        ]
        .iter()
        .filter_map(|(sev, name)| counts.get(sev).map(|c| format!("{} {}", c, name)))
        .collect();

        writeln!(f, "{}", parts.join(", "))?;

        for issue in &self.issues {
            writeln!(f, "  {}", issue)?;
        }

        Ok(())
    }
}

/// Health report with additional statistics.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Validation issues.
    pub validation: ValidationReport,
    /// Total number of nodes.
    pub node_count: usize,
    /// Number of leaf nodes.
    pub leaf_count: usize,
    /// Maximum depth of the tree.
    pub max_depth: usize,
    /// Average branching factor.
    pub avg_branching_factor: f64,
}

impl HealthReport {
    /// Check if the tree is healthy (no errors or critical issues).
    pub fn is_healthy(&self) -> bool {
        self.validation.is_healthy()
    }
}

impl std::fmt::Display for HealthReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tree Health Report")?;
        writeln!(f, "==================")?;
        writeln!(f, "Nodes: {} ({} leaves)", self.node_count, self.leaf_count)?;
        writeln!(f, "Max depth: {}", self.max_depth)?;
        writeln!(f, "Avg branching factor: {:.2}", self.avg_branching_factor)?;
        writeln!(f)?;
        write!(f, "{}", self.validation)
    }
}

/// Trait for types that can be health-checked.
pub trait HealthCheck {
    /// Perform a health check and return a report.
    fn health_check(&self) -> HealthReport;

    /// Quick check: returns true if healthy.
    fn is_healthy(&self) -> bool {
        self.health_check().is_healthy()
    }
}

impl<T, S> HealthCheck for RaptorTree<T, S> {
    fn health_check(&self) -> HealthReport {
        let node_count = self.len();

        // Build parent/child maps from the node adjacency.
        let mut parents: HashMap<usize, usize> = HashMap::new();
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();

        for node in self.iter() {
            if !node.children.is_empty() {
                let _ = children.insert(node.id, node.children.clone());
                for &child in &node.children {
                    // If a child has multiple parents, keep the first and report later.
                    let _ = parents.entry(child).or_insert(node.id);
                }
            }
        }

        let mut validation = validate_tree_structure(&parents, &children, node_count);

        // RAPTOR-specific structure checks.
        // - leaves should be level 0 and have no children
        // - internal nodes should have at least one child
        // - child level should be strictly less than parent level
        for node in self.iter() {
            if node.level == 0 {
                if !node.is_leaf() {
                    validation.add(
                        ValidationIssue::new(Severity::Error, "level 0 node is not a leaf")
                            .with_node(node.id),
                    );
                }
                if !node.children.is_empty() {
                    validation.add(
                        ValidationIssue::new(
                            Severity::Error,
                            "leaf node unexpectedly has children",
                        )
                        .with_node(node.id),
                    );
                }
            } else {
                if node.is_leaf() {
                    validation.add(
                        ValidationIssue::new(
                            Severity::Error,
                            "internal node is unexpectedly a leaf",
                        )
                        .with_node(node.id),
                    );
                }
                if node.children.is_empty() {
                    validation.add(
                        ValidationIssue::new(Severity::Warning, "internal node has no children")
                            .with_node(node.id),
                    );
                }
                for &child_id in &node.children {
                    if let Some(child) = self.get_node(child_id) {
                        if child.level >= node.level {
                            validation.add(
                                ValidationIssue::new(
                                    Severity::Error,
                                    "child level violation (child.level >= parent.level)",
                                )
                                .with_node(node.id)
                                .with_context(format!(
                                    "parent level {}, child {} level {}",
                                    node.level, child_id, child.level
                                )),
                            );
                        }
                    } else {
                        validation.add(
                            ValidationIssue::new(Severity::Error, "child id does not exist")
                                .with_node(node.id)
                                .with_context(format!("missing child id {child_id}")),
                        );
                    }
                }
            }
        }

        // Light heuristic: the number of nodes per level should generally decrease.
        for level in 1..self.depth() {
            let prev = self.get_level(level - 1).map(|v| v.len()).unwrap_or(0);
            let curr = self.get_level(level).map(|v| v.len()).unwrap_or(0);
            if curr > prev {
                validation.warn(format!(
                    "level {level} has more nodes than prior level ({curr} > {prev})"
                ));
            }
        }

        let leaf_count = self.leaves().len();
        let max_depth = self.depth().saturating_sub(1);
        let avg_branching_factor = if node_count == 0 {
            0.0
        } else {
            let total_children: usize = self.iter().map(|n| n.children.len()).sum();
            total_children as f64 / node_count as f64
        };

        HealthReport {
            validation,
            node_count,
            leaf_count,
            max_depth,
            avg_branching_factor,
        }
    }
}

/// Validate that a parent-child relationship forms a proper tree.
///
/// # Arguments
/// * `parents` - Map from node ID to parent ID (root has no entry)
/// * `children` - Map from node ID to child IDs
/// * `node_count` - Total number of nodes
///
/// # Returns
/// A validation report with any issues found.
pub fn validate_tree_structure(
    parents: &HashMap<usize, usize>,
    children: &HashMap<usize, Vec<usize>>,
    node_count: usize,
) -> ValidationReport {
    let mut report = ValidationReport::new();

    // Find root(s) - nodes without parents
    let all_nodes: HashSet<usize> = (0..node_count).collect();
    let nodes_with_parents: HashSet<usize> = parents.keys().copied().collect();
    let roots: Vec<usize> = all_nodes.difference(&nodes_with_parents).copied().collect();

    if roots.is_empty() {
        report.critical("No root node found - tree has cycles");
    } else if roots.len() > 1 {
        report.warn(format!("Multiple roots found: {:?}", roots));
    }

    // Check for orphaned nodes (not reachable from root)
    let mut reachable = HashSet::new();
    let mut stack = roots.clone();

    while let Some(node) = stack.pop() {
        if reachable.insert(node) {
            if let Some(node_children) = children.get(&node) {
                stack.extend(node_children);
            }
        }
    }

    let orphans: Vec<usize> = all_nodes.difference(&reachable).copied().collect();
    if !orphans.is_empty() {
        report.add(
            ValidationIssue::new(
                Severity::Error,
                format!("{} orphaned nodes not reachable from root", orphans.len()),
            )
            .with_context(format!("first few: {:?}", &orphans[..orphans.len().min(5)])),
        );
    }

    // Check parent-child consistency
    for (child, parent) in parents {
        if let Some(parent_children) = children.get(parent) {
            if !parent_children.contains(child) {
                report.add(
                    ValidationIssue::new(
                        Severity::Error,
                        "Parent-child inconsistency: child claims parent but parent doesn't list child",
                    )
                    .with_node(*child)
                    .with_context(format!("parent: {}", parent)),
                );
            }
        } else {
            report.add(
                ValidationIssue::new(Severity::Error, "Parent node has no children list")
                    .with_node(*parent),
            );
        }
    }

    // Check for cycles using DFS with coloring
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    fn detect_cycle(
        node: usize,
        children: &HashMap<usize, Vec<usize>>,
        visited: &mut HashSet<usize>,
        in_stack: &mut HashSet<usize>,
    ) -> bool {
        if in_stack.contains(&node) {
            return true; // Cycle detected
        }
        if visited.contains(&node) {
            return false;
        }

        let _ = visited.insert(node);
        let _ = in_stack.insert(node);

        if let Some(node_children) = children.get(&node) {
            for &child in node_children {
                if detect_cycle(child, children, visited, in_stack) {
                    return true;
                }
            }
        }

        let _ = in_stack.remove(&node);
        false
    }

    for &root in &roots {
        if detect_cycle(root, children, &mut visited, &mut in_stack) {
            report.critical("Cycle detected in tree structure");
            break;
        }
    }

    report
}

#[cfg(test)]
#[allow(clippy::unwrap_used, unused_results)]
mod tests {
    use super::*;
    use crate::hierarchy::{RaptorTree, TreeConfig};
    use proptest::prelude::*;

    fn chunk_cluster(ids: &[usize], fanout: usize) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut cur = Vec::new();
        for &id in ids {
            cur.push(id);
            if cur.len() == fanout {
                out.push(cur);
                cur = Vec::new();
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_validation_report_healthy() {
        let mut report = ValidationReport::new();
        report.info("Just some info");
        report.warn("A warning");

        assert!(report.is_healthy()); // No errors or critical

        report.error("An error");
        assert!(!report.is_healthy());
    }

    #[test]
    fn test_validation_issue_display() {
        let issue = ValidationIssue::new(Severity::Error, "Something wrong")
            .with_node(42)
            .with_context("additional info");

        let s = format!("{}", issue);
        assert!(s.contains("ERROR"));
        assert!(s.contains("Something wrong"));
        assert!(s.contains("42"));
        assert!(s.contains("additional info"));
    }

    #[test]
    fn test_validate_valid_tree() {
        // Simple tree: 0 -> [1, 2]
        let parents: HashMap<usize, usize> = [(1, 0), (2, 0)].into_iter().collect();
        let children: HashMap<usize, Vec<usize>> = [(0, vec![1, 2])].into_iter().collect();

        let report = validate_tree_structure(&parents, &children, 3);
        assert!(report.is_healthy());
    }

    #[test]
    fn test_validate_orphaned_nodes() {
        // Orphaned nodes are nodes that are not reachable from any root.
        //
        // Construct:
        // - root: 0 (no parent, no children)
        // - cycle component: 1 <-> 2 (both have parents, so they are not roots)
        //
        // Nodes {1,2} are not reachable from root {0} => orphaned.
        let parents: HashMap<usize, usize> = [(1, 2), (2, 1)].into_iter().collect();
        let children: HashMap<usize, Vec<usize>> =
            [(1, vec![2]), (2, vec![1])].into_iter().collect();

        let report = validate_tree_structure(&parents, &children, 3);
        assert!(!report.is_healthy());
        assert!(report.issues.iter().any(|i| i.message.contains("orphaned")));
    }

    #[test]
    fn test_validate_multiple_roots() {
        // Two separate trees: 0 -> [1], 2 -> [3]
        let parents: HashMap<usize, usize> = [(1, 0), (3, 2)].into_iter().collect();
        let children: HashMap<usize, Vec<usize>> =
            [(0, vec![1]), (2, vec![3])].into_iter().collect();

        let report = validate_tree_structure(&parents, &children, 4);
        // Multiple roots is a warning, not error
        assert!(report.is_healthy());
        assert!(report
            .issues
            .iter()
            .any(|i| i.message.contains("Multiple roots")));
    }

    #[test]
    fn raptor_health_check_is_healthy_for_basic_build() {
        let items = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let tree = RaptorTree::build(
            items,
            TreeConfig::default()
                .with_max_depth(3)
                .with_fanout(2)
                .with_min_cluster_size(2),
            chunk_cluster,
            |group| {
                group
                    .iter()
                    .map(|s| (*s).clone())
                    .collect::<Vec<_>>()
                    .join(" ")
            },
        )
        .unwrap();

        let report = tree.health_check();
        assert!(report.is_healthy(), "{}", report);
    }

    proptest! {
        #[test]
        fn raptor_health_check_is_healthy_for_chunk_cluster(
            items in proptest::collection::vec(".{1,40}", 1..80),
            fanout in 2usize..10,
            max_depth in 1usize..6,
        ) {
            let tree = RaptorTree::build(
                items,
                TreeConfig::default()
                    .with_max_depth(max_depth)
                    .with_fanout(fanout)
                    .with_min_cluster_size(2),
                chunk_cluster,
                |group| group.iter().map(|s| (*s).clone()).collect::<Vec<_>>().join(" "),
            ).unwrap();

            let report = tree.health_check();
            prop_assert!(report.is_healthy(), "{}", report);
        }
    }
}
