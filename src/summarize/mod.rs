//! Summarization strategies for hierarchical abstraction.
//!
//! This module provides traits and utilities for summarizing groups
//! of items in a hierarchical tree.
//!
//! The actual summarization logic (e.g., LLM calls) is provided by the user
//! via closures or trait implementations, keeping this crate lightweight.

/// Trait for summarization strategies.
///
/// Implementors define how a group of items is condensed into a summary.
pub trait Summarizer<T, S = T> {
    /// Summarize a group of items.
    fn summarize(&self, items: &[&T]) -> S;
}

/// A simple concatenation summarizer (for testing).
#[derive(Debug, Clone, Default)]
pub struct ConcatSummarizer {
    /// Separator between items.
    pub separator: String,
    /// Maximum length in bytes (truncate if exceeded). Truncation stops at a
    /// UTF-8 character boundary, so the result can be a few bytes shorter.
    pub max_len: Option<usize>,
}

impl ConcatSummarizer {
    /// Create a new concatenation summarizer.
    pub fn new() -> Self {
        Self {
            separator: " | ".to_string(),
            max_len: None,
        }
    }

    /// Set separator.
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Set maximum length.
    pub fn with_max_len(mut self, len: usize) -> Self {
        self.max_len = Some(len);
        self
    }
}

impl Summarizer<String> for ConcatSummarizer {
    fn summarize(&self, items: &[&String]) -> String {
        let joined = items
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(&self.separator);

        match self.max_len {
            Some(max) if joined.len() > max => {
                let mut end = max.saturating_sub(3);
                while !joined.is_char_boundary(end) {
                    end -= 1;
                }
                let mut truncated = joined[..end].to_string();
                truncated.push_str("...");
                truncated
            }
            _ => joined,
        }
    }
}

/// A function-based summarizer.
#[derive(Clone)]
pub struct FnSummarizer<F> {
    f: F,
}

impl<F> FnSummarizer<F> {
    /// Create a summarizer from a function.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<T, S, F> Summarizer<T, S> for FnSummarizer<F>
where
    F: Fn(&[&T]) -> S,
{
    fn summarize(&self, items: &[&T]) -> S {
        (self.f)(items)
    }
}

/// Create a summarizer from a closure.
pub fn from_fn<T, S, F>(f: F) -> FnSummarizer<F>
where
    F: Fn(&[&T]) -> S,
{
    FnSummarizer::new(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_summarizer() {
        let summarizer = ConcatSummarizer::new().with_separator(", ");

        let items = ["a".to_string(), "b".to_string(), "c".to_string()];
        let refs: Vec<&String> = items.iter().collect();

        let summary = summarizer.summarize(&refs);
        assert_eq!(summary, "a, b, c");
    }

    #[test]
    fn test_fn_summarizer() {
        let summarizer = from_fn(|items: &[&i32]| items.iter().copied().sum::<i32>());

        let items = [1, 2, 3];
        let refs: Vec<&i32> = items.iter().collect();

        let summary: i32 = summarizer.summarize(&refs);
        assert_eq!(summary, 6);
    }
}
