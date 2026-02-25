use core::fmt;

/// Result alias for `parti`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by clustering and hierarchy primitives.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Input was empty.
    EmptyInput,

    /// Matrix dimension mismatch (usize).
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Found dimension.
        found: usize,
    },

    /// Shape mismatch (string description).
    ShapeMismatch {
        /// Expected shape description.
        expected: String,
        /// Actual shape description.
        actual: String,
    },

    /// Matrix inversion failure.
    InversionFailed,

    /// Invalid number of clusters requested.
    InvalidClusterCount {
        /// Requested count.
        requested: usize,
        /// Number of items.
        n_items: usize,
    },
    /// Clustering did not converge within iteration limit.
    ConvergenceFailure {
        /// Number of iterations attempted.
        iterations: usize,
    },
    /// Invalid parameter value.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Error message.
        message: &'static str,
    },
    /// Graph is disconnected where connected was required.
    DisconnectedGraph,
    /// Generic error with message.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyInput => write!(f, "empty input provided"),
            Error::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }
            Error::ShapeMismatch { expected, actual } => {
                write!(f, "shape mismatch: expected {expected}, actual {actual}")
            }
            Error::InversionFailed => write!(f, "matrix inversion failed"),
            Error::InvalidClusterCount { requested, n_items } => {
                write!(f, "cannot create {requested} clusters from {n_items} items")
            }
            Error::ConvergenceFailure { iterations } => {
                write!(f, "did not converge after {iterations} iterations")
            }
            Error::InvalidParameter { name, message } => {
                write!(f, "invalid parameter '{name}': {message}")
            }
            Error::DisconnectedGraph => write!(f, "graph is disconnected"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
