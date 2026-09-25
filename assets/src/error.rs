//! Diagnostic helpers shared by every compile stage.

use std::fmt;

/// A compile-stage error tied to a specific asset path.
#[derive(Debug, Clone)]
pub struct AssetError {
    pub path: String,
    pub message: String,
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for AssetError {}

pub type AssetResult<T> = Result<T, AssetError>;

pub fn err<T>(path: &str, message: impl Into<String>) -> AssetResult<T> {
    Err(AssetError {
        path: path.to_string(),
        message: message.into(),
    })
}
