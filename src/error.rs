use std::io;
use std::sync::Arc;

#[derive(Debug, Clone, thiserror::Error)]
pub enum Error {
    #[error("io operation failed: {0}")]
    IOFailed(Arc<io::Error>),
    #[error("serialization failed: {0}")]
    SerializationFailed(Arc<serde_json::Error>),
    #[error("docker operation failed")]
    DockerFailed,
    #[error("invalid output: {0}")]
    InvalidOutput(String),
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::IOFailed(Arc::new(error))
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::SerializationFailed(Arc::new(error))
    }
}
