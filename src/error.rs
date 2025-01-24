use std::io;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Error {
    IOFailed(Arc<io::Error>),
    SerializationFailed(Arc<serde_json::Error>),
    DockerFailed,
    DataDirectoryNotFound,
    ConfigDirectoryNotFound,
    InvalidModelSettings(toml::de::Error),
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
