use crate::Error;

use std::fmt;
use std::path::{Path, PathBuf};

use tokio::fs;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Model(PathBuf);

impl Model {
    pub async fn list(directory: impl AsRef<Path>) -> Result<Vec<Self>, Error> {
        let mut models = Vec::new();
        let mut entries = fs::read_dir(directory).await?;

        while let Some(entry) = entries.next_entry().await? {
            if !entry.file_type().await?.is_file() {
                continue;
            }

            let extension = entry
                .path()
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();

            if extension != "safetensors" {
                continue;
            }

            models.push(Self(entry.path()));
        }

        Ok(models)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.file_name().unwrap_or_default().to_string_lossy())
    }
}
