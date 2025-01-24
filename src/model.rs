use crate::Error;

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tokio::fs;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Model(PathBuf);

impl Model {
    pub async fn list() -> Result<Vec<Self>, Error> {
        // TODO: Run on the server
        let mut models = Vec::new();
        let mut entries = fs::read_dir(directory()?).await?;

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

    pub fn name(&self) -> String {
        self.0
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Settings(BTreeMap<String, Metadata>);

impl Settings {
    pub async fn fetch() -> Result<Self, Error> {
        let config_file = dirs::config_dir()
            .ok_or(Error::ConfigDirectoryNotFound)?
            .join("kiroshi")
            .join("models.toml");

        let config = fs::read_to_string(config_file).await?;

        toml::from_str(&config)
            .map(Self)
            .map_err(Error::InvalidModelSettings)
    }

    pub fn get(&self, model: &Model) -> Metadata {
        self.0.get(&model.name()).cloned().unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Metadata {
    pub prompt_template: String,
    pub negative_prompt_template: String,
}

pub(crate) fn directory() -> Result<PathBuf, Error> {
    Ok(dirs::data_dir()
        .ok_or(Error::DataDirectoryNotFound)?
        .join("kiroshi")
        .join("models"))
}
