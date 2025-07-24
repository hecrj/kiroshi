use crate::Error;
use crate::protocol;

use serde::{Deserialize, Serialize};

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
}

impl Model {
    pub async fn list() -> Result<Vec<Self>, Error> {
        let mut stream = protocol::connect(LIST).await?;

        Ok(stream.read().await?)
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

pub const LIST: protocol::Plug<protocol::Never, Vec<Model>> = protocol::Plug::new("list_models");
