use crate::Error;
use crate::protocol;

use serde::{Deserialize, Serialize};

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Model(String);

impl Model {
    pub async fn list() -> Result<Vec<Self>, Error> {
        let mut stream = protocol::perform(protocol::Request::ListModels).await?;

        #[derive(Deserialize)]
        struct Response {
            models: Vec<String>,
        }

        let mut buffer = Vec::new();
        let Response { models } = protocol::read_json(&mut stream, &mut buffer).await?;

        Ok(models.into_iter().map(Self).collect())
    }

    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
