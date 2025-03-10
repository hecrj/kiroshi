use crate::server;
use crate::Error;

use serde::{Deserialize, Serialize};

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Model(String);

impl Model {
    pub async fn list() -> Result<Vec<Self>, Error> {
        let mut stream = server::connect().await?;

        #[derive(Serialize)]
        struct Request {
            task: &'static str,
        }

        #[derive(Deserialize)]
        struct Response {
            models: Vec<String>,
        }

        server::send_json(
            &mut stream,
            Request {
                task: "list_models",
            },
        )
        .await?;

        let mut buffer = Vec::new();
        let Response { models } = server::read_json(&mut stream, &mut buffer).await?;

        Ok(models.into_iter().map(Self).collect())
    }

    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}
