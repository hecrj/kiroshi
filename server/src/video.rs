use kiroshi::protocol;
use kiroshi::video;
use kiroshi::video::{Definition, Model};
use kiroshi::{Duration, Guidance, Size, Steps};

use serde::Serialize;
use tokio::fs;
use tokio::io;

use std::ffi::OsStr;

pub async fn list_models(
    mut client: protocol::Connection<Vec<Model>, protocol::Never>,
) -> io::Result<()> {
    let mut directory = fs::read_dir("/models/video").await?;
    let mut models = Vec::new();

    while let Some(entry) = directory.next_entry().await? {
        if !entry.metadata().await?.is_file() {
            continue;
        }

        if entry.path().extension().and_then(OsStr::to_str) != Some("safetensors") {
            continue;
        }

        models.push(Model {
            name: entry
                .path()
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        });
    }

    client.write(models).await
}

pub async fn generate(
    mut client: protocol::Connection<video::generate::Response, video::generate::Request>,
) -> io::Result<()> {
    #[derive(Serialize)]
    struct Request {
        task: &'static str,
        #[serde(flatten)]
        recipe: Recipe,
        preview_after: Option<f32>,
    }

    let video::generate::Request { definition } = client.read().await?;

    let request = Request {
        task: "generate_video",
        recipe: Recipe::from(definition),
        preview_after: None,
    };

    let mut generation = crate::connect().await?;
    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

#[derive(Debug, Serialize)]
struct Recipe {
    model: String,
    precision: String,
    seed: u64,
    prompt: String,
    negative_prompt: String,
    size: Size,
    max_area: u32,
    steps: Steps,
    guidance: Guidance,
    duration: Duration,
}

impl From<Definition> for Recipe {
    fn from(definition: Definition) -> Self {
        Self {
            model: definition.model.name,
            precision: definition.precision.to_string(),
            seed: definition.seed.value(),
            prompt: definition.prompt,
            negative_prompt: definition.negative_prompt,
            size: definition.size,
            max_area: definition.max_area,
            steps: definition.steps,
            guidance: definition.guidance,
            duration: definition.duration,
        }
    }
}
