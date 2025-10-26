use kiroshi::image::{self, Model};
use kiroshi::{Detail, Guidance, Lora, Pag, Sampler, Size, Steps, Upscaler, protocol};

use serde::Serialize;
use tokio::fs;

use std::ffi::OsStr;
use std::io;

#[derive(Debug, Serialize)]
struct Recipe {
    model: String,
    precision: String,
    prompt: String,
    negative_prompt: String,
    size: Size,
    sampler: String,
    steps: Steps,
    guidance: Guidance,
    pag: Option<Pag>,
    seed: u64,
    loras: Vec<Lora>,
}

impl From<image::Definition> for Recipe {
    fn from(definition: image::Definition) -> Self {
        Self {
            model: definition.model.name.clone(),
            precision: match definition.precision {
                kiroshi::Precision::Float16 => "float16",
                kiroshi::Precision::BFloat16 => "bfloat16",
                kiroshi::Precision::Float32 => "float32",
            }
            .to_owned(),
            prompt: definition.prompt.clone(),
            negative_prompt: definition.negative_prompt.clone(),
            size: definition.size,
            sampler: match definition.sampler {
                Sampler::EulerAncestral => "euler_a",
                Sampler::DPMSDEKarras => "dpm++_sde_karras",
                Sampler::DPM2MKarras => "dpm++_2m_karras",
                Sampler::DPM2MSDEKarras => "dpm++_2m_sde_karras",
            }
            .to_owned(),
            steps: definition.steps,
            guidance: definition.guidance,
            pag: definition.pag,
            seed: definition.seed.value(),
            loras: definition.loras.clone(),
        }
    }
}

pub async fn generate(
    mut client: protocol::Connection<image::generate::Response, image::generate::Request>,
) -> io::Result<()> {
    #[derive(Serialize)]
    struct Request {
        task: &'static str,
        #[serde(flatten)]
        recipe: Recipe,
        preview_after: Option<f32>,
    }

    let image::generate::Request { definition } = client.read().await?;

    let request = Request {
        task: "generate_image",
        recipe: Recipe::from(definition),
        preview_after: Some(0.0),
    };

    let mut generation = crate::connect().await?;
    generation.write(request).await?;
    generation.copy(&mut client).await?;

    Ok(())
}

pub async fn detail_faces(
    mut client: protocol::Connection<image::detail_faces::Response, image::detail_faces::Request>,
) -> io::Result<()> {
    #[derive(Debug, Serialize)]
    struct Request {
        task: &'static str,
        #[serde(flatten)]
        recipe: Recipe,
        detail: Detail,
        preview_after: Option<f32>,
    }

    let image::detail_faces::Request { definition, detail } = client.read().await?;

    let request = Request {
        task: "detail_faces",
        recipe: Recipe::from(definition),
        detail,
        preview_after: Some(0.0),
    };

    let mut generation = crate::connect().await?;
    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

pub async fn detail_hands(
    mut client: protocol::Connection<image::detail_hands::Response, image::detail_hands::Request>,
) -> io::Result<()> {
    #[derive(Debug, Serialize)]
    struct Request {
        task: &'static str,
        #[serde(flatten)]
        recipe: Recipe,
        detail: Detail,
        preview_after: Option<f32>,
    }

    let image::detail_hands::Request { definition, detail } = client.read().await?;

    let request = Request {
        task: "detail_hands",
        recipe: Recipe::from(definition),
        detail,
        preview_after: Some(0.0),
    };

    let mut generation = crate::connect().await?;
    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

pub async fn upscale(
    mut client: protocol::Connection<image::upscale::Response, image::upscale::Request>,
) -> io::Result<()> {
    #[derive(Debug, Serialize)]
    struct Request {
        task: &'static str,
        upscaler: Upscaler,
        size: Size,
        cache: bool,
    }

    let image::upscale::Request {
        upscaler,
        size,
        cache,
    } = client.read().await?;

    let request = Request {
        task: "upscale",
        upscaler,
        size,
        cache: cache == image::Cache::Enabled,
    };

    let mut generation = crate::connect().await?;
    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

pub async fn list_models(
    mut client: protocol::Connection<Vec<Model>, protocol::Never>,
) -> io::Result<()> {
    let mut directory = fs::read_dir("/models/image").await?;
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
