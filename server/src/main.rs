use kiroshi::image;
use kiroshi::model;
use kiroshi::server;
use kiroshi::{
    Detail, Error, Guidance, Lora, Model, Pag, Sampler, Size, Steps, Upscaler, protocol,
};

use futures::future;
use serde::Serialize;
use tokio::fs;
use tokio::io;
use tokio::net;
use tokio::process;
use tokio::signal::unix;
use tokio::task;

use std::ffi::OsStr;

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut interrupt = unix::signal(unix::SignalKind::interrupt())?;
    let mut terminate = unix::signal(unix::SignalKind::terminate())?;

    let generation = process::Command::new("uv")
        .arg("run")
        .arg("--no-sync")
        .arg("--frozen")
        .arg("--offline")
        .arg("--verbose")
        .arg("main.py")
        .spawn()
        .expect("Start generation server");

    let server = task::spawn(run());

    let _ = future::select(Box::pin(interrupt.recv()), Box::pin(terminate.recv())).await;
    server.abort();

    let _ = process::Command::new("kill")
        .arg(generation.id().unwrap_or_default().to_string())
        .status()
        .await?;

    Ok(())
}

async fn run() -> Result<(), Error> {
    let router = protocol::Strip::new()
        .plug(server::PING, ping)
        .plug(image::GENERATE, generate_image)
        .plug(image::DETAIL_FACES, detail_faces)
        .plug(image::DETAIL_HANDS, detail_hands)
        .plug(image::UPSCALE, upscale)
        .plug(model::LIST, list_models);

    let server = net::TcpListener::bind(&format!("0.0.0.0:{}", protocol::PORT)).await?;

    loop {
        let (client, _) = server.accept().await?;

        if let Err(error) = router.attach(client).await {
            log::error!("{error}");
        }
    }
}

async fn ping(mut client: protocol::Connection<bool, protocol::Never>) -> io::Result<()> {
    client.write(true).await
}

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

async fn generate_image(
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

    let mut generation =
        protocol::Connection::seize(net::TcpStream::connect("127.0.0.1:9148").await?);

    generation.write(request).await?;
    generation.copy(&mut client).await?;

    Ok(())
}

async fn detail_faces(
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

    let mut generation =
        protocol::Connection::seize(net::TcpStream::connect("127.0.0.1:9148").await?);

    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

async fn detail_hands(
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

    let mut generation =
        protocol::Connection::seize(net::TcpStream::connect("127.0.0.1:9148").await?);

    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

async fn upscale(
    mut client: protocol::Connection<image::upscale::Response, image::upscale::Request>,
) -> io::Result<()> {
    #[derive(Debug, Serialize)]
    struct Request {
        task: &'static str,
        upscaler: Upscaler,
        size: Size,
    }

    let image::upscale::Request { upscaler, size } = client.read().await?;

    let request = Request {
        task: "upscale",
        upscaler,
        size,
    };

    let mut generation =
        protocol::Connection::seize(net::TcpStream::connect("127.0.0.1:9148").await?);

    generation.write(request).await?;
    generation.connect(&mut client).await?;

    Ok(())
}

async fn list_models(
    mut client: protocol::Connection<Vec<Model>, protocol::Never>,
) -> io::Result<()> {
    let mut directory = fs::read_dir("/models").await?;
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
