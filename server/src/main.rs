use kiroshi::image;
use kiroshi::model;
use kiroshi::server;
use kiroshi::{Detail, Error, Inpaint, Lora, Model, Sampler, Size, Steps, Upscaler, protocol};

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

    let generation = process::Command::new(".env/bin/python")
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

async fn generate_image(
    mut client: protocol::Connection<image::generate::Response, image::generate::Request>,
) -> io::Result<()> {
    #[derive(Serialize)]
    struct Request {
        model: String,
        prompt: String,
        negative_prompt: String,
        size: Size,
        quality: String,
        sampler: String,
        upscaler: Option<Upscaler>,
        steps: Steps,
        seed: u64,
        face_detail: Option<Detail>,
        hand_detail: Option<Detail>,
        inpaints: Vec<Inpaint>,
        loras: Vec<Lora>,
        preview_after: Option<f32>,
    }

    let image::generate::Request {
        definition,
        preview_after,
    } = client.read().await?;

    let request = Request {
        model: definition.model.name.clone(),
        prompt: definition.prompt.clone(),
        negative_prompt: definition.negative_prompt.clone(),
        size: definition.size,
        quality: definition.quality.to_string().to_lowercase(),
        sampler: match definition.sampler {
            Sampler::EulerAncestral => "euler_a",
            Sampler::DPMSDEKarras => "dpm++_sde_karras",
            Sampler::DPM2MKarras => "dpm++_2m_karras",
            Sampler::DPM2MSDEKarras => "dpm++_2m_sde_karras",
        }
        .to_owned(),
        upscaler: definition.upscaler,
        steps: definition.steps,
        seed: definition.seed.value(),
        face_detail: definition.face_detail,
        hand_detail: definition.hand_detail,
        inpaints: definition.inpaints.clone(),
        loras: definition.loras.clone(),
        preview_after,
    };

    let mut generation =
        protocol::Connection::seize(net::TcpStream::connect("127.0.0.1:9148").await?);

    generation.write(request).await?;
    generation.copy(&mut client).await?;

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
