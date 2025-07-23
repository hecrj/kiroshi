use kiroshi::{Detail, Error, Inpaint, Lora, Sampler, Size, Steps, Upscaler, protocol};

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
    let server = net::TcpListener::bind(&format!("0.0.0.0:{}", protocol::PORT)).await?;
    let mut buffer = Vec::new();

    loop {
        let (mut client, _) = server.accept().await?;

        let Ok(request) = protocol::read_json(&mut client, &mut buffer).await else {
            continue;
        };

        match dbg!(request) {
            protocol::Request::Ping => {
                #[derive(Serialize)]
                struct Pong(bool);

                protocol::send_json(&mut client, Pong(true)).await?;
            }
            protocol::Request::ListModels => {
                let mut directory = fs::read_dir("/models").await?;
                let mut models = Vec::new();

                while let Some(entry) = directory.next_entry().await? {
                    if !entry.metadata().await?.is_file() {
                        continue;
                    }

                    if entry.path().extension().and_then(OsStr::to_str) != Some("safetensors") {
                        continue;
                    }

                    models.push(
                        entry
                            .path()
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned(),
                    );
                }

                #[derive(Serialize)]
                struct Response {
                    models: Vec<String>,
                }

                protocol::send_json(&mut client, Response { models }).await?;
            }
            protocol::Request::GenerateImage {
                definition,
                preview_after,
            } => {
                #[derive(Serialize)]
                struct Request {
                    task: &'static str,
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

                let request = Request {
                    task: "generate_image",
                    model: definition.model.name().to_owned(),
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

                let mut generation = net::TcpStream::connect("127.0.0.1:9148").await?;
                protocol::send_json(&mut generation, request).await?;

                io::copy(&mut generation, &mut client).await?;
            }
        }
    }
}
