mod image;
mod video;

use kiroshi::Error;
use kiroshi::protocol;
use kiroshi::server;

use futures::future;
use tokio::io;
use tokio::net;
use tokio::process;
use tokio::signal::unix;
use tokio::task;

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
        .plug(kiroshi::image::LIST_MODELS, image::list_models)
        .plug(kiroshi::image::GENERATE, image::generate)
        .plug(kiroshi::image::DETAIL_FACES, image::detail_faces)
        .plug(kiroshi::image::DETAIL_HANDS, image::detail_hands)
        .plug(kiroshi::image::UPSCALE, image::upscale)
        .plug(kiroshi::video::LIST_MODELS, video::list_models)
        .plug(kiroshi::video::GENERATE, video::generate);

    let server = net::TcpListener::bind(&format!("0.0.0.0:{}", protocol::PORT)).await?;

    loop {
        let (client, _) = server.accept().await?;

        if let Err(error) = router.attach(client).await {
            log::error!("{error}");
        }
    }
}

async fn connect<I, O>() -> io::Result<protocol::Connection<I, O>> {
    Ok(protocol::Connection::seize(
        net::TcpStream::connect(format!("127.0.0.1:{}", protocol::PORT - 1)).await?,
    ))
}

async fn ping(mut client: protocol::Connection<bool, protocol::Never>) -> io::Result<()> {
    client.write(true).await
}
