use crate::Error;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::io;
use tokio::net;
use tokio::process;
use tokio::time;

use std::path::Path;
use std::sync::Arc;

const ADDRESS: &str = "127.0.0.1:9149";

#[derive(Debug, Clone)]
pub struct Server {
    _container: Arc<Container>,
}

#[derive(Debug)]
struct Container(String);

impl Server {
    pub async fn run(models_dir: impl AsRef<Path>) -> Result<Server, Error> {
        let models = {
            let models_dir = models_dir.as_ref();
            fs::create_dir_all(&models_dir).await?;

            format!("{host}:/models", host = models_dir.to_string_lossy())
        };

        let mut process = process::Command::new("docker")
            .arg("create")
            .args(["-t", "--rm"])
            .args(["--gpus", "all"])
            .args(["-p", "9149:9149"])
            .args(["-v", &models])
            .arg("ghcr.io/hecrj/kiroshi/server:latest")
            .stdout(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        let container = {
            use io::AsyncBufReadExt;

            let output = io::BufReader::new(process.stdout.take().expect("piped stdout"));

            let mut lines = output.lines();

            lines.next_line().await?.ok_or(Error::DockerFailed)?
        };

        let _start = process::Command::new("docker")
            .args(["start", &container])
            .output()
            .await?;

        let _logs = process::Command::new("docker")
            .args(["logs", "-f", &container])
            .spawn()?;

        // Wait until server is accepting connections
        while ping().await.is_err() {
            time::sleep(time::Duration::from_millis(500)).await;
        }

        Ok(Server {
            _container: Arc::new(Container(container)),
        })
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        use std::process;

        let _ = process::Command::new("docker")
            .args(["stop", &self.0])
            .stdin(process::Stdio::null())
            .stdout(process::Stdio::null())
            .stderr(process::Stdio::null())
            .spawn();
    }
}

pub async fn connect() -> Result<net::TcpStream, Error> {
    Ok(net::TcpStream::connect(ADDRESS).await?)
}

async fn ping() -> Result<(), Error> {
    let mut stream = connect().await?;

    #[derive(Serialize)]
    struct Request {
        task: &'static str,
    }

    #[derive(Deserialize)]
    struct Response(bool);

    send_json(&mut stream, Request { task: "ping" }).await?;

    let mut buffer = Vec::new();
    let Response(_pong) = read_json(&mut stream, &mut buffer).await?;

    Ok(())
}

pub async fn read_bytes(stream: &mut net::TcpStream, buffer: &mut Vec<u8>) -> Result<usize, Error> {
    use tokio::io::AsyncReadExt;

    let message_size = stream.read_u64().await? as usize;

    if buffer.len() < message_size {
        buffer.resize(message_size, 0);
    }

    Ok(stream.read_exact(&mut buffer[..message_size]).await?)
}

pub async fn read_json<T: DeserializeOwned>(
    stream: &mut net::TcpStream,
    buffer: &mut Vec<u8>,
) -> Result<T, Error> {
    let message_size = read_bytes(stream, buffer).await?;
    let data = serde_json::from_reader(&buffer[..message_size])?;

    Ok(data)
}

pub async fn send_json<T: Serialize>(stream: &mut net::TcpStream, data: T) -> Result<(), Error> {
    use tokio::io::AsyncWriteExt;

    let bytes = serde_json::to_vec(&data)?;

    stream.write_u64(bytes.len() as u64).await?;
    stream.write_all(bytes.as_slice()).await?;
    stream.flush().await?;

    Ok(())
}
