use crate::model;
use crate::Error;

use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::fs;
use tokio::io;
use tokio::net;
use tokio::process;

const ADDRESS: &'static str = "127.0.0.1:9149";

#[derive(Debug, Clone)]
pub struct Server {
    _container: Arc<Container>,
}

#[derive(Debug)]
struct Container(String);

impl Server {
    pub async fn run() -> Result<Server, Error> {
        let models_dir = model::directory()?;
        fs::create_dir_all(&models_dir).await?;

        let mut process = process::Command::new("docker")
            .arg("create")
            .args(["-t", "--rm"])
            .args(["--gpus", "all"])
            .args(["-p", "9149:9149"])
            .args(["-v", {
                let model_dir = models_dir.to_string_lossy().into_owned();

                &format!(
                    "{host}:{container}",
                    host = model_dir,
                    container = model_dir,
                )
            }])
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
