use crate::Error;
use crate::protocol;

use tokio::fs;
use tokio::io;
use tokio::process;
use tokio::time;

use std::path::Path;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Server {
    _container: Arc<Container>,
}

#[derive(Debug)]
struct Container(String);

impl Server {
    pub async fn run(
        image_models_dir: impl AsRef<Path>,
        video_models_dir: impl AsRef<Path>,
    ) -> Result<Server, Error> {
        let image_models = {
            let image_models_dir = image_models_dir.as_ref();
            fs::create_dir_all(&image_models_dir).await?;

            format!(
                "{host}:/models/image",
                host = image_models_dir.to_string_lossy()
            )
        };

        let video_models = {
            let video_models_dir = video_models_dir.as_ref();
            fs::create_dir_all(&video_models_dir).await?;

            format!(
                "{host}:/models/video",
                host = video_models_dir.to_string_lossy()
            )
        };

        let mut process = process::Command::new("docker")
            .arg("create")
            .args(["-t", "--rm"])
            .args(["--gpus", "all"])
            .args(["-p", "9149:9149"])
            .args(["-v", &image_models])
            .args(["-v", &video_models])
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

pub const PING: protocol::Plug<protocol::Never, bool> = protocol::Plug::new("ping");

async fn ping() -> Result<bool, Error> {
    let mut stream = protocol::connect(PING).await?;

    Ok(stream.read().await?)
}
