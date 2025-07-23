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
        while protocol::ping().await.is_err() {
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
