use iced::futures::channel::mpsc;
use iced::futures::{SinkExt, Stream};
use iced::stream;
use serde::Deserialize;
use tokio::fs;
use tokio::task;

use std::fmt;
use std::io;
use std::sync::Arc;

pub use bytes::Bytes;

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: Id,
    pub name: String,
    pub description: String,
    #[serde(rename = "modelVersions")]
    pub versions: Vec<Version>,
}

impl Model {
    pub fn list() -> impl Stream<Item = Result<Vec<Self>, Error>> {
        stream::try_channel(1, move |mut sender: mpsc::Sender<_>| async move {
            let cache = dirs::cache_dir()
                .map(|cache| cache.join("kiroshi").join("models").join("all.json"));

            if let Some(cache) = &cache {
                if let Ok(json) = fs::read_to_string(&cache).await {
                    if let Some(response) =
                        task::spawn_blocking(move || serde_json::from_str::<Response>(&json).ok())
                            .await
                            .ok()
                            .flatten()
                    {
                        let _ = sender.send(response.items).await;
                    }
                }
            }

            let client = reqwest::Client::new();

            let request = get(&client, "/models").query(&[
                ("types", "Checkpoint"),
                ("sort", "Highest Rated"),
                ("period", "AllTime"),
                ("nsfw", "false"),
                ("baseModels", "Illustrious"),
                ("baseModels", "SDXL 1.0"),
                // ("baseModels", "Pony"), // TODO: Add configurable NSFW flag
            ]);

            #[derive(Deserialize)]
            struct Response {
                items: Vec<Model>,
            }

            let json = request.send().await?.error_for_status()?.text().await?;

            if let Some(cache) = &cache {
                let _ = fs::create_dir_all(cache.parent().unwrap_or(cache)).await;
                let _ = fs::write(cache, &json).await;
            }

            let response: Response =
                task::spawn_blocking(move || serde_json::from_str(&json)).await??;
            let _ = sender.send(response.items.clone()).await;

            Ok(())
        })
    }

    pub fn image(&self) -> &Image {
        static PLACEHOLDER: &Image = &Image {
            id: 0,
            url: String::new(),
            type_: String::new(),
        };

        for version in &self.versions {
            if let Some(image) = version.images.iter().find(|image| image.type_ == "image") {
                return image;
            }
        }

        PLACEHOLDER
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
pub struct Id(u64);

#[derive(Debug, Clone, Deserialize)]
pub struct Version {
    pub name: String,
    pub files: Vec<File>,
    pub images: Vec<Image>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct File {
    pub name: String,
    #[serde(rename = "downloadUrl")]
    pub download_url: String,
    #[serde(rename = "sizeKB")]
    pub size: KBytes,
    #[serde(rename = "primary", default)]
    pub is_primary: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Image {
    id: u64,
    url: String,
    #[serde(rename = "type")]
    type_: String,
}

impl Image {
    pub async fn download(self) -> Result<Rgba, Error> {
        use image::ImageReader;
        use std::io::Cursor;

        static PLACEHOLDER: Bytes =
            Bytes::from_static(include_bytes!("../assets/model_placeholder.jpg"));

        let cache = dirs::cache_dir().map(|cache| {
            cache
                .join("kiroshi")
                .join("images")
                .join(self.id.to_string())
        });

        let bytes = match &cache {
            Some(cache) if fs::try_exists(cache).await.unwrap_or_default() => {
                Bytes::from(fs::read(cache).await?)
            }
            _ if self.id == 0 => PLACEHOLDER.clone(),
            _ => {
                let client = reqwest::Client::new();

                // TODO: Logging
                println!("[civitai] Downloading {}", &self.url);

                let bytes = client
                    .get(self.url.replace("width=450", "width=640"))
                    .send()
                    .await?
                    .error_for_status()?
                    .bytes()
                    .await?;

                if let Some(cache) = cache {
                    let _ = fs::create_dir_all(cache.parent().unwrap_or(&cache)).await;
                    let _ = fs::write(cache, &bytes).await;
                }

                bytes
            }
        };

        // Decode image as RGBA in a background blocking thread

        let decode = |bytes| {
            task::spawn_blocking(move || {
                let image = ImageReader::new(Cursor::new(bytes))
                    .with_guessed_format()?
                    .decode()?
                    .to_rgba8();

                Ok::<_, Error>(image)
            })
        };

        let image = if let Ok(image) = decode(bytes).await? {
            image
        } else {
            decode(PLACEHOLDER.clone()).await??
        };

        Ok(Rgba {
            width: image.width(),
            height: image.height(),
            pixels: Bytes::from(image.into_raw()),
        })
    }
}

#[derive(Clone)]
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub pixels: Bytes,
}

impl fmt::Debug for Rgba {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rgba")
            .field(
                "pixels",
                &format!("Bytes(total: {})", self.pixels.len() / 4),
            )
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct KBytes(f64);

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Error {
    RequestFailed(Arc<reqwest::Error>),
    IOFailed(Arc<io::Error>),
    JoinFailed(Arc<task::JoinError>),
    ImageDecodingFailed(Arc<image::ImageError>),
    JsonDecodingFailed(Arc<serde_json::Error>),
}

impl From<reqwest::Error> for Error {
    fn from(error: reqwest::Error) -> Self {
        Error::RequestFailed(Arc::new(error))
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::IOFailed(Arc::new(error))
    }
}

impl From<task::JoinError> for Error {
    fn from(error: task::JoinError) -> Self {
        Error::JoinFailed(Arc::new(error))
    }
}

impl From<image::ImageError> for Error {
    fn from(error: image::ImageError) -> Self {
        Error::ImageDecodingFailed(Arc::new(error))
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::JsonDecodingFailed(Arc::new(error))
    }
}

const API_URL: &str = "https://civitai.com/api/v1";

fn get(client: &reqwest::Client, endpoint: &str) -> reqwest::RequestBuilder {
    client.get(format!("{API_URL}{endpoint}"))
}
