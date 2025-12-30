use crate::protocol;
use crate::{Bytes, Duration, Error, Guidance, Image, Precision, Seed, Size, Steps};

use serde::{Deserialize, Serialize};

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Video {
    pub id: Id,
    pub size: Size,
    pub framerate: u32,
    pub frames: Arc<[Image]>,
    pub path: Option<PathBuf>,
}

impl Video {
    pub async fn generate(
        definition: Definition,
        first_frame: Image,
        last_frame: Option<Image>,
    ) -> Result<Self, Error> {
        let mut stream = protocol::connect(GENERATE).await?;

        stream
            .write(generate::Request {
                definition: definition.clone(),
            })
            .await?;

        first_frame.send(&mut stream).await?;

        if let Some(last_frame) = last_frame {
            stream.write_bytes(&[1]).await?;
            last_frame.send(&mut stream).await?;
        } else {
            stream.write_bytes(&[0]).await?;
        }

        let generate::Response {
            id,
            width,
            height,
            framerate,
            frames: frame_ids,
        } = stream.read().await?;

        let size = Size::new(width, height);
        let mut frames = Vec::new();

        for frame in frame_ids {
            let bytes = stream.read_bytes().await?;
            let rgba = Bytes::from(bytes.to_vec());

            frames.push(Image {
                id: frame,
                size,
                rgba,
                path: None,
                definition: None,
            });
        }

        Ok(Self {
            id,
            size,
            framerate,
            frames: Arc::from(frames),
            path: first_frame.path,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
}

impl Model {
    pub async fn list() -> Result<Vec<Self>, Error> {
        let mut stream = protocol::connect(LIST_MODELS).await?;

        Ok(stream.read().await?)
    }
}

pub const LIST_MODELS: protocol::Plug<protocol::Never, Vec<Model>> =
    protocol::Plug::new("list_video_models");

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(i64);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Definition {
    pub model: Model,
    pub precision: Precision,
    pub prompt: String,
    pub negative_prompt: String,
    pub size: Size,
    pub max_area: u32,
    pub seed: Seed,
    pub steps: Steps,
    pub guidance: Guidance,
    pub duration: Duration,
}

pub const GENERATE: protocol::Plug<generate::Request, generate::Response> =
    protocol::Plug::new("generate_video");

pub mod generate {
    use super::{Definition, Id};
    use crate::image;

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct Request {
        pub definition: Definition,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Response {
        pub id: Id,
        pub width: u32,
        pub height: u32,
        pub framerate: u32,
        pub frames: Vec<image::Id>,
    }
}
