use crate::protocol;
use crate::stream::{SinkExt, Stream};
use crate::{
    Detail, Error, Guidance, Lora, Model, Pag, Precision, Quality, Rectangle, Sampler, Seed, Size,
    Steps, Upscaler,
};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Image {
    pub id: Id,
    pub size: Size,
    pub rgba: Bytes,
    pub path: Option<PathBuf>,
    pub definition: Option<Definition>,
}

impl Image {
    pub const DEFAULT_SIZE: Size = Size::new(512, 768);

    pub fn from_file(path: impl AsRef<Path>, size: Size, rgba: Bytes) -> Self {
        use std::hash::{DefaultHasher, Hash, Hasher};

        let path = path.as_ref();

        let id = {
            let mut hasher = DefaultHasher::new();
            path.hash(&mut hasher);

            Id(i64::from_be_bytes(hasher.finish().to_be_bytes()))
        };

        Self {
            id,
            size,
            rgba,
            path: Some(path.to_path_buf()),
            definition: None,
        }
    }

    pub fn generate(definition: Definition) -> impl Stream<Item = Result<Generation, Error>> {
        crate::stream::from_future(move |mut sender| async move {
            let mut stream = protocol::connect(GENERATE).await?;

            stream
                .write(generate::Request {
                    definition: definition.clone(),
                })
                .await?;

            loop {
                let response = stream.read().await?;
                let bytes = stream.read_bytes().await?;

                let image = {
                    let rgba = Bytes::from(bytes.to_vec());
                    let size = Size::new(response.width, response.height);

                    Image {
                        id: response.id,
                        rgba,
                        size,
                        path: None,
                        definition: Some(definition.clone()),
                    }
                };

                let _ = sender
                    .send(if response.is_final {
                        Generation::Finished {
                            image,
                            metadata: (),
                        }
                    } else {
                        Generation::Sampling {
                            image,
                            progress: response.progress,
                        }
                    })
                    .await;

                if response.is_final {
                    break;
                }
            }

            Ok(())
        })
    }

    pub fn detail_faces<'a>(
        &self,
        detail: Detail,
    ) -> impl Stream<Item = Result<Generation<Vec<Rectangle>>, Error>> + 'a {
        let image = self.clone();

        crate::stream::from_future(move |mut sender| async move {
            let Some(definition) = &image.definition else {
                let _ = sender
                    .send(Generation::Finished {
                        image,
                        metadata: Vec::new(),
                    })
                    .await;

                return Ok(());
            };

            let mut stream = protocol::connect(DETAIL_FACES).await?;

            stream
                .write(detail_faces::Request {
                    definition: definition.clone(),
                    detail,
                })
                .await?;

            image.send(&mut stream).await?;

            loop {
                let response = stream.read().await?;
                let bytes = stream.read_bytes().await?;

                let image = {
                    let rgba = Bytes::from(bytes.to_vec());
                    let size = Size::new(response.width, response.height);

                    Image {
                        id: response.id,
                        rgba,
                        size,
                        path: None,
                        definition: Some(definition.clone()),
                    }
                };

                let _ = sender
                    .send(if response.is_final {
                        Generation::Finished {
                            image,
                            metadata: response.faces,
                        }
                    } else {
                        Generation::Sampling {
                            image,
                            progress: response.progress,
                        }
                    })
                    .await;

                if response.is_final {
                    break;
                }
            }

            Ok(())
        })
    }

    pub fn detail_hands<'a>(
        &self,
        detail: Detail,
    ) -> impl Stream<Item = Result<Generation<Vec<Rectangle>>, Error>> + 'a {
        let image = self.clone();

        crate::stream::from_future(move |mut sender| async move {
            let Some(definition) = &image.definition else {
                let _ = sender
                    .send(Generation::Finished {
                        image,
                        metadata: Vec::new(),
                    })
                    .await;

                return Ok(());
            };

            let mut stream = protocol::connect(DETAIL_HANDS).await?;

            stream
                .write(detail_hands::Request {
                    definition: definition.clone(),
                    detail,
                })
                .await?;

            image.send(&mut stream).await?;

            loop {
                let response = stream.read().await?;
                let bytes = stream.read_bytes().await?;

                let image = {
                    let rgba = Bytes::from(bytes.to_vec());
                    let size = Size::new(response.width, response.height);

                    Image {
                        id: response.id,
                        rgba,
                        size,
                        path: None,
                        definition: Some(definition.clone()),
                    }
                };

                let _ = sender
                    .send(if response.is_final {
                        Generation::Finished {
                            image,
                            metadata: response.hands,
                        }
                    } else {
                        Generation::Sampling {
                            image,
                            progress: response.progress,
                        }
                    })
                    .await;

                if response.is_final {
                    break;
                }
            }

            Ok(())
        })
    }

    pub fn upscale<'a>(
        &self,
        upscaler: Upscaler,
    ) -> impl Future<Output = Result<Image, Error>> + 'a {
        let image = self.clone();

        async move {
            let mut stream = protocol::connect(UPSCALE).await?;

            stream
                .write(upscale::Request {
                    upscaler,
                    size: image.size,
                })
                .await?;

            image.send(&mut stream).await?;

            let upscale::Response { id, width, height } = stream.read().await?;
            let rgba = stream.read_bytes().await?;
            let size = Size::new(width, height);

            Ok(Self {
                id,
                rgba: Bytes::from(rgba.to_vec()),
                size,
                path: image.path,
                definition: image
                    .definition
                    .clone()
                    .map(|definition| Definition { size, ..definition }),
            })
        }
    }

    async fn send<I, O>(&self, stream: &mut plug::Connection<I, O>) -> io::Result<()>
    where
        I: Serialize,
    {
        stream.write_bytes(&self.id.0.to_be_bytes()).await?;

        let n = stream.read_bytes().await?;

        // Image is cached by the server
        if n == [1] {
            return Ok(());
        }

        stream.write_bytes(&self.rgba).await
    }
}

impl fmt::Debug for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Image")
            .field("rgba", &format!("{} pixels", self.rgba.len() / 4))
            .field("size", &self.size)
            .field("definition", &self.definition)
            .finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(i64);

#[derive(Debug, Clone)]
pub enum Generation<T = ()> {
    Sampling { image: Image, progress: f32 },
    Finished { image: Image, metadata: T },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Definition {
    pub model: Model,
    pub precision: Precision,
    pub prompt: String,
    pub negative_prompt: String,
    pub size: Size,
    pub seed: Seed,
    pub steps: Steps,
    pub guidance: Guidance,
    pub sampler: Sampler,
    pub pag: Option<Pag>,
    pub loras: Vec<Lora>,
}

impl Definition {
    pub fn with_quality(mut self, quality: Quality) -> Self {
        let scale = quality.scale_factor();

        self.size = Size::new(
            (self.size.width as f32 * scale).round() as u32,
            (self.size.height as f32 * scale).round() as u32,
        );

        self
    }
}

pub const GENERATE: protocol::Plug<generate::Request, generate::Response> =
    protocol::Plug::new("generate_image");

pub mod generate {
    use crate::image::{Definition, Id};

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
        pub progress: f32,
        pub is_final: bool,
    }
}

pub const DETAIL_FACES: protocol::Plug<detail_faces::Request, detail_faces::Response> =
    protocol::Plug::new("detail_faces");

pub mod detail_faces {
    use crate::image::{Definition, Id};
    use crate::{Detail, Rectangle};

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct Request {
        pub definition: Definition,
        pub detail: Detail,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Response {
        pub id: Id,
        pub width: u32,
        pub height: u32,
        pub progress: f32,
        pub is_final: bool,
        #[serde(default)]
        pub faces: Vec<Rectangle>,
    }
}

pub const DETAIL_HANDS: protocol::Plug<detail_hands::Request, detail_hands::Response> =
    protocol::Plug::new("detail_hands");

pub mod detail_hands {
    use crate::image::{Definition, Id};
    use crate::{Detail, Rectangle};

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct Request {
        pub definition: Definition,
        pub detail: Detail,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Response {
        pub id: Id,
        pub width: u32,
        pub height: u32,
        pub progress: f32,
        pub is_final: bool,
        #[serde(default)]
        pub hands: Vec<Rectangle>,
    }
}

pub const UPSCALE: protocol::Plug<upscale::Request, upscale::Response> =
    protocol::Plug::new("upscale");

pub mod upscale {
    use crate::Size;
    use crate::image::{Id, Upscaler};

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct Request {
        pub upscaler: Upscaler,
        pub size: Size,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Response {
        pub id: Id,
        pub width: u32,
        pub height: u32,
    }
}
