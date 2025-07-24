use crate::protocol;
use crate::stream::{SinkExt, Stream};
use crate::{
    Detail, Error, Inpaint, Lora, Model, Pag, Quality, Rectangle, Sampler, Seed, Size, Steps,
    Upscaler,
};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use std::fmt;

#[derive(Clone)]
pub struct Image {
    pub rgba: Bytes,
    pub size: Size,
    pub definition: Definition,
}

impl Image {
    pub const DEFAULT_SIZE: Size = Size::new(512, 768);

    pub fn generate(
        definition: Definition,
        preview_after: Option<f32>,
    ) -> impl Stream<Item = Result<Generation, Error>> {
        crate::stream::from_future(move |mut sender| async move {
            let mut stream = protocol::connect(GENERATE).await?;

            stream
                .write(generate::Request {
                    definition: definition.clone(),
                    preview_after,
                })
                .await?;

            loop {
                let response = stream.read().await?;
                let bytes = stream.read_bytes().await?;

                let image = {
                    let rgba = Bytes::from(bytes.to_vec());
                    let size = Size::new(response.width, response.height);

                    Image {
                        rgba,
                        size,
                        definition: definition.clone(),
                    }
                };

                let _ = sender
                    .send(if response.is_final {
                        Generation::Finished {
                            image,
                            faces: response
                                .faces
                                .into_iter()
                                .map(Rectangle::from_array)
                                .collect(),
                            hands: response
                                .hands
                                .into_iter()
                                .map(Rectangle::from_array)
                                .collect(),
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

#[derive(Debug, Clone)]
pub enum Generation {
    Sampling {
        image: Image,
        progress: f32,
    },
    Finished {
        image: Image,
        faces: Vec<Rectangle>,
        hands: Vec<Rectangle>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Definition {
    pub model: Model,
    pub prompt: String,
    pub negative_prompt: String,
    pub size: Size,
    pub seed: Seed,
    pub steps: Steps,
    pub quality: Quality,
    pub sampler: Sampler,
    pub upscaler: Option<Upscaler>,
    pub pag: Option<Pag>,
    pub face_detail: Option<Detail>,
    pub hand_detail: Option<Detail>,
    pub inpaints: Vec<Inpaint>,
    pub loras: Vec<Lora>,
}

pub const GENERATE: protocol::Plug<generate::Request, generate::Response> =
    protocol::Plug::new("generate_image");

pub mod generate {
    use crate::image::Definition;

    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct Request {
        pub definition: Definition,
        pub preview_after: Option<f32>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Response {
        pub width: u32,
        pub height: u32,
        pub progress: f32,
        pub is_final: bool,
        #[serde(default)]
        pub faces: Vec<[f32; 4]>,
        #[serde(default)]
        pub hands: Vec<[f32; 4]>,
    }
}
