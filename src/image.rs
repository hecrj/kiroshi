use crate::protocol;
use crate::stream::{SinkExt, Stream};
use crate::{
    Detail, Error, Inpaint, Lora, Model, Quality, Rectangle, Sampler, Seed, Size, Steps, Upscaler,
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
        #[derive(Deserialize)]
        struct Response {
            width: u32,
            height: u32,
            progress: f32,
            is_final: bool,
            #[serde(default)]
            faces: Vec<[f32; 4]>,
            #[serde(default)]
            hands: Vec<[f32; 4]>,
        }

        crate::stream::from_future(move |mut sender| async move {
            let mut stream = protocol::perform(protocol::Request::GenerateImage {
                definition: definition.clone(),
                preview_after,
            })
            .await?;
            let mut buffer = Vec::new();

            loop {
                let response: Response = protocol::read_json(&mut stream, &mut buffer).await?;
                let n_bytes = protocol::read_bytes(&mut stream, &mut buffer).await?;

                let image = {
                    let rgba = Bytes::from(buffer[..n_bytes].to_vec());
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
    pub face_detail: Option<Detail>,
    pub hand_detail: Option<Detail>,
    pub inpaints: Vec<Inpaint>,
    pub loras: Vec<Lora>,
}
