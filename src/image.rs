use crate::server;
use crate::stream::{SinkExt, Stream};
use crate::{Detail, Error, Lora, Model, Quality, Rectangle, Sampler, Seed, Size, Steps};

use std::path::PathBuf;

use bytes::Bytes;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct Image {
    pub bytes: Bytes,
    pub size: Size,
    pub definition: Definition,
    pub upscale_factor: f32,
}

impl Image {
    pub const DEFAULT_SIZE: Size = Size::new(512, 768);

    pub fn generate(
        definition: Definition,
        preview_after: Option<f32>,
    ) -> impl Stream<Item = Result<Generation, Error>> {
        #[derive(Serialize)]
        struct Request {
            model: PathBuf,
            prompt: String,
            negative_prompt: String,
            size: Size,
            quality: String,
            sampler: String,
            steps: Steps,
            seed: u64,
            face_detail: Option<Detail>,
            hand_detail: Option<Detail>,
            loras: Vec<Lora>,
            preview_after: Option<f32>,
        }

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
            let mut stream = server::connect().await?;
            let mut buffer = Vec::new();

            let request = Request {
                model: definition.model.path().to_path_buf(),
                prompt: definition.prompt.clone(),
                negative_prompt: definition.negative_prompt.clone(),
                size: definition.size,
                quality: definition.quality.to_string().to_lowercase(),
                sampler: match definition.sampler {
                    Sampler::EulerAncestral => "euler_a",
                    Sampler::DPMSDEKarras => "dpm++_sde_karras",
                    Sampler::DPM2MKarras => "dpm++_2m_karras",
                    Sampler::DPM2MSDEKarras => "dpm++_2m_sde_karras",
                }
                .to_owned(),
                steps: definition.steps,
                seed: definition.seed.value(),
                face_detail: definition.face_detail,
                hand_detail: definition.hand_detail,
                loras: definition.loras.clone(),
                preview_after,
            };

            server::send_json(&mut stream, request).await?;

            loop {
                let response: Response = server::read_json(&mut stream, &mut buffer).await?;
                let n_bytes = server::read_bytes(&mut stream, &mut buffer).await?;

                let image = {
                    let bytes = Bytes::from(buffer[..n_bytes].to_vec());
                    let size = Size::new(response.width, response.height);

                    Image {
                        bytes,
                        size,
                        definition: definition.clone(),
                        upscale_factor: 2.0,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Definition {
    pub model: Model,
    pub prompt: String,
    pub negative_prompt: String,
    pub size: Size,
    pub seed: Seed,
    pub steps: Steps,
    pub quality: Quality,
    pub sampler: Sampler,
    pub face_detail: Option<Detail>,
    pub hand_detail: Option<Detail>,
    pub loras: Vec<Lora>,
}
