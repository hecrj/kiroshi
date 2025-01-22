use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum Sampler {
    #[default]
    EulerAncestral,
    DPMSDEKarras,
    DPM2MKarras,
    DPM2MSDEKarras,
}

impl Sampler {
    pub const ALL: &'static [Self] = &[
        Self::EulerAncestral,
        Self::DPMSDEKarras,
        Self::DPM2MKarras,
        Self::DPM2MSDEKarras,
    ];
}

impl fmt::Display for Sampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Sampler::EulerAncestral => "Euler a",
            Sampler::DPMSDEKarras => "DPM++ SDE Karras",
            Sampler::DPM2MKarras => "DPM++ 2M Karras",
            Sampler::DPM2MSDEKarras => "DPM++ 2M SDE Karras",
        })
    }
}
