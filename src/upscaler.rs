use crate::Padding;

use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Upscaler {
    pub model: Model,
    pub tile_size: TileSize,
    pub tile_padding: Padding,
}

impl Default for Upscaler {
    fn default() -> Self {
        Self {
            model: Model::UltrasharpX4,
            tile_size: TileSize(192),
            tile_padding: Padding::from(24),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Model {
    #[serde(rename = "2x-real_esrgan")]
    RealEsrganX2,
    #[serde(rename = "4x-ultrasharp")]
    UltrasharpX4,
}

impl Model {
    pub const ALL: &'static [Self] = &[Self::RealEsrganX2, Self::UltrasharpX4];
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::RealEsrganX2 => "RealESRGAN (2x)",
            Self::UltrasharpX4 => "UltraSharp (4x)",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileSize(u32);

impl TileSize {
    pub const RANGE: RangeInclusive<Self> = Self(100)..=Self(300);
}

impl fmt::Display for TileSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}px", self.0)
    }
}

impl From<u32> for TileSize {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl num_traits::FromPrimitive for TileSize {
    fn from_i64(n: i64) -> Option<Self> {
        u32::try_from(n).ok().map(Self)
    }

    fn from_u64(n: u64) -> Option<Self> {
        u32::try_from(n).ok().map(Self)
    }
}

impl num_traits::AsPrimitive<f64> for TileSize {
    fn as_(self) -> f64 {
        f64::from(self.0)
    }
}
