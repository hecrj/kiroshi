use std::fmt;
use std::ops::RangeInclusive;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Lora {
    pub file: PathBuf,
    pub strength: Strength,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Strength(u32);

impl Strength {
    pub const RANGE: RangeInclusive<Self> = Self(0)..=Self(500);
}

impl Default for Strength {
    fn default() -> Self {
        Self(100)
    }
}

impl fmt::Display for Strength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}%", self.0)
    }
}

impl From<u8> for Strength {
    fn from(value: u8) -> Self {
        Self(u32::from(value))
    }
}

impl From<Strength> for f64 {
    fn from(value: Strength) -> Self {
        f64::from(value.0)
    }
}

impl num_traits::FromPrimitive for Strength {
    fn from_i64(n: i64) -> Option<Self> {
        u32::try_from(n).ok().map(Self)
    }

    fn from_u64(n: u64) -> Option<Self> {
        u32::try_from(n).ok().map(Self)
    }
}
