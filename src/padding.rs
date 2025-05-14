use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Padding(u32);

impl Padding {
    pub const RANGE: RangeInclusive<Self> = Self(0)..=Self(100);
}

impl fmt::Display for Padding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}px", self.0)
    }
}

impl From<u8> for Padding {
    fn from(value: u8) -> Self {
        Self(u32::from(value))
    }
}

impl From<Padding> for f64 {
    fn from(value: Padding) -> Self {
        f64::from(value.0)
    }
}

impl From<Padding> for f32 {
    fn from(value: Padding) -> Self {
        value.0 as f32
    }
}

impl num_traits::FromPrimitive for Padding {
    fn from_i64(n: i64) -> Option<Self> {
        u32::try_from(n).ok().map(Self)
    }

    fn from_u64(n: u64) -> Option<Self> {
        u32::try_from(n).ok().map(Self)
    }
}
