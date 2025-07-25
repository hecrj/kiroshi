use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Steps(u32);

impl Steps {
    pub const RANGE: RangeInclusive<Self> = Self(1)..=Self(100);
}

impl Default for Steps {
    fn default() -> Self {
        Self(30)
    }
}

impl fmt::Display for Steps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u8> for Steps {
    fn from(value: u8) -> Self {
        Self(u32::from(value))
    }
}

impl From<Steps> for f64 {
    fn from(value: Steps) -> Self {
        f64::from(value.0)
    }
}

impl num_traits::FromPrimitive for Steps {
    fn from_i64(n: i64) -> Option<Self> {
        u32::try_from(n).map(Self).ok()
    }

    fn from_u64(n: u64) -> Option<Self> {
        u32::try_from(n).map(Self).ok()
    }
}
