use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Strength(u32);

impl Strength {
    pub const RANGE: RangeInclusive<Self> = Self(0)..=Self(100);
}

impl Default for Strength {
    fn default() -> Self {
        Self(50)
    }
}

impl fmt::Display for Strength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}%", self.0)
    }
}

impl From<u32> for Strength {
    fn from(value: u32) -> Self {
        Self(value)
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

impl num_traits::AsPrimitive<f64> for Strength {
    fn as_(self) -> f64 {
        f64::from(self.0)
    }
}
