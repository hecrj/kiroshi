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

impl From<u32> for Padding {
    fn from(value: u32) -> Self {
        Self(value)
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

impl num_traits::AsPrimitive<f64> for Padding {
    fn as_(self) -> f64 {
        f64::from(self.0)
    }
}
