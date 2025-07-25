use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Pag {
    pub scale: Scale,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Scale(f64);

impl Scale {
    pub const ZERO: Self = Self(0.0);
    pub const RANGE: RangeInclusive<Self> = Self::ZERO..=Self(10.0);
}

impl Default for Scale {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 > 0.0 {
            write!(f, "{:.2}", self.0)
        } else {
            f.write_str("Disabled")
        }
    }
}

impl From<u8> for Scale {
    fn from(value: u8) -> Self {
        Self(f64::from(value))
    }
}

impl From<Scale> for f64 {
    fn from(value: Scale) -> Self {
        value.0
    }
}

impl num_traits::FromPrimitive for Scale {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(n as f64))
    }
}
