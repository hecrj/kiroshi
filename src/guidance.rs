use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Guidance(f64);

impl Guidance {
    pub const RANGE: RangeInclusive<Self> = Self(0.0)..=Self(10.0);
}

impl Default for Guidance {
    fn default() -> Self {
        Self(5.0)
    }
}

impl fmt::Display for Guidance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

impl From<f64> for Guidance {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl num_traits::FromPrimitive for Guidance {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self(n as f64))
    }
}

impl num_traits::AsPrimitive<f64> for Guidance {
    fn as_(self) -> f64 {
        self.0
    }
}
