use serde::{Deserialize, Serialize};

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Duration(u32);

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} s", self.0)
    }
}

impl From<u8> for Duration {
    fn from(value: u8) -> Self {
        Self(u32::from(value))
    }
}

impl From<Duration> for f64 {
    fn from(value: Duration) -> Self {
        f64::from(value.0)
    }
}

impl num_traits::FromPrimitive for Duration {
    fn from_i64(n: i64) -> Option<Self> {
        u32::try_from(n).map(Self).ok()
    }

    fn from_u64(n: u64) -> Option<Self> {
        u32::try_from(n).map(Self).ok()
    }
}
