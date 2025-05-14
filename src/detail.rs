use crate::Padding;

use serde::{Deserialize, Serialize};

use std::fmt;
use std::ops::RangeInclusive;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Detail {
    pub strength: Strength,
    pub padding: Padding,
    pub max_area: Option<Area>,
}

impl Default for Detail {
    fn default() -> Self {
        Self {
            strength: Strength::default(),
            padding: Padding::from(16),
            max_area: None,
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Area(u32);

impl Area {
    pub fn parse(value: u32) -> Option<Area> {
        if value > 0 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(self) -> u32 {
        self.0
    }
}
