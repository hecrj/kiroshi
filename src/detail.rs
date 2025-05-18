use crate::{Padding, Strength};

use serde::{Deserialize, Serialize};

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
