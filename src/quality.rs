use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum Quality {
    Low,
    Normal,
    #[default]
    High,
    Ultra,
    Insane,
}

impl Quality {
    pub const ALL: &'static [Self] = &[
        Self::Low,
        Self::Normal,
        Self::High,
        Self::Ultra,
        Self::Insane,
    ];

    pub fn scale_factor(self) -> f32 {
        match self {
            Quality::Low => 1.0,
            Quality::Normal => 1.25,
            Quality::High => 1.5,
            Quality::Ultra => 1.75,
            Quality::Insane => 2.0,
        }
    }
}

impl fmt::Display for Quality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Quality::Low => "Low",
            Quality::Normal => "Normal",
            Quality::High => "High",
            Quality::Ultra => "Ultra",
            Quality::Insane => "Insane",
        })
    }
}
