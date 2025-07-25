use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum Precision {
    Float16,
    #[default]
    BFloat16,
    Float32,
}

impl Precision {
    pub const ALL: &'static [Self] = &[Self::Float16, Self::BFloat16, Self::Float32];
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Precision::Float16 => "float16",
            Precision::BFloat16 => "bfloat16",
            Precision::Float32 => "float32",
        })
    }
}
