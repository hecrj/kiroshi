use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Seed(u64);

impl Seed {
    pub fn random() -> Self {
        Self(rand::random())
    }

    pub fn value(self) -> u64 {
        self.0
    }

    pub fn rng(self) -> impl rand::Rng {
        use rand::SeedableRng;

        rand::rngs::StdRng::seed_from_u64(self.0)
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{seed}", seed = self.0)
    }
}

impl From<u64> for Seed {
    fn from(value: u64) -> Self {
        Self(value)
    }
}
