use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Steps(u32);

impl Default for Steps {
    fn default() -> Self {
        Self(30)
    }
}
