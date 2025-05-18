use crate::{Padding, Rectangle, Strength};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Inpaint {
    pub region: Rectangle,
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub strength: Strength,
    pub padding: Padding,
}
