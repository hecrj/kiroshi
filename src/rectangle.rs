use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rectangle {
    pub fn from_array(coordinates: [f32; 4]) -> Self {
        let [left, top, right, bottom] = coordinates;

        Self {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        }
    }
}
