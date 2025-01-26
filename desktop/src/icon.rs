// Generated automatically by iced_fontello at build time.
// Do not edit manually. Source: ../fonts/kiroshi-icons.toml
// b5a8bed255f8af27dd8f08e998897380494ee0ce7866cc2f4a3f50dcbb4f5878
use iced::widget::{text, Text};
use iced::Font;

pub const FONT: &[u8] = include_bytes!("../fonts/kiroshi-icons.ttf");

pub fn brush<'a>() -> Text<'a> {
    icon("\u{F1FC}")
}

pub fn checkmark<'a>() -> Text<'a> {
    icon("\u{2713}")
}

pub fn cubes<'a>() -> Text<'a> {
    icon("\u{F1B3}")
}

pub fn hidden<'a>() -> Text<'a> {
    icon("\u{E70B}")
}

pub fn refresh<'a>() -> Text<'a> {
    icon("\u{1F504}")
}

pub fn visible<'a>() -> Text<'a> {
    icon("\u{E70A}")
}

fn icon<'a>(codepoint: &'a str) -> Text<'a> {
    text(codepoint).font(Font::with_name("kiroshi-icons"))
}
