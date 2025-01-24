// Generated automatically by iced_fontello at build time.
// Do not edit manually. Source: ../fonts/kiroshi-icons.toml
// c7b35070b593a7e33c64f3cbcefeb46579d95fe7cab37c62fb36a96387d18df9
use iced::widget::{text, Text};
use iced::Font;

pub const FONT: &[u8] = include_bytes!("../fonts/kiroshi-icons.ttf");

pub fn checkmark<'a>() -> Text<'a> {
    icon("\u{2713}")
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
