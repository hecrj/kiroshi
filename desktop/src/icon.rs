// Generated automatically by iced_fontello at build time.
// Do not edit manually. Source: ../fonts/kiroshi-icons.toml
// 496b54933ad4a8dac06598a2daf5b3f3b5b519e49dd0732d4f93037664f1f6f3
use iced::widget::{text, Text};
use iced::Font;

pub const FONT: &[u8] = include_bytes!("../fonts/kiroshi-icons.ttf");

pub fn checkmark<'a>() -> Text<'a> {
    icon("\u{2713}")
}

pub fn hidden<'a>() -> Text<'a> {
    icon("\u{E70B}")
}

pub fn visible<'a>() -> Text<'a> {
    icon("\u{E70A}")
}

fn icon<'a>(codepoint: &'a str) -> Text<'a> {
    text(codepoint).font(Font::with_name("kiroshi-icons"))
}
