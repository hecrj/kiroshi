use iced::border;
use iced::widget::container;
use iced::{Color, Theme};

pub use iced::widget::container::{dark, rounded_box, Container, Style};

pub fn translucent(theme: &Theme) -> Style {
    let mut style = container::dark(theme);

    style.background = Some(Color::BLACK.scale_alpha(0.8).into());

    style
}

pub fn tab_header(theme: &Theme) -> Style {
    Style {
        border: border::rounded(border::top(2)),
        ..translucent(theme)
    }
}

pub fn tab_content(theme: &Theme) -> Style {
    Style {
        border: border::rounded(border::bottom(2)),
        ..translucent(theme)
    }
}
