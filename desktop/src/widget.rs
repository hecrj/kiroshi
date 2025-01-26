mod diffused_text;

pub mod container;
pub mod optic;

pub use container::Container;
pub use diffused_text::DiffusedText;

use iced::advanced;
use iced::border;
use iced::font;
use iced::widget::{
    bottom_center, horizontal_space, progress_bar, row, slider, stack, text, tooltip, Row,
};
use iced::{Center, Element, Fill, FillPortion, Font, Pixels, Theme};

use std::ops::RangeInclusive;

pub const LOGO_FONT: &'static [u8] = include_bytes!("../fonts/Orbitron-Medium.ttf");

pub fn container<'a, Message>(element: impl Into<Element<'a, Message>>) -> Container<'a, Message> {
    Container::new(element)
}

pub fn diffused_text<'a, Theme, Renderer>(
    fragment: impl text::IntoFragment<'a>,
) -> DiffusedText<'a, Theme, Renderer>
where
    Theme: text::Catalog,
    Renderer: advanced::text::Renderer,
{
    DiffusedText::new(fragment)
}

pub fn logo<'a, Message: 'a>(size: impl Into<Pixels>) -> Element<'a, Message> {
    container(text("kiroshi").size(size).font(Font {
        weight: font::Weight::Medium,
        ..Font::with_name("Orbitron")
    }))
    .padding([5, 0])
    .into()
}

pub fn labeled_slider<'a, T, Message: Clone + 'static>(
    label: impl text::IntoFragment<'a>,
    range: RangeInclusive<T>,
    current: T,
    on_change: impl Fn(T) -> Message + 'a,
    to_string: impl Fn(&T) -> String,
) -> Row<'a, Message>
where
    T: 'static + Copy + PartialOrd + Into<f64> + From<u8> + num_traits::FromPrimitive,
{
    row![
        text(label).size(14).width(FillPortion(2)),
        slider(range, current, on_change).width(FillPortion(5)),
        container(
            text(to_string(&current))
                .font(Font::MONOSPACE)
                .size(14)
                .line_height(1.0)
        )
        .center_x(FillPortion(3))
        .padding([5, 2])
        .style(container::rounded_box),
    ]
    .spacing(10)
    .align_y(Center)
}

pub fn gauge<'a, Message: 'a>(
    label: impl text::IntoFragment<'a>,
    value: impl ToString,
    ratio: f32,
) -> Element<'a, Message> {
    stack![
        tooltip(
            progress_bar(0.0..=1.0, ratio)
                .length(Fill)
                .girth(20)
                .vertical()
                .style(match (ratio * 100.0) as u32 {
                    81.. => progress_bar::danger,
                    61..=80 => progress_bar::warning,
                    _ => progress_bar::primary,
                }),
            container(text(value.to_string()).size(10).font(Font::MONOSPACE))
                .padding(5)
                .style(container::dark),
            tooltip::Position::Top
        ),
        bottom_center(
            text(label)
                .size(6)
                .font(Font::MONOSPACE)
                .style(|theme: &Theme| {
                    text::Style {
                        color: Some(theme.palette().background),
                    }
                })
        ),
    ]
    .into()
}

pub fn indicator<'a, Message: 'a>(
    on: impl text::IntoFragment<'a>,
    off: impl text::IntoFragment<'a>,
    status: bool,
) -> Element<'a, Message> {
    let circle = container(horizontal_space())
        .width(6)
        .height(6)
        .style(move |theme| {
            let theme = theme.palette();

            container::Style {
                background: Some(if status { theme.success } else { theme.danger }.into()),
                border: border::rounded(3),
                ..container::Style::default()
            }
        });

    row![
        circle,
        if status { text(on) } else { text(off) }
            .size(10)
            .font(Font::MONOSPACE)
    ]
    .spacing(5)
    .align_y(Center)
    .into()
}
