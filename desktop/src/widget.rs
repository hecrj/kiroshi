pub mod container;
pub mod optic;

pub use container::Container;

use std::ops::RangeInclusive;

use iced::font;
use iced::widget::{row, slider, text, Row};
use iced::{Center, Element, FillPortion, Font, Pixels};

pub const LOGO_FONT: &'static [u8] = include_bytes!("../fonts/Orbitron-Medium.ttf");

pub fn container<'a, Message>(element: impl Into<Element<'a, Message>>) -> Container<'a, Message> {
    Container::new(element)
}

pub fn logo<'a, Message>(size: impl Into<Pixels>) -> Element<'a, Message> {
    text("kiroshi")
        .size(size)
        .font(Font {
            weight: font::Weight::Medium,
            ..Font::with_name("Orbitron")
        })
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
