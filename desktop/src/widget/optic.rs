use crate::kiroshi::Image;

use iced::widget::{container, horizontal_space, image};
use iced::Element;

pub struct Handle {
    raw: image::Handle,
    image: Image,
}

impl Handle {
    pub fn new(image: Image) -> Self {
        Self {
            raw: image::Handle::from_rgba(image.size.width, image.size.height, image.bytes.clone()),
            image,
        }
    }
}

pub fn original<'a, Message>(handle: &Handle) -> Element<'a, Message> {
    let size = handle.image.definition.size;

    image(handle.raw.clone())
        .width(size.width as f32)
        .height(size.height as f32)
        .into()
}

pub fn placeholder<'a, Message: 'static>() -> Element<'a, Message> {
    container(horizontal_space())
        .width(Image::DEFAULT_SIZE.width as f32)
        .height(Image::DEFAULT_SIZE.height as f32)
        .into()
}
