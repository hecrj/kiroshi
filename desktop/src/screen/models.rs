use crate::animation::Animation;
use crate::civitai::{Error, Id, Model, Rgba};
use crate::widget::{container, diffused_text};

use iced::padding;
use iced::time::Instant;
use iced::widget::{bottom, center_x, horizontal_space, image, pop, row, scrollable, stack, text};
use iced::window;
use iced::{Border, Bottom, ContentFit, Element, Fill, Font, Subscription, Task};

use std::collections::HashMap;

pub struct Models {
    models: Vec<Model>,
    images: HashMap<Id, image::Handle>,
    animations: HashMap<Id, Animation<bool>>,
    now: Instant,
}

#[derive(Debug, Clone)]
pub enum Message {
    ModelsListed(Result<Vec<Model>, Error>),
    ImageDownloaded(Id, Result<Rgba, Error>),
    ModelPopped(Id),
    Animate(Instant),
}

impl Models {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                models: Vec::new(),
                images: HashMap::new(),
                animations: HashMap::new(),
                now: Instant::now(),
            },
            Task::run(Model::list(), Message::ModelsListed),
        )
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let is_animating = self
            .animations
            .values()
            .any(|animation| animation.in_progress(self.now));

        if is_animating {
            window::frames().map(Message::Animate)
        } else {
            Subscription::none()
        }
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ModelsListed(Ok(models)) => {
                self.models = models;

                Task::none()
            }
            Message::ImageDownloaded(model, result) => {
                match result {
                    Ok(image) => {
                        let handle =
                            image::Handle::from_rgba(image.width, image.height, image.pixels);
                        self.images.insert(model, handle);
                    }
                    Err(error) => {
                        dbg!(error);
                    }
                }

                let animation = Animation::new(false).slow().go(true);
                self.animations.insert(model, animation);

                Task::none()
            }
            Message::ModelPopped(model) => {
                let image = self
                    .models
                    .iter()
                    .find(|candidate| candidate.id == model)
                    .map(Model::image)
                    .cloned();

                if let Some(image) = image {
                    Task::perform(image.download(), move |result| {
                        Message::ImageDownloaded(model, result)
                    })
                } else {
                    Task::none()
                }
            }
            Message::Animate(now) => {
                self.now = now;

                Task::none()
            }
            Message::ModelsListed(Err(error)) => {
                dbg!(error);

                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<Message> {
        scrollable(center_x(
            row(self.models.iter().map(|model| {
                card(
                    model,
                    self.images.get(&model.id),
                    self.animations.get(&model.id),
                    self.now,
                )
            }))
            .spacing(10)
            .wrap(),
        ))
        .spacing(10)
        .into()
    }
}

fn card<'a>(
    model: &'a Model,
    cover: Option<&'a image::Handle>,
    animation: Option<&'a Animation<bool>>,
    now: Instant,
) -> Element<'a, Message> {
    use iced::gradient;
    use iced::{Color, Degrees};

    let card = if let Some(animation) = animation {
        let background: Element<_> = if let Some(handle) = cover {
            image(handle)
                .width(Fill)
                .height(Fill)
                .content_fit(ContentFit::Cover)
                .opacity(animation.interpolate(0.0, 1.0, now))
                .into()
        } else {
            container(horizontal_space()).height(Fill).into()
        };

        let name = container(
            diffused_text(&model.name)
                .font(Font::MONOSPACE)
                .size(20)
                .width(Fill)
                .center()
                .shaping(text::Shaping::Advanced),
        )
        .padding(padding::all(10).top(20))
        .align_y(Bottom)
        .style(|theme| container::Style {
            border: Border::default(),
            background: Some(
                gradient::Linear::new(Degrees(180.0))
                    .add_stops([gradient::ColorStop {
                        offset: 0.0,
                        color: Color::TRANSPARENT,
                    }])
                    .add_stops([gradient::ColorStop {
                        offset: 0.5,
                        color: Color::BLACK.scale_alpha(0.7),
                    }])
                    .add_stops([gradient::ColorStop {
                        offset: 1.0,
                        color: Color::BLACK,
                    }])
                    .into(),
            ),
            ..container::translucent(theme)
        });

        stack![background, bottom(name)]
    } else {
        stack![horizontal_space()]
    };

    let card = container(card)
        .width(320)
        .height(410)
        .style(container::dark);

    if cover.is_some() {
        card.into()
    } else {
        pop(card)
            .on_show(Message::ModelPopped(model.id))
            .anticipate(200)
            .into()
    }
}
