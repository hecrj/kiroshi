use crate::animation::{Animation, Easing};
use crate::civitai::{Error, Id, Model, Rgba};
use crate::widget::{container, diffused_text};

use iced::padding;
use iced::time::Instant;
use iced::widget::{
    bottom, button, center_x, horizontal_space, image, mouse_area, pop, row, scrollable, stack,
    text,
};
use iced::window;
use iced::{Border, Bottom, ContentFit, Element, Fill, Font, Subscription, Task};

use std::collections::HashMap;

pub struct Models {
    models: Vec<Model>,
    images: HashMap<Id, image::Handle>,
    effects: HashMap<Id, Effect>,
    now: Instant,
}

struct Effect {
    fade_in: Animation<bool>,
    zoom: Animation<bool>,
}

#[derive(Debug, Clone)]
pub enum Message {
    ModelsListed(Result<Vec<Model>, Error>),
    ModelPopped(Id),
    ShowModel(Id),
    ImageDownloaded(Id, Result<Rgba, Error>),
    ImageHovered(Id, bool),
    Animate(Instant),
}

impl Models {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                models: Vec::new(),
                images: HashMap::new(),
                effects: HashMap::new(),
                now: Instant::now(),
            },
            Task::run(Model::list(), Message::ModelsListed),
        )
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let is_animating = self.effects.values().any(|effect| {
            effect.fade_in.in_progress(self.now) || effect.zoom.in_progress(self.now)
        });

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
            Message::ShowModel(model) => {
                dbg!(model);

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

                let effect = Effect {
                    fade_in: Animation::new(false).slow().go(true),
                    zoom: Animation::new(false).quick().easing(Easing::EaseInOut),
                };

                self.effects.insert(model, effect);

                Task::none()
            }
            Message::ImageHovered(model, is_hovered) => {
                if let Some(effect) = self.effects.get_mut(&model) {
                    effect.zoom.go_mut(is_hovered);
                }

                Task::none()
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
                    self.effects.get(&model.id),
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
    effect: Option<&'a Effect>,
    now: Instant,
) -> Element<'a, Message> {
    use iced::gradient;
    use iced::{Color, Degrees};

    let card = if let Some(effect) = effect {
        let background: Element<_> = if let Some(handle) = cover {
            mouse_area(
                image(handle)
                    .width(Fill)
                    .height(Fill)
                    .content_fit(ContentFit::Cover)
                    .opacity(effect.fade_in.interpolate(0.0, 1.0, now))
                    .scale(effect.zoom.interpolate(1.0, 1.1, now)),
            )
            .on_enter(Message::ImageHovered(model.id, true))
            .on_exit(Message::ImageHovered(model.id, false))
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

    let card = button(
        container(card)
            .width(320)
            .height(410)
            .style(container::dark),
    )
    .on_press(Message::ShowModel(model.id))
    .padding(0)
    .style(button::text);

    if cover.is_some() {
        card.into()
    } else {
        pop(card)
            .on_show(|_| Message::ModelPopped(model.id))
            .anticipate(200)
            .into()
    }
}
