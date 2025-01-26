use kiroshi;

mod animation;
mod civitai;
mod icon;
mod screen;
mod widget;

use crate::kiroshi::{Error, Server, Stats};
use crate::screen::generator;
use crate::screen::models;
use crate::screen::Screen;
use crate::widget::{container, gauge, indicator, logo};

use iced::time;
use iced::widget::{button, column, horizontal_space, row, text, Text};
use iced::{color, Center, Element, Fill, Subscription, Task, Theme};

fn main() -> iced::Result {
    iced::application(Kiroshi::title, Kiroshi::update, Kiroshi::view)
        .subscription(Kiroshi::subscription)
        .theme(Kiroshi::theme)
        .window_size((1024.0, 820.0))
        .font(icon::FONT)
        .font(widget::LOGO_FONT)
        .run_with(Kiroshi::new)
}

struct Kiroshi {
    server: Option<Server>,
    stats: Option<Stats>,
    theme: Theme,
    screen: Screen,
}

#[derive(Debug, Clone)]
enum Message {
    ServerBooted(Result<Server, Error>),
    StatsFetched(Result<Stats, Error>),
    Generator(generator::Message),
    Models(models::Message),
    NavigateToGenerator,
    NavigateToModels,
}

impl Kiroshi {
    pub fn new() -> (Self, Task<Message>) {
        let (generate, task) = screen::Generator::new();

        (
            Self {
                server: None,
                stats: None,
                theme: Theme::TokyoNight,
                screen: Screen::Generator(generate),
            },
            Task::batch([
                Task::perform(Server::run(), Message::ServerBooted),
                task.map(Message::Generator),
            ]),
        )
    }

    pub fn title(&self) -> String {
        "Kiroshi".to_owned()
    }

    pub fn theme(&self) -> Theme {
        self.theme.clone()
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let fetch_stats = time::repeat(Stats::fetch, time::seconds(2)).map(Message::StatsFetched);

        let screen = match &self.screen {
            Screen::Generator(_generator) => Subscription::none(),
            Screen::Models(models) => models.subscription().map(Message::Models),
        };

        Subscription::batch([fetch_stats, screen])
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ServerBooted(Ok(server)) => {
                self.server = Some(server);

                Task::none()
            }
            Message::StatsFetched(Ok(stats)) => {
                self.stats = Some(stats);

                Task::none()
            }
            Message::Generator(message) => {
                let Screen::Generator(generator) = &mut self.screen else {
                    return Task::none();
                };

                generator.update(message).map(Message::Generator)
            }
            Message::Models(message) => {
                let Screen::Models(models) = &mut self.screen else {
                    return Task::none();
                };

                models.update(message).map(Message::Models)
            }
            Message::NavigateToGenerator => {
                let (generator, task) = screen::Generator::new();

                self.screen = Screen::Generator(generator);

                task.map(Message::Generator)
            }
            Message::NavigateToModels => {
                let (models, task) = screen::Models::new();

                self.screen = Screen::Models(models);

                task.map(Message::Models)
            }
            Message::ServerBooted(Err(error)) | Message::StatsFetched(Err(error)) => {
                dbg!(error);

                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<Message> {
        let navbar = {
            let tabs = {
                fn tab<'a>(
                    name: &'a str,
                    icon: Text<'a>,
                    is_active: bool,
                    on_press: Message,
                ) -> Element<'a, Message> {
                    button(
                        row![icon.size(14), text(name).size(14)]
                            .spacing(10)
                            .height(Fill)
                            .align_y(Center),
                    )
                    .on_press(on_press)
                    .padding([5, 15])
                    .style(move |theme, status| {
                        if !is_active {
                            button::text(theme, status)
                        } else {
                            let palette = theme.extended_palette();

                            button::Style {
                                background: Some(palette.background.base.color.into()),
                                text_color: palette.background.base.text,
                                ..button::Style::default()
                            }
                        }
                    })
                    .into()
                }

                row![
                    tab(
                        "Generator",
                        icon::brush(),
                        matches!(self.screen, Screen::Generator(_)),
                        Message::NavigateToGenerator,
                    ),
                    tab(
                        "Models",
                        icon::cubes(),
                        matches!(self.screen, Screen::Models(_)),
                        Message::NavigateToModels,
                    )
                ]
            };

            let status = indicator("Online", "Offline", self.server.is_some());

            let stats = if let Some(stats) = self.stats {
                let gpu_temperature = gauge(
                    "TEMP",
                    stats.gpu_temperature,
                    stats.gpu_temperature.celsius as f32 / 90.0,
                );

                let vram_usage = gauge(
                    "VRAM",
                    format!("{} / {}", stats.vram_usage.used(), stats.vram_usage.total()),
                    stats.vram_usage.ratio(),
                );

                row![vram_usage, gpu_temperature]
                    .spacing(10)
                    .padding([5, 0])
            } else {
                row![]
            };

            container(
                row![logo(20), tabs, horizontal_space(), status, stats]
                    .spacing(10)
                    .padding([0, 5])
                    .align_y(Center),
            )
            .height(30)
            .width(Fill)
            .style(|_theme| container::Style::default().background(color!(0x000000, 0.5)))
        };

        let screen = match &self.screen {
            Screen::Generator(generate) => generate.view().map(Message::Generator),
            Screen::Models(models) => models.view().map(Message::Models),
        };

        column![container(screen).height(Fill).padding(10), navbar].into()
    }
}
