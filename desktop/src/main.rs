use kiroshi;

mod icon;
mod widget;

use crate::kiroshi::detail;
use crate::kiroshi::image;
use crate::kiroshi::model;
use crate::kiroshi::{
    Detail, Error, Image, Model, Quality, Rectangle, Sampler, Seed, Server, Steps,
};
use crate::widget::{container, labeled_slider, logo, optic};

use iced::padding;
use iced::widget::{
    button, checkbox, column, hover, pick_list, row, stack, text, text_editor, text_input, tooltip,
    Column,
};
use iced::{color, Center, Color, Element, Fill, FillPortion, Font, Task, Theme};

fn main() -> iced::Result {
    iced::application(Kiroshi::title, Kiroshi::update, Kiroshi::view)
        .theme(Kiroshi::theme)
        .window_size((1024.0, 820.0))
        .font(icon::FONT)
        .font(widget::LOGO_FONT)
        .run_with(Kiroshi::new)
}

struct Kiroshi {
    models: Vec<Model>,
    model: Option<Model>,
    model_settings: model::Settings,
    quality: Quality,
    sampler: Sampler,
    seed: String,
    prompt: text_editor::Content,
    negative_prompt: text_editor::Content,
    image: Option<optic::Handle>,
    previous_image: Option<optic::Handle>,
    faces: Vec<Rectangle>,
    hands: Vec<Rectangle>,
    show_target_bounds: bool,
    face_detail_enabled: bool,
    hand_detail_enabled: bool,
    face_detail: Detail,
    hand_detail: Detail,
    active_target: Target,

    server: Option<Server>,
    theme: Theme,
}

#[derive(Debug, Clone)]
enum Message {
    ServerBooted(Result<Server, Error>),
    ModelsListed(Result<Vec<Model>, Error>),
    ModelSettingsFetched(Result<model::Settings, Error>),
    ModelSelected(Model),
    QualitySelected(Quality),
    SamplerSelected(Sampler),
    SeedChanged(String),
    RandomizeSeed,
    ToggleTargetBounds,
    TargetOpened(Target),
    DetailToggled(Target, bool),
    DetailChanged(Target, Detail),
    PromptEdited(text_editor::Action),
    NegativePromptEdited(text_editor::Action),
    Generate,
    ImageGenerated(Result<image::Generation, Error>),
}

impl Kiroshi {
    pub fn new() -> (Self, Task<Message>) {
        (
            Self {
                models: Vec::new(),
                model: None,
                model_settings: model::Settings::default(),
                quality: Quality::default(),
                sampler: Sampler::default(),
                prompt: text_editor::Content::new(),
                negative_prompt: text_editor::Content::new(),
                seed: Seed::random().to_string(),
                image: None,
                previous_image: None,
                faces: Vec::new(),
                hands: Vec::new(),
                show_target_bounds: false,
                face_detail_enabled: false,
                hand_detail_enabled: false,
                face_detail: Detail::default(),
                hand_detail: Detail::default(),
                active_target: Target::Face,
                server: None,
                theme: Theme::TokyoNight,
            },
            Task::batch([
                Task::perform(Server::run(), Message::ServerBooted),
                Task::perform(Model::list(), Message::ModelsListed),
                Task::perform(model::Settings::fetch(), Message::ModelSettingsFetched),
            ]),
        )
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ServerBooted(Ok(server)) => {
                self.server = Some(server);

                Task::none()
            }
            Message::ModelsListed(Ok(models)) => {
                self.models = models;

                Task::none()
            }
            Message::ModelSettingsFetched(Ok(settings)) => {
                self.model_settings = settings;

                Task::none()
            }
            Message::ModelSelected(model) => {
                self.model = Some(model);

                Task::none()
            }
            Message::QualitySelected(quality) => {
                self.quality = quality;

                Task::none()
            }
            Message::SamplerSelected(sampler) => {
                self.sampler = sampler;

                Task::none()
            }
            Message::SeedChanged(seed) => {
                self.seed = seed;

                Task::none()
            }
            Message::RandomizeSeed => {
                self.seed = Seed::random().to_string();

                Task::none()
            }
            Message::ToggleTargetBounds => {
                self.show_target_bounds = !self.show_target_bounds;

                Task::none()
            }
            Message::TargetOpened(target) => {
                self.active_target = target;

                Task::none()
            }
            Message::DetailToggled(target, enabled) => {
                match target {
                    Target::Face => {
                        self.face_detail_enabled = enabled;
                    }
                    Target::Hand => {
                        self.hand_detail_enabled = enabled;
                    }
                }

                Task::none()
            }
            Message::DetailChanged(target, detail) => {
                match target {
                    Target::Face => {
                        self.face_detail = detail;
                    }
                    Target::Hand => {
                        self.hand_detail = detail;
                    }
                }

                Task::none()
            }
            Message::PromptEdited(action) => {
                self.prompt.perform(action);

                Task::none()
            }
            Message::NegativePromptEdited(action) => {
                self.negative_prompt.perform(action);

                Task::none()
            }
            Message::Generate => {
                let Some(model) = &self.model else {
                    return Task::none();
                };

                self.previous_image = self.image.clone();
                self.faces.clear();
                self.hands.clear();

                let seed = {
                    let sanitized: String = self.seed.chars().filter(|c| c.is_numeric()).collect();

                    sanitized
                        .parse::<u64>()
                        .ok()
                        .map(Seed::from)
                        .unwrap_or_else(Seed::random)
                };

                self.seed = seed.value().to_string();

                let format_prompt = |prompt: &mut text_editor::Content, template| {
                    *prompt = text_editor::Content::with_text(
                        &prompt
                            .text()
                            .split(',')
                            .map(str::trim)
                            .filter(|line| !line.is_empty())
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );

                    let user_prompt = prompt.text().trim().replace("\n", ", ");

                    [user_prompt, template]
                        .join(", ")
                        .trim_start_matches(", ")
                        .to_owned()
                };

                let metadata = self.model_settings.get(&model);
                let prompt = format_prompt(&mut self.prompt, metadata.prompt_template);
                let negative_prompt =
                    format_prompt(&mut self.negative_prompt, metadata.negative_prompt_template);

                Task::run(
                    Image::generate(
                        image::Definition {
                            model: model.clone(),
                            prompt,
                            negative_prompt,
                            quality: self.quality,
                            sampler: self.sampler,
                            seed,
                            size: Image::DEFAULT_SIZE,
                            steps: Steps::default(),
                            face_detail: self.face_detail_enabled.then_some(self.face_detail),
                            hand_detail: self.hand_detail_enabled.then_some(self.hand_detail),
                            loras: Vec::new(),
                        },
                        Some(0.0),
                    ),
                    Message::ImageGenerated,
                )
            }
            Message::ImageGenerated(Ok(generation)) => {
                match generation {
                    image::Generation::Sampling { image, .. } => {
                        self.image = Some(optic::Handle::new(image));
                    }
                    image::Generation::Finished {
                        image,
                        faces,
                        hands,
                    } => {
                        self.image = Some(optic::Handle::new(image));
                        self.faces = faces;
                        self.hands = hands;
                    }
                }

                Task::none()
            }
            Message::ServerBooted(Err(error))
            | Message::ModelsListed(Err(error))
            | Message::ModelSettingsFetched(Err(error))
            | Message::ImageGenerated(Err(error)) => {
                dbg!(error);

                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<Message> {
        let image_and_controls = {
            let preview = if let Some(image) = &self.image {
                optic::original(image)
            } else {
                container(optic::placeholder())
                    .style(container::rounded_box)
                    .into()
            };

            let controls = {
                let models = with_label(
                    "Model",
                    pick_list(
                        self.models.as_slice(),
                        self.model.as_ref(),
                        Message::ModelSelected,
                    )
                    .width(Fill)
                    .placeholder("Select an image model..."),
                );

                let quality = with_label(
                    "Quality",
                    pick_list(Quality::ALL, Some(self.quality), Message::QualitySelected),
                );

                let sampler = with_label(
                    "Sampler",
                    pick_list(Sampler::ALL, Some(self.sampler), Message::SamplerSelected)
                        .width(Fill),
                );

                let seed = with_label(
                    "Seed",
                    stack![
                        text_input("Type a numeric seed...", &self.seed)
                            .font(Font::MONOSPACE)
                            .padding(padding::all(5).right(25))
                            .on_input(Message::SeedChanged),
                        container(
                            button(icon::refresh())
                                .on_press(Message::RandomizeSeed)
                                .style(button::text)
                        )
                        .align_right(Fill)
                        .center_y(Fill),
                    ],
                );

                let detailing = {
                    let tabs = {
                        let title = {
                            let show_detail_bounds = self.show_target_bounds;

                            let label = text("Detailing").size(12).font(Font::MONOSPACE);

                            let toggle_bounds = tooltip(
                                button(
                                    if show_detail_bounds {
                                        icon::visible()
                                    } else {
                                        icon::hidden()
                                    }
                                    .size(12),
                                )
                                .padding(0)
                                .on_press(Message::ToggleTargetBounds)
                                .style(move |theme, status| {
                                    let style = button::text(theme, status);

                                    button::Style {
                                        text_color: if show_detail_bounds {
                                            Color::WHITE
                                        } else {
                                            style.text_color
                                        },
                                        ..style
                                    }
                                }),
                                container(
                                    text(if show_detail_bounds {
                                        "Hide bounds"
                                    } else {
                                        "Show bounds"
                                    })
                                    .size(12),
                                )
                                .padding(5)
                                .style(container::dark),
                                tooltip::Position::Top,
                            );

                            row![label, toggle_bounds].spacing(10).align_y(Center)
                        };

                        row![title]
                            .extend(
                                [
                                    (Target::Face, self.face_detail_enabled),
                                    (Target::Hand, self.hand_detail_enabled),
                                ]
                                .map(|(target, is_enabled)| {
                                    if self.active_target == target {
                                        let toggle = checkbox(target.as_str(), is_enabled)
                                            .size(14)
                                            .text_size(14)
                                            .font(Font::MONOSPACE)
                                            .on_toggle(move |enabled| {
                                                Message::DetailToggled(target, enabled)
                                            });

                                        container(toggle)
                                            .padding(5)
                                            .center_x(Fill)
                                            .style(container::tab_header)
                                            .into()
                                    } else {
                                        let tab = row![]
                                            .push_maybe(
                                                is_enabled.then(|| icon::checkmark().size(14)),
                                            )
                                            .push(
                                                text(target.as_str())
                                                    .size(14)
                                                    .font(Font::MONOSPACE),
                                            )
                                            .spacing(10);

                                        button(container(tab).center_x(Fill))
                                            .padding(5)
                                            .style(move |theme, status| {
                                                use iced::border;

                                                let style = button::secondary(theme, status);

                                                button::Style {
                                                    background: style.background.map(
                                                        |background| background.scale_alpha(0.5),
                                                    ),
                                                    border: border::rounded(border::top(2)),
                                                    ..style
                                                }
                                            })
                                            .on_press(Message::TargetOpened(target))
                                            .into()
                                    }
                                }),
                            )
                            .spacing(10)
                            .align_y(Center)
                    };

                    let active_detail = match self.active_target {
                        Target::Face => self.face_detail,
                        Target::Hand => self.hand_detail,
                    };

                    let form = container(
                        detail_controls(active_detail)
                            .map(move |detail| Message::DetailChanged(self.active_target, detail)),
                    )
                    .padding(10)
                    .style(container::tab_content);

                    column![tabs, form]
                };

                let generate = button(text("Generate").width(Fill).center())
                    .on_press_maybe(self.model.is_some().then_some(Message::Generate));

                column![
                    row![models, quality].spacing(10),
                    row![sampler, seed].spacing(10),
                    detailing,
                    generate
                ]
                .spacing(10)
            };

            let controls = container(
                container(controls)
                    .padding(10)
                    .style(container::translucent),
            );

            if let Some(handle) = &self.image {
                let small_preview = container(
                    container(if self.show_target_bounds {
                        stack![
                            optic::small(handle),
                            targets(handle.image(), &self.faces, &self.hands)
                        ]
                        .into()
                    } else if let Some(previous_image) = &self.previous_image {
                        hover(optic::small(handle), optic::small(previous_image))
                    } else {
                        optic::small(handle)
                    })
                    .style(container::translucent)
                    .padding(10),
                )
                .align_right(Fill);

                hover(
                    preview,
                    container(column![small_preview, controls].spacing(10))
                        .padding(10)
                        .align_bottom(Fill),
                )
            } else {
                stack![preview, controls.center_y(Fill).padding(10)].into()
            }
        };

        let prompts = column![
            with_label(
                "Prompt",
                text_editor(&self.prompt)
                    .on_action(Message::PromptEdited)
                    .font(Font::MONOSPACE)
                    .placeholder("Describe what you want to see here...")
                    .padding(10)
                    .height(Fill)
            )
            .height(FillPortion(3)),
            with_label(
                "Negative Prompt",
                text_editor(&self.negative_prompt)
                    .on_action(Message::NegativePromptEdited)
                    .font(Font::MONOSPACE)
                    .placeholder("Describe what you want to unsee here...")
                    .height(Fill)
                    .padding(10)
            ),
        ]
        .spacing(10);

        let content = row![container(image_and_controls).center_y(Fill), prompts]
            .spacing(10)
            .padding(10);

        let navbar = container(row![logo(20)].padding([2, 10]))
            .width(Fill)
            .padding(padding::top(5))
            .style(|_theme| container::Style::default().background(color!(0x000000, 0.5)));

        column![content, navbar].into()
    }

    pub fn title(&self) -> String {
        "Kiroshi".to_owned()
    }

    pub fn theme(&self) -> Theme {
        self.theme.clone()
    }
}

fn with_label<'a>(label: &'a str, element: impl Into<Element<'a, Message>>) -> Column<'a, Message> {
    column![text(label).size(12).font(Font::MONOSPACE), element.into()].spacing(5)
}

fn detail_controls(detail: Detail) -> Element<'static, Detail> {
    let strength = labeled_slider(
        "Strength",
        detail::Strength::RANGE,
        detail.strength,
        move |strength| Detail { strength, ..detail },
        detail::Strength::to_string,
    );

    let padding = labeled_slider(
        "Padding",
        detail::Padding::RANGE,
        detail.padding,
        move |padding| Detail { padding, ..detail },
        detail::Padding::to_string,
    );

    let area = labeled_slider(
        "Max. Area",
        0..=Image::DEFAULT_SIZE.width.pow(2),
        detail.max_area.map(detail::Area::value).unwrap_or_default(),
        move |max_area| Detail {
            max_area: detail::Area::parse(max_area),
            ..detail
        },
        |value| {
            if *value > 0 {
                format!("{value}px²")
            } else {
                "No Limit".to_owned()
            }
        },
    );

    column![strength, padding, area].spacing(10).into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Target {
    Face,
    Hand,
}

impl Target {
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Face => "Face",
            Target::Hand => "Hands",
        }
    }
}

fn targets<'a>(
    image: &'a Image,
    faces: &'a [Rectangle],
    hands: &'a [Rectangle],
) -> Element<'a, Message> {
    use iced::mouse;
    use iced::widget::canvas;
    use iced::{Bottom, Point, Renderer, Size, Theme, Vector};

    struct Targets<'a> {
        image: &'a Image,
        faces: &'a [Rectangle],
        hands: &'a [Rectangle],
    }

    impl canvas::Program<Message> for Targets<'_> {
        type State = ();

        fn draw(
            &self,
            _state: &Self::State,
            renderer: &Renderer,
            theme: &Theme,
            bounds: iced::Rectangle,
            _cursor: mouse::Cursor,
        ) -> Vec<canvas::Geometry> {
            let size = bounds.size();
            let scale = size.width as f32 / self.image.size.width as f32;
            let palette = theme.extended_palette();

            let mut frame = canvas::Frame::new(renderer, size);
            frame.scale(scale);

            let draw_target =
                |frame: &mut canvas::Frame, label: &str, bounds: &Rectangle, color| {
                    frame.fill_rectangle(
                        Point::new(bounds.x, bounds.y),
                        Size::new(bounds.width, bounds.height),
                        color,
                    );

                    for (color, offset) in [
                        (Color::BLACK, Vector::new(5.0, 5.0)),
                        (Color::WHITE, Vector::ZERO),
                    ] {
                        frame.fill_text(canvas::Text {
                            content: label.to_owned(),
                            position: Point::new(bounds.x, bounds.y) + offset,
                            vertical_alignment: Bottom,
                            color,
                            size: (12.0 / scale).into(),
                            font: Font::MONOSPACE,
                            ..canvas::Text::default()
                        });
                    }
                };

            for face in self.faces {
                draw_target(
                    &mut frame,
                    "Face",
                    face,
                    palette.background.base.color.scale_alpha(0.4),
                );
            }

            for hand in self.hands {
                draw_target(
                    &mut frame,
                    "Hand",
                    hand,
                    palette.primary.base.color.scale_alpha(0.3),
                );
            }

            vec![frame.into_geometry()]
        }
    }

    canvas(Targets {
        image,
        faces,
        hands,
    })
    .width(Fill)
    .height(Fill)
    .into()
}
