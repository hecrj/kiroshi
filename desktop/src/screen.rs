pub mod generator;
pub mod models;

pub use generator::Generator;
pub use models::Models;

pub enum Screen {
    Generator(Generator),
    Models(Models),
}
