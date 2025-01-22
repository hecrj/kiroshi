pub fn main() {
    println!("cargo::rerun-if-changed=assets/joi-icons.toml");
    iced_fontello::build("fonts/kiroshi-icons.toml").expect("Build icons font");
}
