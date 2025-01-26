use iced::time::{Duration, Instant};

pub use lilt::Easing;

#[derive(Debug, Clone)]
pub struct Animation<T>
where
    T: Clone + Copy + PartialEq + lilt::FloatRepresentable,
{
    raw: lilt::Animated<T, Instant>,
}

impl<T> Animation<T>
where
    T: Clone + Copy + PartialEq + lilt::FloatRepresentable,
{
    pub fn new(value: T) -> Self {
        Self {
            raw: lilt::Animated::new(value),
        }
    }

    pub fn easing(mut self, easing: Easing) -> Self {
        self.raw = self.raw.easing(easing);
        self
    }

    pub fn _very_quick(self) -> Self {
        self.duration(Duration::from_millis(100))
    }

    pub fn _quick(self) -> Self {
        self.duration(Duration::from_millis(200))
    }

    pub fn slow(self) -> Self {
        self.duration(Duration::from_millis(400))
    }

    pub fn very_slow(self) -> Self {
        self.duration(Duration::from_millis(500))
    }

    pub fn duration(mut self, duration: Duration) -> Self {
        self.raw = self.raw.duration(duration.as_secs_f32() * 1_000.0);
        self
    }

    pub fn _delay(mut self, duration: Duration) -> Self {
        self.raw = self.raw.delay(duration.as_secs_f64() as f32 * 1000.0);
        self
    }

    pub fn go(mut self, new_value: T) -> Self {
        self.go_mut(new_value);
        self
    }

    pub fn go_mut(&mut self, new_value: T) {
        self.raw.transition(new_value, Instant::now());
    }

    pub fn _interpolate_with<I>(&self, f: impl Fn(T) -> I, at: Instant) -> I
    where
        I: lilt::Interpolable,
    {
        self.raw.animate(f, at)
    }

    pub fn in_progress(&self, at: Instant) -> bool {
        self.raw.in_progress(at)
    }
}

impl Animation<bool> {
    pub fn interpolate<I>(&self, start: I, end: I, at: Instant) -> I
    where
        I: lilt::Interpolable + Clone,
    {
        self.raw.animate_bool(start, end, at)
    }
}
