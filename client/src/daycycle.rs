//! Day/night cycle: a time-of-day clock feeding the renderer's sun.
//!
//! The renderer's `sun_state(angle, rotation)` convention: angle 0 = sunrise
//! on the horizon, 0.25 = noon, 0.5 = sunset, 0.75 = midnight (fraction of a
//! full day, directly the cycle fraction). The client advances a day length
//! in real seconds and hands the renderer `day_fraction`.

/// Default day length: 20 real minutes like vanilla (1200 s).
pub const DEFAULT_DAY_SECONDS: f32 = 1200.0;
/// Start of day: just after sunrise.
pub const START_FRACTION: f32 = 0.02;

#[derive(Debug, Clone, Copy)]
pub struct DayCycle {
    /// 0..1 through the day (0 sunrise → 0.25 noon → 0.5 sunset → 0.75 mid).
    pub fraction: f32,
    pub seconds_per_day: f32,
}

impl Default for DayCycle {
    fn default() -> Self {
        DayCycle {
            fraction: START_FRACTION,
            seconds_per_day: DEFAULT_DAY_SECONDS,
        }
    }
}

impl DayCycle {
    pub fn new(seconds_per_day: f32) -> DayCycle {
        DayCycle {
            fraction: START_FRACTION,
            seconds_per_day: seconds_per_day.max(1.0),
        }
    }

    pub fn advance(&mut self, dt: f32) {
        self.fraction = (self.fraction + dt / self.seconds_per_day).rem_euclid(1.0);
    }

    /// Jump the clock to an absolute fraction (world load). Also accepts
    /// `None` to freeze the sun (a saved world without a day cycle).
    pub fn set_fraction(&mut self, fraction: Option<f32>) {
        if let Some(f) = fraction {
            self.fraction = f.rem_euclid(1.0);
        }
    }

    /// Is it night (sun below the horizon, elev < 0)? Sunset (0.5) itself is
    /// elevation 0 — still day; night starts just after.
    pub fn is_night(&self) -> bool {
        self.fraction > 0.5
    }

    /// Sun elevation sin at this fraction (matches the renderer's great-
    /// circle path with zero rotation): elevation = sin(2π·fraction).
    pub fn elevation_sin(&self) -> f32 {
        (self.fraction * std::f32::consts::TAU).sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_day_wraps_and_starts_at_sunrise() {
        let mut day = DayCycle::new(100.0);
        assert!((day.fraction - START_FRACTION).abs() < 1e-6);
        day.advance(25.0); // quarter day
        assert!((day.fraction - (START_FRACTION + 0.25)).abs() < 1e-6);
        day.advance(100.0); // full wrap: a whole day later, same phase, in [0,1)
        assert!(
            (day.fraction - (START_FRACTION + 0.25)).abs() < 1e-5,
            "a full-day advance returns to the same phase, got {}",
            day.fraction
        );
        assert!(day.fraction < 1.0, "fraction wraps into [0, 1)");
    }

    #[test]
    fn noon_has_the_sun_at_the_zenith() {
        let mut day = DayCycle::default();
        day.fraction = 0.25;
        assert!(
            (day.elevation_sin() - 1.0).abs() < 1e-5,
            "noon elevation = 1"
        );
        assert!(!day.is_night());
        day.fraction = 0.0;
        assert!(day.elevation_sin().abs() < 1e-5, "sunrise elevation = 0");
        day.fraction = 0.5;
        assert!(day.elevation_sin().abs() < 1e-5, "sunset elevation = 0");
        assert!(!day.is_night(), "sunset edge still counts as day");
        day.fraction = 0.75;
        assert!(
            (day.elevation_sin() + 1.0).abs() < 1e-5,
            "midnight elevation = -1"
        );
        assert!(day.is_night());
    }

    #[test]
    fn day_length_is_respected() {
        // A 60-second day advances 1/60 of the cycle per second.
        let mut day = DayCycle::new(60.0);
        day.advance(1.0);
        assert!((day.fraction - (START_FRACTION + 1.0 / 60.0)).abs() < 1e-6);
    }
}
