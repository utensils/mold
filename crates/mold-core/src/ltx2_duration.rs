//! Shared LTX-2.5 automatic-duration policy.
//!
//! Server admission reserves the maximum allowed frame count while inference
//! replaces it with the caption prediction. Both paths must use the same
//! bounds and causal-VAE grid arithmetic.

use anyhow::{ensure, Result};

pub const DEFAULT_MIN_SECONDS: f64 = 1.0;
pub const DEFAULT_MAX_SECONDS: f64 = 20.0;
pub const TEMPORAL_SCALE: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoDurationBounds {
    pub min_seconds: f64,
    pub max_seconds: f64,
}

impl Default for AutoDurationBounds {
    fn default() -> Self {
        Self {
            min_seconds: DEFAULT_MIN_SECONDS,
            max_seconds: DEFAULT_MAX_SECONDS,
        }
    }
}

impl AutoDurationBounds {
    pub fn validate(self) -> Result<Self> {
        ensure!(
            self.min_seconds.is_finite() && self.min_seconds > 0.0,
            "auto-duration minimum must be a finite positive number"
        );
        ensure!(
            self.max_seconds.is_finite() && self.max_seconds >= self.min_seconds,
            "auto-duration maximum must be finite and at least the minimum"
        );
        Ok(self)
    }
}

pub fn seconds_to_clamped_frames(
    seconds: f64,
    fps: u32,
    bounds: AutoDurationBounds,
) -> Result<u32> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0,
        "invalid predicted duration"
    );
    ensure!(fps > 0, "auto-duration fps must be positive");
    let bounds = bounds.validate()?;
    let min_frames = (bounds.min_seconds * f64::from(fps)).round().max(1.0) as u32;
    let max_frames = (bounds.max_seconds * f64::from(fps)).round() as u32;
    ensure!(
        max_frames >= min_frames,
        "auto-duration bounds admit no frames"
    );
    let raw = (seconds * f64::from(fps))
        .round()
        .clamp(f64::from(min_frames), f64::from(max_frames)) as u32;
    let mut frames = ((raw - 1) / TEMPORAL_SCALE) * TEMPORAL_SCALE + 1;
    if frames < min_frames {
        let snapped_up = frames.saturating_add(TEMPORAL_SCALE);
        frames = if snapped_up <= max_frames || snapped_up.abs_diff(raw) < frames.abs_diff(raw) {
            snapped_up
        } else {
            frames
        };
    }
    Ok(frames)
}

/// Worst-case frame count used before the caption prediction is available.
pub fn admission_frames(fps: u32) -> Result<u32> {
    seconds_to_clamped_frames(DEFAULT_MAX_SECONDS, fps, AutoDurationBounds::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_snaps_to_causal_grid_and_clamps() {
        let bounds = AutoDurationBounds::default();
        assert_eq!(seconds_to_clamped_frames(2.0, 24, bounds).unwrap(), 41);
        assert_eq!(seconds_to_clamped_frames(0.1, 24, bounds).unwrap(), 25);
        assert_eq!(seconds_to_clamped_frames(40.0, 24, bounds).unwrap(), 473);
        assert_eq!(admission_frames(24).unwrap(), 473);
    }

    #[test]
    fn duration_uses_nearest_grid_point_when_bounds_have_none() {
        assert_eq!(
            seconds_to_clamped_frames(
                1.0,
                24,
                AutoDurationBounds {
                    min_seconds: 1.0,
                    max_seconds: 1.0,
                },
            )
            .unwrap(),
            25
        );
    }
}
