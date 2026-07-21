//! Studio motion — tachyonfx transitions behind the reduce-motion setting.
//!
//! The spec's restraint budget (§10) allows exactly this much: a short
//! workspace fade when switching tabs and a completion sweep across the
//! preview caption when a print finishes developing. Both are one-shot
//! effects that expire on their own, so the manager drains naturally and a
//! disabled `MotionState` is a guaranteed no-op — two renders of the same
//! app state produce byte-identical buffers (contract-tested).
//!
//! Effects only ever recolor cells that normal rendering already painted;
//! they are never applied over image cells — Kitty/iTerm2/Sixel graphics
//! are escape-sequence passthrough and repainting them corrupts the
//! protocol stream. The workspace fade covers the content area during the
//! first ~160ms after a switch (image panels repaint on top next frame);
//! the completion sweep is confined to the caption row beneath the preview.

use std::time::Instant;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use tachyonfx::{fx, EffectManager, Interpolation, Motion};

/// Key type for the effect manager (a single shared lane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct FxLane;

/// Milliseconds for the workspace-switch fade (mockup: ~160ms ease).
const FADE_MS: u32 = 160;
/// Milliseconds for the completion sweep over the caption row.
const SWEEP_MS: u32 = 320;

pub struct MotionState {
    manager: EffectManager<FxLane>,
    last_frame: Instant,
    /// When false every trigger and process call is a no-op.
    pub enabled: bool,
}

impl MotionState {
    pub fn new(enabled: bool) -> Self {
        Self {
            manager: EffectManager::default(),
            last_frame: Instant::now(),
            enabled,
        }
    }

    /// Resolve the enabled flag from the environment and the persisted
    /// reduce-motion preference: `MOLD_TUI_NO_MOTION=1` always wins (UAT
    /// determinism), then the `tui.reduce_motion` settings key (default
    /// false — motion on).
    pub fn from_env_and_prefs() -> Self {
        if std::env::var("MOLD_TUI_NO_MOTION").is_ok_and(|v| v == "1") {
            return Self::new(false);
        }
        // Raw key read (the Settings redesign owns the const and the
        // toggle UI); default false = motion on. DB unavailable ⇒ motion
        // on, matching every other graceful-degradation path.
        let reduce = match mold_db::open_default() {
            Ok(Some(db)) => mold_db::settings::Settings::new(&db)
                .get_str("tui.reduce_motion")
                .ok()
                .flatten()
                .is_some_and(|v| v == "true" || v == "1"),
            _ => false,
        };
        Self::new(!reduce)
    }

    /// Fade the content region in from the workspace background — called
    /// after a workspace switch. `bg` is the theme background so the fade
    /// reads as the new surface developing in, not a flash.
    pub fn trigger_workspace_fade(&mut self, area: Rect, bg: Color) {
        if !self.enabled || area.width == 0 || area.height == 0 {
            return;
        }
        self.manager
            .add_effect(fx::fade_from(bg, bg, (FADE_MS, Interpolation::SineOut)).with_area(area));
    }

    /// Sweep the preview caption row in after a print finishes developing.
    /// The caller passes the caption row only — never image cells.
    pub fn trigger_completion_sweep(&mut self, caption_row: Rect, bg: Color) {
        if !self.enabled || caption_row.width == 0 || caption_row.height == 0 {
            return;
        }
        self.manager.add_effect(
            fx::sweep_in(
                Motion::LeftToRight,
                10,
                0,
                bg,
                (SWEEP_MS, Interpolation::SineOut),
            )
            .with_area(caption_row),
        );
    }

    /// Post-process the frame buffer — the last step of `ui::render`.
    /// Duration is wall-clock so slow terminals skip ahead instead of
    /// lagging; effects always complete within their real-time budget.
    pub fn process(&mut self, buf: &mut Buffer, area: Rect) {
        let elapsed = self.last_frame.elapsed();
        self.last_frame = Instant::now();
        if !self.enabled {
            return;
        }
        self.manager.process_effects(elapsed.into(), buf, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_motion_never_mutates_the_buffer() {
        let mut motion = MotionState::new(false);
        let area = Rect::new(0, 0, 20, 6);
        motion.trigger_workspace_fade(area, Color::Rgb(10, 10, 20));
        motion.trigger_completion_sweep(Rect::new(0, 5, 20, 1), Color::Rgb(10, 10, 20));

        let mut buf = Buffer::empty(area);
        buf.set_string(0, 0, "steady state", ratatui::style::Style::default());
        let before = buf.clone();
        motion.process(&mut buf, area);
        motion.process(&mut buf, area);
        assert_eq!(
            before, buf,
            "disabled motion must be a byte-identical no-op"
        );
    }

    #[test]
    fn enabled_motion_recolors_within_budget_then_expires() {
        let mut motion = MotionState::new(true);
        let area = Rect::new(0, 0, 20, 6);
        motion.trigger_workspace_fade(area, Color::Rgb(10, 10, 20));

        let mut buf = Buffer::empty(area);
        motion.process(&mut buf, area); // must not panic; effect active

        // After the fade budget passes, the manager drains and processing
        // becomes a no-op again.
        std::thread::sleep(std::time::Duration::from_millis(u64::from(FADE_MS) + 60));
        motion.process(&mut buf, area);
        let settled = buf.clone();
        motion.process(&mut buf, area);
        assert_eq!(settled, buf, "expired effects must leave the buffer alone");
    }

    #[test]
    fn zero_area_triggers_are_noops() {
        let mut motion = MotionState::new(true);
        motion.trigger_workspace_fade(Rect::new(0, 0, 0, 0), Color::Reset);
        motion.trigger_completion_sweep(Rect::new(0, 0, 0, 0), Color::Reset);
        let area = Rect::new(0, 0, 4, 2);
        let mut buf = Buffer::empty(area);
        let before = buf.clone();
        motion.process(&mut buf, area);
        assert_eq!(before, buf);
    }
}
