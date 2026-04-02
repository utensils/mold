use ratatui::style::{Color, Modifier, Style};

/// Color palette for the TUI.
///
/// Defaults to a dark theme inspired by Catppuccin Mocha.
/// Future: load from `[tui.theme]` in config.toml.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Main background.
    pub bg: Color,
    /// Elevated surface (panels, cards).
    pub surface: Color,
    /// Panel borders.
    pub border: Color,
    /// Focused panel border.
    pub border_focus: Color,
    /// Primary text.
    pub text: Color,
    /// Dimmed / secondary text.
    pub text_dim: Color,
    /// Accent color (selected items, active tabs).
    pub accent: Color,
    /// Success indicators (checkmarks, completed stages).
    pub success: Color,
    /// Warning indicators.
    pub warning: Color,
    /// Error indicators.
    pub error: Color,
    /// Highlighted / selected row background.
    pub highlight: Color,
    /// Progress bar filled portion.
    pub progress_fill: Color,
    /// Progress bar empty portion.
    pub progress_empty: Color,
    /// Tab bar background.
    pub tab_bg: Color,
    /// Active tab text.
    pub tab_active: Color,
    /// Inactive tab text.
    pub tab_inactive: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme — Catppuccin Mocha inspired.
    pub fn dark() -> Self {
        Self {
            bg: Color::Rgb(30, 30, 46),               // #1e1e2e
            surface: Color::Rgb(49, 50, 68),          // #313244
            border: Color::Rgb(69, 71, 90),           // #45475a
            border_focus: Color::Rgb(137, 180, 250),  // #89b4fa (blue)
            text: Color::Rgb(205, 214, 244),          // #cdd6f4
            text_dim: Color::Rgb(127, 132, 156),      // #7f849c
            accent: Color::Rgb(137, 180, 250),        // #89b4fa (blue)
            success: Color::Rgb(166, 227, 161),       // #a6e3a1 (green)
            warning: Color::Rgb(249, 226, 175),       // #f9e2af (yellow)
            error: Color::Rgb(243, 139, 168),         // #f38ba8 (red)
            highlight: Color::Rgb(69, 71, 90),        // #45475a
            progress_fill: Color::Rgb(137, 180, 250), // #89b4fa
            progress_empty: Color::Rgb(49, 50, 68),   // #313244
            tab_bg: Color::Rgb(24, 24, 37),           // #181825
            tab_active: Color::Rgb(137, 180, 250),    // #89b4fa
            tab_inactive: Color::Rgb(127, 132, 156),  // #7f849c
        }
    }

    // ── Style helpers ───────────────────────────────────────────────

    /// Base style applied to the entire frame background.
    pub fn base(&self) -> Style {
        Style::default().bg(self.bg).fg(self.text)
    }

    /// Style for panel/block borders.
    pub fn border(&self) -> Style {
        Style::default().fg(self.border)
    }

    /// Style for focused panel borders.
    pub fn border_focused(&self) -> Style {
        Style::default().fg(self.border_focus)
    }

    /// Style for block titles.
    pub fn title(&self) -> Style {
        Style::default().fg(self.text).add_modifier(Modifier::BOLD)
    }

    /// Style for focused block titles.
    pub fn title_focused(&self) -> Style {
        Style::default()
            .fg(self.border_focus)
            .add_modifier(Modifier::BOLD)
    }

    /// Dimmed secondary text.
    pub fn dim(&self) -> Style {
        Style::default().fg(self.text_dim)
    }

    /// Style for parameter labels in the form.
    pub fn param_label(&self) -> Style {
        Style::default().fg(self.text_dim)
    }

    /// Style for parameter values.
    pub fn param_value(&self) -> Style {
        Style::default().fg(self.text)
    }

    /// Style for the currently focused parameter row.
    pub fn param_selected(&self) -> Style {
        Style::default().fg(self.accent).bg(self.highlight)
    }

    /// Style for success indicators (checkmarks).
    pub fn success(&self) -> Style {
        Style::default().fg(self.success)
    }

    /// Style for error text.
    pub fn error(&self) -> Style {
        Style::default().fg(self.error)
    }

    /// Style for warning text.
    pub fn warning(&self) -> Style {
        Style::default().fg(self.warning)
    }

    /// Active tab style.
    pub fn tab_active(&self) -> Style {
        Style::default()
            .fg(self.tab_active)
            .add_modifier(Modifier::BOLD)
    }

    /// Inactive tab style.
    pub fn tab_inactive(&self) -> Style {
        Style::default().fg(self.tab_inactive)
    }

    /// Status bar / shortcut hint style.
    pub fn status_bar(&self) -> Style {
        Style::default().bg(self.tab_bg).fg(self.text_dim)
    }

    /// Status bar key highlight.
    pub fn status_key(&self) -> Style {
        Style::default()
            .bg(self.tab_bg)
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
    }

    /// Highlighted list row.
    pub fn list_selected(&self) -> Style {
        Style::default().bg(self.highlight).fg(self.text)
    }

    /// Progress gauge filled style.
    pub fn progress_filled(&self) -> Style {
        Style::default().fg(self.progress_fill)
    }

    /// Progress gauge empty style.
    pub fn progress_empty(&self) -> Style {
        Style::default().fg(self.progress_empty)
    }

    /// Popup/overlay border style.
    pub fn popup_border(&self) -> Style {
        Style::default().fg(self.accent)
    }

    /// Popup background style.
    pub fn popup_bg(&self) -> Style {
        Style::default().bg(self.surface).fg(self.text)
    }
}
