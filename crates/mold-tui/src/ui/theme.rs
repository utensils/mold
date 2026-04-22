use ratatui::style::{Color, Modifier, Style};

/// A named colour palette preset.
///
/// The TUI ships seven presets matching the mold design system's
/// `ThemePicker` swatch list. Mocha is the default and maps directly to the
/// tokens in `colors_and_type.css`. The remaining presets provide a light
/// counterpart (Latte) and five alternate dark palettes for users who
/// prefer a different accent hue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemePreset {
    #[default]
    Mocha,
    Latte,
    Ristretto,
    Gruvbox,
    Tokyo,
    Nord,
    Dracula,
}

impl ThemePreset {
    /// All presets in display order (used by the Appearance swatch grid).
    /// Mirrors the seven-preset list in
    /// `mold-design-system/ui_kits/mold-tui/Primitives.jsx`.
    pub const ALL: [ThemePreset; 7] = [
        ThemePreset::Mocha,
        ThemePreset::Latte,
        ThemePreset::Ristretto,
        ThemePreset::Gruvbox,
        ThemePreset::Tokyo,
        ThemePreset::Nord,
        ThemePreset::Dracula,
    ];

    /// Short display label (title-case).
    pub fn label(self) -> &'static str {
        match self {
            ThemePreset::Mocha => "Mocha",
            ThemePreset::Latte => "Latte",
            ThemePreset::Ristretto => "Ristretto",
            ThemePreset::Gruvbox => "Gruvbox",
            ThemePreset::Tokyo => "Tokyo",
            ThemePreset::Nord => "Nord",
            ThemePreset::Dracula => "Dracula",
        }
    }

    /// Kebab-case slug, used for config / session persistence.
    pub fn slug(self) -> &'static str {
        match self {
            ThemePreset::Mocha => "mocha",
            ThemePreset::Latte => "latte",
            ThemePreset::Ristretto => "ristretto",
            ThemePreset::Gruvbox => "gruvbox",
            ThemePreset::Tokyo => "tokyo",
            ThemePreset::Nord => "nord",
            ThemePreset::Dracula => "dracula",
        }
    }

    /// Parse a slug back into a preset. Unknown slugs fall back to Mocha.
    pub fn from_slug(slug: &str) -> Self {
        match slug.trim().to_ascii_lowercase().as_str() {
            "latte" => ThemePreset::Latte,
            "ristretto" => ThemePreset::Ristretto,
            "gruvbox" => ThemePreset::Gruvbox,
            "tokyo" | "tokyonight" | "tokyo-night" => ThemePreset::Tokyo,
            "nord" => ThemePreset::Nord,
            "dracula" => ThemePreset::Dracula,
            _ => ThemePreset::Mocha,
        }
    }

    /// Swatch colour shown in the Appearance grid — usually the accent hue.
    pub fn swatch(self) -> Color {
        self.build().accent
    }

    /// Build a concrete [`Theme`] for this preset.
    pub fn build(self) -> Theme {
        match self {
            ThemePreset::Mocha => Theme::mocha(),
            ThemePreset::Latte => Theme::latte(),
            ThemePreset::Ristretto => Theme::ristretto(),
            ThemePreset::Gruvbox => Theme::gruvbox(),
            ThemePreset::Tokyo => Theme::tokyo(),
            ThemePreset::Nord => Theme::nord(),
            ThemePreset::Dracula => Theme::dracula(),
        }
    }
}

/// Colour palette for the TUI.
///
/// Each field is a semantic token — the role, not the hue. Presets populate
/// these fields from well-known palettes (Catppuccin, Gruvbox, Nord, …).
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
    /// Accent colour (selected items, active tabs).
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
        Self::mocha()
    }
}

impl Theme {
    /// Historical alias for [`Theme::mocha`]. Retained so external callers
    /// that reference `Theme::dark()` keep compiling.
    pub fn dark() -> Self {
        Self::mocha()
    }

    /// Catppuccin Mocha — the default dark theme and the design-system baseline.
    pub fn mocha() -> Self {
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

    /// Catppuccin Latte — a light counterpart to Mocha.
    pub fn latte() -> Self {
        Self {
            bg: Color::Rgb(239, 241, 245),           // #eff1f5
            surface: Color::Rgb(230, 233, 239),      // #e6e9ef
            border: Color::Rgb(204, 208, 218),       // #ccd0da
            border_focus: Color::Rgb(30, 102, 245),  // #1e66f5 (blue)
            text: Color::Rgb(76, 79, 105),           // #4c4f69
            text_dim: Color::Rgb(108, 111, 133),     // #6c6f85
            accent: Color::Rgb(30, 102, 245),        // #1e66f5
            success: Color::Rgb(64, 160, 43),        // #40a02b
            warning: Color::Rgb(223, 142, 29),       // #df8e1d
            error: Color::Rgb(210, 15, 57),          // #d20f39
            highlight: Color::Rgb(220, 224, 232),    // #dce0e8
            progress_fill: Color::Rgb(30, 102, 245), // #1e66f5
            progress_empty: Color::Rgb(220, 224, 232),
            tab_bg: Color::Rgb(230, 233, 239), // #e6e9ef
            tab_active: Color::Rgb(30, 102, 245),
            tab_inactive: Color::Rgb(108, 111, 133),
        }
    }

    /// Ristretto — warm espresso-toned dark theme from Monokai Pro.
    /// Accent is the signature `#fd6883` pink-red, matched to the swatch
    /// defined in the mold design system.
    pub fn ristretto() -> Self {
        Self {
            bg: Color::Rgb(44, 37, 37),               // #2c2525
            surface: Color::Rgb(64, 56, 56),          // #403838
            border: Color::Rgb(91, 83, 73),           // #5b5349
            border_focus: Color::Rgb(253, 104, 131),  // #fd6883 (pink-red)
            text: Color::Rgb(255, 241, 243),          // #fff1f3
            text_dim: Color::Rgb(114, 105, 106),      // #72696a
            accent: Color::Rgb(253, 104, 131),        // #fd6883
            success: Color::Rgb(173, 218, 120),       // #adda78
            warning: Color::Rgb(249, 204, 108),       // #f9cc6c
            error: Color::Rgb(243, 141, 112),         // #f38d70 (orange — distinct from accent)
            highlight: Color::Rgb(64, 56, 56),        // #403838
            progress_fill: Color::Rgb(253, 104, 131), // #fd6883
            progress_empty: Color::Rgb(64, 56, 56),
            tab_bg: Color::Rgb(35, 29, 29), // slightly deeper than bg
            tab_active: Color::Rgb(253, 104, 131),
            tab_inactive: Color::Rgb(114, 105, 106),
        }
    }

    /// Gruvbox Dark (hard).
    pub fn gruvbox() -> Self {
        Self {
            bg: Color::Rgb(29, 32, 33),              // #1d2021
            surface: Color::Rgb(50, 48, 47),         // #32302f
            border: Color::Rgb(80, 73, 69),          // #504945
            border_focus: Color::Rgb(131, 165, 152), // #83a598 (aqua)
            text: Color::Rgb(235, 219, 178),         // #ebdbb2
            text_dim: Color::Rgb(168, 153, 132),     // #a89984
            accent: Color::Rgb(131, 165, 152),       // #83a598
            success: Color::Rgb(184, 187, 38),       // #b8bb26
            warning: Color::Rgb(250, 189, 47),       // #fabd2f
            error: Color::Rgb(251, 73, 52),          // #fb4934
            highlight: Color::Rgb(60, 56, 54),       // #3c3836
            progress_fill: Color::Rgb(131, 165, 152),
            progress_empty: Color::Rgb(60, 56, 54),
            tab_bg: Color::Rgb(40, 40, 40), // #282828
            tab_active: Color::Rgb(131, 165, 152),
            tab_inactive: Color::Rgb(168, 153, 132),
        }
    }

    /// Tokyo Night (storm).
    pub fn tokyo() -> Self {
        Self {
            bg: Color::Rgb(26, 27, 38),              // #1a1b26
            surface: Color::Rgb(36, 40, 59),         // #24283b
            border: Color::Rgb(59, 66, 97),          // #3b4261
            border_focus: Color::Rgb(122, 162, 247), // #7aa2f7
            text: Color::Rgb(192, 202, 245),         // #c0caf5
            text_dim: Color::Rgb(130, 139, 184),     // #828bb8
            accent: Color::Rgb(122, 162, 247),       // #7aa2f7
            success: Color::Rgb(158, 206, 106),      // #9ece6a
            warning: Color::Rgb(224, 175, 104),      // #e0af68
            error: Color::Rgb(247, 118, 142),        // #f7768e
            highlight: Color::Rgb(41, 46, 66),       // #292e42
            progress_fill: Color::Rgb(122, 162, 247),
            progress_empty: Color::Rgb(41, 46, 66),
            tab_bg: Color::Rgb(22, 22, 30), // #16161e
            tab_active: Color::Rgb(122, 162, 247),
            tab_inactive: Color::Rgb(130, 139, 184),
        }
    }

    /// Nord — cold, muted palette.
    pub fn nord() -> Self {
        Self {
            bg: Color::Rgb(46, 52, 64),              // #2e3440
            surface: Color::Rgb(59, 66, 82),         // #3b4252
            border: Color::Rgb(67, 76, 94),          // #434c5e
            border_focus: Color::Rgb(136, 192, 208), // #88c0d0
            text: Color::Rgb(216, 222, 233),         // #d8dee9
            text_dim: Color::Rgb(136, 143, 161),     // #888fa1
            accent: Color::Rgb(136, 192, 208),       // #88c0d0
            success: Color::Rgb(163, 190, 140),      // #a3be8c
            warning: Color::Rgb(235, 203, 139),      // #ebcb8b
            error: Color::Rgb(191, 97, 106),         // #bf616a
            highlight: Color::Rgb(76, 86, 106),      // #4c566a
            progress_fill: Color::Rgb(136, 192, 208),
            progress_empty: Color::Rgb(59, 66, 82),
            tab_bg: Color::Rgb(36, 41, 51), // slightly deeper than bg
            tab_active: Color::Rgb(136, 192, 208),
            tab_inactive: Color::Rgb(136, 143, 161),
        }
    }

    /// Dracula — high-contrast purple/cyan.
    pub fn dracula() -> Self {
        Self {
            bg: Color::Rgb(40, 42, 54),              // #282a36
            surface: Color::Rgb(68, 71, 90),         // #44475a
            border: Color::Rgb(98, 114, 164),        // #6272a4
            border_focus: Color::Rgb(139, 233, 253), // #8be9fd (cyan)
            text: Color::Rgb(248, 248, 242),         // #f8f8f2
            text_dim: Color::Rgb(152, 160, 192),     // lighter version of comment
            accent: Color::Rgb(189, 147, 249),       // #bd93f9 (purple)
            success: Color::Rgb(80, 250, 123),       // #50fa7b
            warning: Color::Rgb(241, 250, 140),      // #f1fa8c
            error: Color::Rgb(255, 85, 85),          // #ff5555
            highlight: Color::Rgb(68, 71, 90),       // #44475a
            progress_fill: Color::Rgb(189, 147, 249),
            progress_empty: Color::Rgb(68, 71, 90),
            tab_bg: Color::Rgb(33, 34, 44), // deeper than surface
            tab_active: Color::Rgb(189, 147, 249),
            tab_inactive: Color::Rgb(152, 160, 192),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_presets_build() {
        for preset in ThemePreset::ALL {
            let theme = preset.build();
            // Sanity: distinct bg and text so we never render invisible copy.
            assert_ne!(theme.bg, theme.text, "{:?}", preset);
        }
    }

    #[test]
    fn slug_round_trips() {
        for preset in ThemePreset::ALL {
            assert_eq!(ThemePreset::from_slug(preset.slug()), preset);
        }
    }

    #[test]
    fn unknown_slug_falls_back_to_mocha() {
        assert_eq!(ThemePreset::from_slug("🐠"), ThemePreset::Mocha);
        assert_eq!(ThemePreset::from_slug(""), ThemePreset::Mocha);
    }

    #[test]
    fn all_seven_design_system_themes_ship() {
        // The design system (`ui_kits/mold-tui/Primitives.jsx:140`) lists
        // seven theme presets: Mocha / Latte / Ristretto / Gruvbox /
        // Tokyo / Nord / Dracula. Missing any of them is a gap in the
        // Appearance picker.
        let slugs: Vec<&str> = ThemePreset::ALL.iter().map(|p| p.slug()).collect();
        for expected in [
            "mocha",
            "latte",
            "ristretto",
            "gruvbox",
            "tokyo",
            "nord",
            "dracula",
        ] {
            assert!(
                slugs.contains(&expected),
                "theme `{expected}` is missing from ThemePreset::ALL",
            );
        }
    }

    #[test]
    fn ristretto_swatch_matches_design_system() {
        // The design-system swatch for Ristretto is `#fd6883` — an easy
        // regression guard in case the accent hex drifts.
        assert_eq!(
            ThemePreset::Ristretto.swatch(),
            Color::Rgb(0xfd, 0x68, 0x83),
        );
    }
}
