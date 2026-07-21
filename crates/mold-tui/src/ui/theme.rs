use ratatui::style::{Color, Modifier, Style};

/// A named colour palette preset.
///
/// The TUI ships eleven presets: the four Mold Studio themes from the design
/// spec (`docs/design/mold-studio-spec.html` §05, mockup
/// `docs/design/mold-tui-proposed.html`) followed by the seven legacy
/// palettes. Studio Dark is the default and matches the desktop/web Mold
/// dark tokens; Catppuccin Mocha is retained as the single-blue baseline the
/// TUI shipped before the Studio redesign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemePreset {
    #[default]
    StudioDark,
    StudioLight,
    SafelightDark,
    SafelightLight,
    Mocha,
    Latte,
    Ristretto,
    Gruvbox,
    Tokyo,
    Nord,
    Dracula,
}

impl ThemePreset {
    /// All presets in display order (used by the Appearance swatch grid) —
    /// Studio family first, then the legacy palettes in their historical
    /// order.
    pub const ALL: [ThemePreset; 11] = [
        ThemePreset::StudioDark,
        ThemePreset::StudioLight,
        ThemePreset::SafelightDark,
        ThemePreset::SafelightLight,
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
            ThemePreset::StudioDark => "Studio Dark",
            ThemePreset::StudioLight => "Studio Light",
            ThemePreset::SafelightDark => "Safelight Dark",
            ThemePreset::SafelightLight => "Safelight Light",
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
            ThemePreset::StudioDark => "studio-dark",
            ThemePreset::StudioLight => "studio-light",
            ThemePreset::SafelightDark => "safelight-dark",
            ThemePreset::SafelightLight => "safelight-light",
            ThemePreset::Mocha => "mocha",
            ThemePreset::Latte => "latte",
            ThemePreset::Ristretto => "ristretto",
            ThemePreset::Gruvbox => "gruvbox",
            ThemePreset::Tokyo => "tokyo",
            ThemePreset::Nord => "nord",
            ThemePreset::Dracula => "dracula",
        }
    }

    /// Parse a slug back into a preset. Unknown slugs fall back to the
    /// Studio Dark default. `studio` and `safelight` are accepted as
    /// input-only aliases for the dark variants (the design mockup uses the
    /// bare names for its dark presets).
    pub fn from_slug(slug: &str) -> Self {
        match slug.trim().to_ascii_lowercase().as_str() {
            "studio-dark" | "studio" => ThemePreset::StudioDark,
            "studio-light" => ThemePreset::StudioLight,
            "safelight-dark" | "safelight" => ThemePreset::SafelightDark,
            "safelight-light" => ThemePreset::SafelightLight,
            "mocha" => ThemePreset::Mocha,
            "latte" => ThemePreset::Latte,
            "ristretto" => ThemePreset::Ristretto,
            "gruvbox" => ThemePreset::Gruvbox,
            "tokyo" | "tokyonight" | "tokyo-night" => ThemePreset::Tokyo,
            "nord" => ThemePreset::Nord,
            "dracula" => ThemePreset::Dracula,
            _ => ThemePreset::StudioDark,
        }
    }

    /// Swatch colour shown in the Appearance grid — usually the accent hue.
    pub fn swatch(self) -> Color {
        self.build().accent
    }

    /// Build a concrete [`Theme`] for this preset.
    pub fn build(self) -> Theme {
        match self {
            ThemePreset::StudioDark => Theme::studio_dark(),
            ThemePreset::StudioLight => Theme::studio_light(),
            ThemePreset::SafelightDark => Theme::safelight_dark(),
            ThemePreset::SafelightLight => Theme::safelight_light(),
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

/// Blend `fg` over `bg` at `alpha_pct` percent opacity, returning a solid
/// colour. Terminal cells can't express alpha, so the CSS `rgba(...)`
/// selection backgrounds from the design mockup are pre-blended over the
/// theme background with this helper — keeping the CSS provenance executable
/// rather than a comment.
pub(crate) fn blend(fg: (u8, u8, u8), bg: (u8, u8, u8), alpha_pct: u16) -> Color {
    let ch = |f: u8, b: u8| -> u8 {
        ((u16::from(f) * alpha_pct + u16::from(b) * (100 - alpha_pct) + 50) / 100) as u8
    };
    Color::Rgb(ch(fg.0, bg.0), ch(fg.1, bg.1), ch(fg.2, bg.2))
}

/// Colour palette for the TUI.
///
/// Each field is a semantic token — the role, not the hue — following the
/// terminal token vocabulary of the Mold Studio TUI mockup. The design
/// system's dual-accent model maps `safelight → accent` (focus, selection,
/// primary action) and `halide → info` (live state, telemetry, spinner).
/// The mockup's `desk` token has no terminal equivalent (the emulator owns
/// everything outside the frame) and is deliberately absent.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Main background.
    pub bg: Color,
    /// Chrome planes: tab strip, activity strip, status bar.
    pub frame: Color,
    /// Elevated surface (panels, cards, popups).
    pub surface: Color,
    /// Second elevation (palette rows, selected cells, gauge troughs).
    pub surface2: Color,
    /// Panel borders.
    pub border: Color,
    /// Focused panel border. Invariant: equals `accent` in every preset.
    pub border_focus: Color,
    /// Primary text.
    pub text: Color,
    /// Dimmed / secondary text.
    pub text_dim: Color,
    /// Tertiary text: hints, placeholders.
    pub faint: Color,
    /// Primary accent (focus, selection, primary action) — "safelight".
    pub accent: Color,
    /// Secondary accent (info, live state, spinner) — "halide".
    pub info: Color,
    /// Success indicators (checkmarks, completed stages).
    pub success: Color,
    /// Warning indicators.
    pub warning: Color,
    /// Error indicators.
    pub error: Color,
    /// Highlighted / selected row background (pre-blended selection tint).
    pub highlight: Color,
    /// Progress bar filled portion.
    pub progress_fill: Color,
    /// Progress bar empty portion.
    pub progress_empty: Color,
    /// Active tab text.
    pub tab_active: Color,
    /// Inactive tab text.
    pub tab_inactive: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self::studio_dark()
    }
}

impl Theme {
    /// Historical alias retained so external callers that reference
    /// `Theme::dark()` keep compiling. Now points at the Studio Dark
    /// default.
    pub fn dark() -> Self {
        Self::studio_dark()
    }

    /// Studio Dark — the Mold-family dark theme and the TUI default.
    pub fn studio_dark() -> Self {
        Self {
            bg: Color::Rgb(0x13, 0x11, 0x1d),
            frame: Color::Rgb(0x0c, 0x0a, 0x14),
            surface: Color::Rgb(0x22, 0x1c, 0x34),
            surface2: Color::Rgb(0x2e, 0x27, 0x43),
            border: Color::Rgb(0x3a, 0x33, 0x52),
            border_focus: Color::Rgb(0xe8, 0x79, 0xf9),
            text: Color::Rgb(0xf3, 0xef, 0xfb),
            text_dim: Color::Rgb(0x90, 0x89, 0xa8),
            faint: Color::Rgb(0x64, 0x5d, 0x7c),
            accent: Color::Rgb(0xe8, 0x79, 0xf9),
            info: Color::Rgb(0x5c, 0xd0, 0xff),
            success: Color::Rgb(0x6e, 0xe7, 0xa5),
            warning: Color::Rgb(0xf5, 0xc4, 0x51),
            error: Color::Rgb(0xfb, 0x71, 0x85),
            highlight: blend((0xe8, 0x79, 0xf9), (0x13, 0x11, 0x1d), 15),
            progress_fill: Color::Rgb(0xe8, 0x79, 0xf9),
            progress_empty: Color::Rgb(0x2e, 0x27, 0x43),
            tab_active: Color::Rgb(0xe8, 0x79, 0xf9),
            tab_inactive: Color::Rgb(0x90, 0x89, 0xa8),
        }
    }

    /// Studio Light — the Mold-family light theme.
    pub fn studio_light() -> Self {
        Self {
            bg: Color::Rgb(0xf6, 0xf4, 0xfb),
            frame: Color::Rgb(0xe5, 0xe0, 0xf1),
            surface: Color::Rgb(0xef, 0xea, 0xf8),
            surface2: Color::Rgb(0xe3, 0xdc, 0xf2),
            border: Color::Rgb(0xcf, 0xc6, 0xe4),
            border_focus: Color::Rgb(0xa2, 0x1c, 0xaf),
            text: Color::Rgb(0x22, 0x14, 0x30),
            text_dim: Color::Rgb(0x5f, 0x56, 0x73),
            faint: Color::Rgb(0x9a, 0x90, 0xb2),
            accent: Color::Rgb(0xa2, 0x1c, 0xaf),
            info: Color::Rgb(0x0e, 0x74, 0x90),
            success: Color::Rgb(0x0f, 0x8a, 0x54),
            warning: Color::Rgb(0xa0, 0x6a, 0x12),
            error: Color::Rgb(0xb4, 0x23, 0x3b),
            highlight: blend((0xa2, 0x1c, 0xaf), (0xf6, 0xf4, 0xfb), 12),
            progress_fill: Color::Rgb(0xa2, 0x1c, 0xaf),
            progress_empty: Color::Rgb(0xe3, 0xdc, 0xf2),
            tab_active: Color::Rgb(0xa2, 0x1c, 0xaf),
            tab_inactive: Color::Rgb(0x5f, 0x56, 0x73),
        }
    }

    /// Safelight Dark — the warm darkroom-family dark theme.
    pub fn safelight_dark() -> Self {
        Self {
            bg: Color::Rgb(0x17, 0x12, 0x10),
            frame: Color::Rgb(0x12, 0x10, 0x0b),
            surface: Color::Rgb(0x24, 0x1c, 0x15),
            surface2: Color::Rgb(0x33, 0x26, 0x19),
            border: Color::Rgb(0x46, 0x3a, 0x2a),
            border_focus: Color::Rgb(0xf7, 0x94, 0x33),
            text: Color::Rgb(0xf1, 0xe8, 0xda),
            text_dim: Color::Rgb(0xa2, 0x93, 0x7d),
            faint: Color::Rgb(0x6f, 0x63, 0x53),
            accent: Color::Rgb(0xf7, 0x94, 0x33),
            info: Color::Rgb(0x8f, 0xb4, 0xc4),
            success: Color::Rgb(0x8f, 0xd3, 0x9a),
            warning: Color::Rgb(0xf5, 0xc4, 0x51),
            error: Color::Rgb(0xe5, 0x71, 0x5a),
            highlight: blend((0xf7, 0x94, 0x33), (0x17, 0x12, 0x10), 15),
            progress_fill: Color::Rgb(0xf7, 0x94, 0x33),
            progress_empty: Color::Rgb(0x33, 0x26, 0x19),
            tab_active: Color::Rgb(0xf7, 0x94, 0x33),
            tab_inactive: Color::Rgb(0xa2, 0x93, 0x7d),
        }
    }

    /// Safelight Light — the warm darkroom-family light theme.
    pub fn safelight_light() -> Self {
        Self {
            bg: Color::Rgb(0xf7, 0xf1, 0xe6),
            frame: Color::Rgb(0xec, 0xe2, 0xcf),
            surface: Color::Rgb(0xf0, 0xe7, 0xd5),
            surface2: Color::Rgb(0xe6, 0xdc, 0xc7),
            border: Color::Rgb(0xd8, 0xcb, 0xb0),
            border_focus: Color::Rgb(0x9b, 0x4d, 0x00),
            text: Color::Rgb(0x23, 0x1d, 0x16),
            text_dim: Color::Rgb(0x6b, 0x5f, 0x4d),
            faint: Color::Rgb(0x9a, 0x8c, 0x74),
            accent: Color::Rgb(0x9b, 0x4d, 0x00),
            info: Color::Rgb(0x3f, 0x65, 0x70),
            success: Color::Rgb(0x2f, 0x7d, 0x4f),
            warning: Color::Rgb(0xa0, 0x6a, 0x12),
            error: Color::Rgb(0xa9, 0x36, 0x29),
            highlight: blend((0x9b, 0x4d, 0x00), (0xf7, 0xf1, 0xe6), 13),
            progress_fill: Color::Rgb(0x9b, 0x4d, 0x00),
            progress_empty: Color::Rgb(0xe6, 0xdc, 0xc7),
            tab_active: Color::Rgb(0x9b, 0x4d, 0x00),
            tab_inactive: Color::Rgb(0x6b, 0x5f, 0x4d),
        }
    }

    /// Catppuccin Mocha — the pre-Studio default, retained as the
    /// single-blue baseline (info = sky, per the design mockup).
    pub fn mocha() -> Self {
        Self {
            bg: Color::Rgb(30, 30, 46),               // #1e1e2e
            frame: Color::Rgb(24, 24, 37),            // #181825
            surface: Color::Rgb(49, 50, 68),          // #313244
            surface2: Color::Rgb(69, 71, 90),         // #45475a
            border: Color::Rgb(69, 71, 90),           // #45475a
            border_focus: Color::Rgb(137, 180, 250),  // #89b4fa (blue)
            text: Color::Rgb(205, 214, 244),          // #cdd6f4
            text_dim: Color::Rgb(127, 132, 156),      // #7f849c
            faint: Color::Rgb(108, 112, 134),         // #6c7086 (overlay0)
            accent: Color::Rgb(137, 180, 250),        // #89b4fa (blue)
            info: Color::Rgb(137, 220, 235),          // #89dceb (sky)
            success: Color::Rgb(166, 227, 161),       // #a6e3a1 (green)
            warning: Color::Rgb(249, 226, 175),       // #f9e2af (yellow)
            error: Color::Rgb(243, 139, 168),         // #f38ba8 (red)
            highlight: Color::Rgb(69, 71, 90),        // #45475a
            progress_fill: Color::Rgb(137, 180, 250), // #89b4fa
            progress_empty: Color::Rgb(49, 50, 68),   // #313244
            tab_active: Color::Rgb(137, 180, 250),    // #89b4fa
            tab_inactive: Color::Rgb(127, 132, 156),  // #7f849c
        }
    }

    /// Catppuccin Latte — a light counterpart to Mocha.
    pub fn latte() -> Self {
        Self {
            bg: Color::Rgb(239, 241, 245),           // #eff1f5
            frame: Color::Rgb(230, 233, 239),        // #e6e9ef
            surface: Color::Rgb(230, 233, 239),      // #e6e9ef
            surface2: Color::Rgb(220, 224, 232),     // #dce0e8
            border: Color::Rgb(204, 208, 218),       // #ccd0da
            border_focus: Color::Rgb(30, 102, 245),  // #1e66f5 (blue)
            text: Color::Rgb(76, 79, 105),           // #4c4f69
            text_dim: Color::Rgb(108, 111, 133),     // #6c6f85
            faint: Color::Rgb(156, 160, 176),        // #9ca0b0 (overlay0)
            accent: Color::Rgb(30, 102, 245),        // #1e66f5
            info: Color::Rgb(4, 165, 229),           // #04a5e5 (sky)
            success: Color::Rgb(64, 160, 43),        // #40a02b
            warning: Color::Rgb(223, 142, 29),       // #df8e1d
            error: Color::Rgb(210, 15, 57),          // #d20f39
            highlight: Color::Rgb(220, 224, 232),    // #dce0e8
            progress_fill: Color::Rgb(30, 102, 245), // #1e66f5
            progress_empty: Color::Rgb(220, 224, 232),
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
            frame: Color::Rgb(35, 29, 29),            // #231d1d
            surface: Color::Rgb(64, 56, 56),          // #403838
            surface2: Color::Rgb(77, 68, 68),         // #4d4444
            border: Color::Rgb(91, 83, 73),           // #5b5349
            border_focus: Color::Rgb(253, 104, 131),  // #fd6883 (pink-red)
            text: Color::Rgb(255, 241, 243),          // #fff1f3
            text_dim: Color::Rgb(114, 105, 106),      // #72696a
            faint: Color::Rgb(95, 87, 87),            // #5f5757
            accent: Color::Rgb(253, 104, 131),        // #fd6883
            info: Color::Rgb(133, 218, 204),          // #85dacc (blue-green)
            success: Color::Rgb(173, 218, 120),       // #adda78
            warning: Color::Rgb(249, 204, 108),       // #f9cc6c
            error: Color::Rgb(243, 141, 112),         // #f38d70 (orange — distinct from accent)
            highlight: Color::Rgb(64, 56, 56),        // #403838
            progress_fill: Color::Rgb(253, 104, 131), // #fd6883
            progress_empty: Color::Rgb(64, 56, 56),
            tab_active: Color::Rgb(253, 104, 131),
            tab_inactive: Color::Rgb(114, 105, 106),
        }
    }

    /// Gruvbox Dark (hard).
    pub fn gruvbox() -> Self {
        Self {
            bg: Color::Rgb(29, 32, 33),              // #1d2021
            frame: Color::Rgb(40, 40, 40),           // #282828
            surface: Color::Rgb(50, 48, 47),         // #32302f
            surface2: Color::Rgb(60, 56, 54),        // #3c3836
            border: Color::Rgb(80, 73, 69),          // #504945
            border_focus: Color::Rgb(131, 165, 152), // #83a598 (aqua)
            text: Color::Rgb(235, 219, 178),         // #ebdbb2
            text_dim: Color::Rgb(168, 153, 132),     // #a89984
            faint: Color::Rgb(124, 111, 100),        // #7c6f64 (fg4)
            accent: Color::Rgb(131, 165, 152),       // #83a598
            info: Color::Rgb(142, 192, 124),         // #8ec07c (bright aqua)
            success: Color::Rgb(184, 187, 38),       // #b8bb26
            warning: Color::Rgb(250, 189, 47),       // #fabd2f
            error: Color::Rgb(251, 73, 52),          // #fb4934
            highlight: Color::Rgb(60, 56, 54),       // #3c3836
            progress_fill: Color::Rgb(131, 165, 152),
            progress_empty: Color::Rgb(60, 56, 54),
            tab_active: Color::Rgb(131, 165, 152),
            tab_inactive: Color::Rgb(168, 153, 132),
        }
    }

    /// Tokyo Night (storm).
    pub fn tokyo() -> Self {
        Self {
            bg: Color::Rgb(26, 27, 38),              // #1a1b26
            frame: Color::Rgb(22, 22, 30),           // #16161e
            surface: Color::Rgb(36, 40, 59),         // #24283b
            surface2: Color::Rgb(41, 46, 66),        // #292e42
            border: Color::Rgb(59, 66, 97),          // #3b4261
            border_focus: Color::Rgb(122, 162, 247), // #7aa2f7
            text: Color::Rgb(192, 202, 245),         // #c0caf5
            text_dim: Color::Rgb(130, 139, 184),     // #828bb8
            faint: Color::Rgb(86, 95, 137),          // #565f89 (comment)
            accent: Color::Rgb(122, 162, 247),       // #7aa2f7
            info: Color::Rgb(125, 207, 255),         // #7dcfff (cyan)
            success: Color::Rgb(158, 206, 106),      // #9ece6a
            warning: Color::Rgb(224, 175, 104),      // #e0af68
            error: Color::Rgb(247, 118, 142),        // #f7768e
            highlight: Color::Rgb(41, 46, 66),       // #292e42
            progress_fill: Color::Rgb(122, 162, 247),
            progress_empty: Color::Rgb(41, 46, 66),
            tab_active: Color::Rgb(122, 162, 247),
            tab_inactive: Color::Rgb(130, 139, 184),
        }
    }

    /// Nord — cold, muted palette.
    pub fn nord() -> Self {
        Self {
            bg: Color::Rgb(46, 52, 64),              // #2e3440
            frame: Color::Rgb(36, 41, 51),           // #242933
            surface: Color::Rgb(59, 66, 82),         // #3b4252
            surface2: Color::Rgb(67, 76, 94),        // #434c5e
            border: Color::Rgb(67, 76, 94),          // #434c5e
            border_focus: Color::Rgb(136, 192, 208), // #88c0d0
            text: Color::Rgb(216, 222, 233),         // #d8dee9
            text_dim: Color::Rgb(136, 143, 161),     // #888fa1
            faint: Color::Rgb(97, 110, 136),         // #616e88
            accent: Color::Rgb(136, 192, 208),       // #88c0d0
            info: Color::Rgb(129, 161, 193),         // #81a1c1 (nord9 blue)
            success: Color::Rgb(163, 190, 140),      // #a3be8c
            warning: Color::Rgb(235, 203, 139),      // #ebcb8b
            error: Color::Rgb(191, 97, 106),         // #bf616a
            highlight: Color::Rgb(76, 86, 106),      // #4c566a
            progress_fill: Color::Rgb(136, 192, 208),
            progress_empty: Color::Rgb(59, 66, 82),
            tab_active: Color::Rgb(136, 192, 208),
            tab_inactive: Color::Rgb(136, 143, 161),
        }
    }

    /// Dracula — high-contrast purple/cyan.
    ///
    /// Since the Studio redesign the focused border matches the purple
    /// accent (`accent == border_focus` invariant); the signature cyan
    /// moved to the `info` role.
    pub fn dracula() -> Self {
        Self {
            bg: Color::Rgb(40, 42, 54),              // #282a36
            frame: Color::Rgb(33, 34, 44),           // #21222c
            surface: Color::Rgb(68, 71, 90),         // #44475a
            surface2: Color::Rgb(68, 71, 90),        // #44475a
            border: Color::Rgb(98, 114, 164),        // #6272a4
            border_focus: Color::Rgb(189, 147, 249), // #bd93f9 (purple, = accent)
            text: Color::Rgb(248, 248, 242),         // #f8f8f2
            text_dim: Color::Rgb(152, 160, 192),     // lighter version of comment
            faint: Color::Rgb(98, 114, 164),         // #6272a4 (comment)
            accent: Color::Rgb(189, 147, 249),       // #bd93f9 (purple)
            info: Color::Rgb(139, 233, 253),         // #8be9fd (cyan)
            success: Color::Rgb(80, 250, 123),       // #50fa7b
            warning: Color::Rgb(241, 250, 140),      // #f1fa8c
            error: Color::Rgb(255, 85, 85),          // #ff5555
            highlight: Color::Rgb(68, 71, 90),       // #44475a
            progress_fill: Color::Rgb(189, 147, 249),
            progress_empty: Color::Rgb(68, 71, 90),
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

    /// Tertiary text — hints and placeholders.
    pub fn faint(&self) -> Style {
        Style::default().fg(self.faint)
    }

    /// Secondary-accent (halide) text — live state, telemetry, spinners.
    pub fn info(&self) -> Style {
        Style::default().fg(self.info)
    }

    /// Chrome base — activity strip / status bar background.
    pub fn chrome(&self) -> Style {
        Style::default().bg(self.frame).fg(self.text_dim)
    }

    /// Chrome key highlight — status-bar key hints.
    pub fn chrome_key(&self) -> Style {
        Style::default()
            .bg(self.frame)
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
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
        Style::default().bg(self.frame).fg(self.text_dim)
    }

    /// Status bar key highlight.
    pub fn status_key(&self) -> Style {
        Style::default()
            .bg(self.frame)
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
            // Sanity: distinct bg and text so we never render invisible copy,
            // and faint must remain legible against the background.
            assert_ne!(theme.bg, theme.text, "{:?}", preset);
            assert_ne!(theme.bg, theme.faint, "{:?}", preset);
        }
    }

    #[test]
    fn slug_round_trips() {
        for preset in ThemePreset::ALL {
            assert_eq!(ThemePreset::from_slug(preset.slug()), preset);
        }
    }

    #[test]
    fn default_preset_is_studio_dark() {
        assert_eq!(ThemePreset::default(), ThemePreset::StudioDark);
        assert_eq!(Theme::default().bg, Theme::studio_dark().bg);
    }

    #[test]
    fn unknown_slug_falls_back_to_studio_dark() {
        assert_eq!(ThemePreset::from_slug("🐠"), ThemePreset::StudioDark);
        assert_eq!(ThemePreset::from_slug(""), ThemePreset::StudioDark);
    }

    #[test]
    fn all_eleven_design_system_themes_ship() {
        // Spec §05: the four Studio themes ship as TUI presets alongside the
        // seven legacy palettes. Missing any of them is a gap in the
        // Appearance picker.
        let slugs: Vec<&str> = ThemePreset::ALL.iter().map(|p| p.slug()).collect();
        assert_eq!(slugs.len(), 11);
        for expected in [
            "studio-dark",
            "studio-light",
            "safelight-dark",
            "safelight-light",
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
    fn studio_presets_match_spec_hexes() {
        // The regression guard equivalent of
        // `ristretto_swatch_matches_design_system` — pin bg / accent / info
        // to the mockup token values so hue drift is caught.
        let cases = [
            (
                ThemePreset::StudioDark,
                (0x13, 0x11, 0x1d),
                (0xe8, 0x79, 0xf9),
                (0x5c, 0xd0, 0xff),
            ),
            (
                ThemePreset::StudioLight,
                (0xf6, 0xf4, 0xfb),
                (0xa2, 0x1c, 0xaf),
                (0x0e, 0x74, 0x90),
            ),
            (
                ThemePreset::SafelightDark,
                (0x17, 0x12, 0x10),
                (0xf7, 0x94, 0x33),
                (0x8f, 0xb4, 0xc4),
            ),
            (
                ThemePreset::SafelightLight,
                (0xf7, 0xf1, 0xe6),
                (0x9b, 0x4d, 0x00),
                (0x3f, 0x65, 0x70),
            ),
        ];
        for (preset, bg, accent, info) in cases {
            let t = preset.build();
            assert_eq!(t.bg, Color::Rgb(bg.0, bg.1, bg.2), "{preset:?} bg");
            assert_eq!(
                t.accent,
                Color::Rgb(accent.0, accent.1, accent.2),
                "{preset:?} accent"
            );
            assert_eq!(
                t.info,
                Color::Rgb(info.0, info.1, info.2),
                "{preset:?} info"
            );
        }
    }

    #[test]
    fn studio_and_safelight_aliases_map_to_dark_variants() {
        assert_eq!(ThemePreset::from_slug("studio"), ThemePreset::StudioDark);
        assert_eq!(
            ThemePreset::from_slug("safelight"),
            ThemePreset::SafelightDark
        );
    }

    #[test]
    fn accent_equals_border_focus_for_all_presets() {
        // The dual-accent model uses one hue for focus + selection + primary
        // action. A preset whose focused border disagrees with its accent
        // splits that role.
        for preset in ThemePreset::ALL {
            let t = preset.build();
            assert_eq!(t.accent, t.border_focus, "{:?}", preset);
        }
    }

    #[test]
    fn sel_bg_blend_matches_css_rgba() {
        // The mockup expresses selection tints as CSS rgba over the theme
        // bg; terminals need solids. Pin the pre-blended values.
        assert_eq!(
            blend((0xe8, 0x79, 0xf9), (0x13, 0x11, 0x1d), 15),
            Color::Rgb(0x33, 0x21, 0x3e)
        );
        assert_eq!(
            blend((0xa2, 0x1c, 0xaf), (0xf6, 0xf4, 0xfb), 12),
            Color::Rgb(0xec, 0xda, 0xf2)
        );
        assert_eq!(
            blend((0xf7, 0x94, 0x33), (0x17, 0x12, 0x10), 15),
            Color::Rgb(0x39, 0x26, 0x15)
        );
        assert_eq!(
            blend((0x9b, 0x4d, 0x00), (0xf7, 0xf1, 0xe6), 13),
            Color::Rgb(0xeb, 0xdc, 0xc8)
        );
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
