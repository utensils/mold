use ratatui::style::{Color, Modifier, Style};

/// A named colour palette preset.
///
/// The TUI carries the SAME ten themes as every other surface: the five Mold
/// Studio families (ui/theme.ts) in two tones each. Its slugs are the GUI's
/// `ThemeId` values verbatim, and every colour is derived from that family's
/// map in `ui/tokens.css` — `theme_presets_match_the_design_system` reads that
/// file and pins the derivation, which is what the eleven-preset palette this
/// replaced never had. Its `Mocha` was Catppuccin's, not the app's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemePreset {
    #[default]
    MochaDark,
    MochaLight,
    SafelightDark,
    SafelightLight,
    BlueprintDark,
    BlueprintLight,
    GraphiteDark,
    GraphiteLight,
    NebulaDark,
    NebulaLight,
}

impl ThemePreset {
    /// All presets in display order (used by the Appearance swatch grid),
    /// family-major and dark tone first — the same order as `THEMES`.
    pub const ALL: [ThemePreset; 10] = [
        ThemePreset::MochaDark,
        ThemePreset::MochaLight,
        ThemePreset::SafelightDark,
        ThemePreset::SafelightLight,
        ThemePreset::BlueprintDark,
        ThemePreset::BlueprintLight,
        ThemePreset::GraphiteDark,
        ThemePreset::GraphiteLight,
        ThemePreset::NebulaDark,
        ThemePreset::NebulaLight,
    ];

    /// The five families, in picker order, each represented by its dark tone.
    /// A picker names the THEME and offers the tone separately, so the grid
    /// walks these five and paints each in whichever tone is in force.
    pub const FAMILIES: [ThemePreset; 5] = [
        ThemePreset::MochaDark,
        ThemePreset::SafelightDark,
        ThemePreset::BlueprintDark,
        ThemePreset::GraphiteDark,
        ThemePreset::NebulaDark,
    ];

    /// This preset's position in [`ThemePreset::FAMILIES`].
    pub fn family_index(self) -> usize {
        ThemePreset::FAMILIES
            .iter()
            .position(|p| p.label() == self.label())
            .unwrap_or(0)
    }

    /// The family at `index`, in `self`'s current tone.
    pub fn family_at(index: usize, light: bool) -> Self {
        ThemePreset::FAMILIES[index.min(ThemePreset::FAMILIES.len() - 1)].with_tone(light)
    }

    /// The theme's name, with no tone in it.
    pub fn label(self) -> &'static str {
        match self {
            ThemePreset::MochaDark | ThemePreset::MochaLight => "Mocha",
            ThemePreset::SafelightDark | ThemePreset::SafelightLight => "Safelight",
            ThemePreset::BlueprintDark | ThemePreset::BlueprintLight => "Blueprint",
            ThemePreset::GraphiteDark | ThemePreset::GraphiteLight => "Graphite",
            ThemePreset::NebulaDark | ThemePreset::NebulaLight => "Nebula",
        }
    }

    /// Whether this preset is the light or the dark tone of its family.
    pub fn is_light(self) -> bool {
        matches!(
            self,
            ThemePreset::MochaLight
                | ThemePreset::SafelightLight
                | ThemePreset::BlueprintLight
                | ThemePreset::GraphiteLight
                | ThemePreset::NebulaLight
        )
    }

    /// The same family in the requested tone. Never another family.
    pub fn with_tone(self, light: bool) -> Self {
        match (self, light) {
            (ThemePreset::MochaDark | ThemePreset::MochaLight, false) => ThemePreset::MochaDark,
            (ThemePreset::MochaDark | ThemePreset::MochaLight, true) => ThemePreset::MochaLight,
            (ThemePreset::SafelightDark | ThemePreset::SafelightLight, false) => {
                ThemePreset::SafelightDark
            }
            (ThemePreset::SafelightDark | ThemePreset::SafelightLight, true) => {
                ThemePreset::SafelightLight
            }
            (ThemePreset::BlueprintDark | ThemePreset::BlueprintLight, false) => {
                ThemePreset::BlueprintDark
            }
            (ThemePreset::BlueprintDark | ThemePreset::BlueprintLight, true) => {
                ThemePreset::BlueprintLight
            }
            (ThemePreset::GraphiteDark | ThemePreset::GraphiteLight, false) => {
                ThemePreset::GraphiteDark
            }
            (ThemePreset::GraphiteDark | ThemePreset::GraphiteLight, true) => {
                ThemePreset::GraphiteLight
            }
            (ThemePreset::NebulaDark | ThemePreset::NebulaLight, false) => ThemePreset::NebulaDark,
            (ThemePreset::NebulaDark | ThemePreset::NebulaLight, true) => ThemePreset::NebulaLight,
        }
    }

    /// Kebab-case slug, used for config / session persistence. Identical to
    /// the GUI's `ThemeId`, so one vocabulary covers every surface.
    pub fn slug(self) -> &'static str {
        match self {
            ThemePreset::MochaDark => "mocha-dark",
            ThemePreset::MochaLight => "mocha-light",
            ThemePreset::SafelightDark => "safelight-dark",
            ThemePreset::SafelightLight => "safelight-light",
            ThemePreset::BlueprintDark => "blueprint-dark",
            ThemePreset::BlueprintLight => "blueprint-light",
            ThemePreset::GraphiteDark => "graphite-dark",
            ThemePreset::GraphiteLight => "graphite-light",
            ThemePreset::NebulaDark => "nebula-dark",
            ThemePreset::NebulaLight => "nebula-light",
        }
    }

    /// Parse a slug back into a preset.
    ///
    /// Every retired slug still resolves, mapped to the nearest surviving
    /// family by accent hue — a saved `tui.theme` must never fail to load or
    /// silently reset. Unknown slugs fall back to the default.
    pub fn from_slug(slug: &str) -> Self {
        match slug.trim().to_ascii_lowercase().as_str() {
            "mocha-dark" | "mocha" => ThemePreset::MochaDark,
            "mocha-light" | "latte" | "studio-light" => ThemePreset::MochaLight,
            "safelight-dark" | "safelight" | "gruvbox" => ThemePreset::SafelightDark,
            "safelight-light" => ThemePreset::SafelightLight,
            // Cool blue grounds land on the new cyanotype.
            "blueprint-dark" | "tokyo" | "tokyonight" | "tokyo-night" | "nord" => {
                ThemePreset::BlueprintDark
            }
            "blueprint-light" | "blueprint" => ThemePreset::BlueprintLight,
            "graphite-dark" | "graphite" => ThemePreset::GraphiteDark,
            // Porcelain retired as a name; its palette is Graphite's light tone.
            "graphite-light" | "porcelain" => ThemePreset::GraphiteLight,
            // Ristretto's #fd6883 is the crimson family.
            "nebula-dark" | "nebula" | "ristretto" => ThemePreset::NebulaDark,
            "nebula-light" => ThemePreset::NebulaLight,
            // Studio Dark and Dracula were both violet-leaning charcoals.
            "studio-dark" | "studio" | "dracula" => ThemePreset::MochaDark,
            _ => ThemePreset::default(),
        }
    }

    /// Swatch colour shown in the Appearance grid — the accent hue.
    pub fn swatch(self) -> Color {
        self.build().accent
    }

    /// Short palette descriptor for the theme card (dim text beside the
    /// swatch dots). Describes the IDENTITY, never the tone — the tone is the
    /// Appearance control's job.
    pub fn description(self) -> &'static str {
        match self {
            ThemePreset::MochaDark | ThemePreset::MochaLight => "violet neutrals · blue",
            ThemePreset::SafelightDark | ThemePreset::SafelightLight => "warm · amber",
            ThemePreset::BlueprintDark | ThemePreset::BlueprintLight => "drafting blue",
            ThemePreset::GraphiteDark | ThemePreset::GraphiteLight => "true neutral · signal",
            ThemePreset::NebulaDark | ThemePreset::NebulaLight => "oxblood · crimson",
        }
    }

    /// Build a concrete [`Theme`] for this preset.
    pub fn build(self) -> Theme {
        match self {
            ThemePreset::MochaDark => Theme::mocha_dark(),
            ThemePreset::MochaLight => Theme::mocha_light(),
            ThemePreset::SafelightDark => Theme::safelight_dark(),
            ThemePreset::SafelightLight => Theme::safelight_light(),
            ThemePreset::BlueprintDark => Theme::blueprint_dark(),
            ThemePreset::BlueprintLight => Theme::blueprint_light(),
            ThemePreset::GraphiteDark => Theme::graphite_dark(),
            ThemePreset::GraphiteLight => Theme::graphite_light(),
            ThemePreset::NebulaDark => Theme::nebula_dark(),
            ThemePreset::NebulaLight => Theme::nebula_light(),
        }
    }
}

/// The accent tint behind a selected row. Mirrors `--mold-accent-tint`
/// (13 % of the accent), pre-blended because a terminal cell has no alpha.
const ACCENT_TINT_PCT: u16 = 13;

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
        ThemePreset::default().build()
    }
}

impl Theme {
    /// Historical alias retained so external callers that reference
    /// `Theme::dark()` keep compiling. Now points at the default.
    pub fn dark() -> Self {
        ThemePreset::default().build()
    }

    /// Mocha Dark — violet-leaning neutrals, one blue accent.
    pub fn mocha_dark() -> Self {
        Self {
            bg: Color::Rgb(0x1e, 0x1e, 0x2e),
            frame: Color::Rgb(0x18, 0x18, 0x25),
            surface: Color::Rgb(0x31, 0x32, 0x44),
            surface2: Color::Rgb(0x45, 0x47, 0x5a),
            border: Color::Rgb(0x45, 0x47, 0x5a),
            border_focus: Color::Rgb(0x89, 0xb4, 0xfa),
            text: Color::Rgb(0xcd, 0xd6, 0xf4),
            text_dim: Color::Rgb(0x97, 0x9d, 0xb2),
            faint: Color::Rgb(0x58, 0x5b, 0x70),
            accent: Color::Rgb(0x89, 0xb4, 0xfa),
            info: Color::Rgb(0x74, 0xc7, 0xec),
            success: Color::Rgb(0xa6, 0xe3, 0xa1),
            warning: Color::Rgb(0xf9, 0xe2, 0xaf),
            error: Color::Rgb(0xf3, 0x8b, 0xa8),
            highlight: blend((0x89, 0xb4, 0xfa), (0x1e, 0x1e, 0x2e), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0x89, 0xb4, 0xfa),
            progress_empty: Color::Rgb(0x45, 0x47, 0x5a),
            tab_active: Color::Rgb(0x89, 0xb4, 0xfa),
            tab_inactive: Color::Rgb(0x97, 0x9d, 0xb2),
        }
    }

    /// Mocha Light — Catppuccin Latte, the flavour's own light half.
    pub fn mocha_light() -> Self {
        Self {
            bg: Color::Rgb(0xef, 0xf1, 0xf5),
            frame: Color::Rgb(0xe6, 0xe9, 0xef),
            surface: Color::Rgb(0xff, 0xff, 0xff),
            surface2: Color::Rgb(0xcc, 0xd0, 0xda),
            border: Color::Rgb(0xd2, 0xd7, 0xe1),
            border_focus: Color::Rgb(0x1a, 0x5a, 0xc4),
            text: Color::Rgb(0x3c, 0x3f, 0x57),
            text_dim: Color::Rgb(0x58, 0x5b, 0x71),
            faint: Color::Rgb(0x9c, 0xa0, 0xb0),
            accent: Color::Rgb(0x1a, 0x5a, 0xc4),
            info: Color::Rgb(0x0e, 0x74, 0x90),
            success: Color::Rgb(0x0d, 0x7a, 0x4e),
            warning: Color::Rgb(0x8a, 0x52, 0x00),
            error: Color::Rgb(0xc0, 0x2b, 0x34),
            highlight: blend((0x1a, 0x5a, 0xc4), (0xef, 0xf1, 0xf5), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0x1a, 0x5a, 0xc4),
            progress_empty: Color::Rgb(0xcc, 0xd0, 0xda),
            tab_active: Color::Rgb(0x1a, 0x5a, 0xc4),
            tab_inactive: Color::Rgb(0x58, 0x5b, 0x71),
        }
    }

    /// Safelight Dark — the darkroom family: warm browns, amber press.
    pub fn safelight_dark() -> Self {
        Self {
            bg: Color::Rgb(0x24, 0x1c, 0x15),
            frame: Color::Rgb(0x17, 0x12, 0x10),
            surface: Color::Rgb(0x33, 0x26, 0x19),
            surface2: Color::Rgb(0x45, 0x34, 0x24),
            border: Color::Rgb(0x3f, 0x37, 0x2f),
            border_focus: Color::Rgb(0xf7, 0x94, 0x33),
            text: Color::Rgb(0xf1, 0xe8, 0xda),
            text_dim: Color::Rgb(0xa2, 0x98, 0x88),
            faint: Color::Rgb(0x6a, 0x61, 0x58),
            accent: Color::Rgb(0xf7, 0x94, 0x33),
            info: Color::Rgb(0x8f, 0xb4, 0xc4),
            success: Color::Rgb(0x8f, 0xd3, 0x9a),
            warning: Color::Rgb(0xf5, 0xa6, 0x23),
            error: Color::Rgb(0xe5, 0x71, 0x5a),
            highlight: blend((0xf7, 0x94, 0x33), (0x24, 0x1c, 0x15), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0xf7, 0x94, 0x33),
            progress_empty: Color::Rgb(0x45, 0x34, 0x24),
            tab_active: Color::Rgb(0xf7, 0x94, 0x33),
            tab_inactive: Color::Rgb(0xa2, 0x98, 0x88),
        }
    }

    /// Safelight Light — the darkroom with the lights on.
    pub fn safelight_light() -> Self {
        Self {
            bg: Color::Rgb(0xfa, 0xf5, 0xec),
            frame: Color::Rgb(0xf2, 0xe9, 0xdb),
            surface: Color::Rgb(0xff, 0xfd, 0xf8),
            surface2: Color::Rgb(0xee, 0xe2, 0xce),
            border: Color::Rgb(0xe3, 0xd7, 0xc4),
            border_focus: Color::Rgb(0x9a, 0x4f, 0x04),
            text: Color::Rgb(0x2a, 0x21, 0x18),
            text_dim: Color::Rgb(0x6f, 0x5b, 0x46),
            faint: Color::Rgb(0xab, 0x9b, 0x85),
            accent: Color::Rgb(0x9a, 0x4f, 0x04),
            info: Color::Rgb(0x2b, 0x63, 0x82),
            success: Color::Rgb(0x16, 0x6b, 0x43),
            warning: Color::Rgb(0x8a, 0x52, 0x00),
            error: Color::Rgb(0xb2, 0x31, 0x27),
            highlight: blend((0x9a, 0x4f, 0x04), (0xfa, 0xf5, 0xec), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0x9a, 0x4f, 0x04),
            progress_empty: Color::Rgb(0xee, 0xe2, 0xce),
            tab_active: Color::Rgb(0x9a, 0x4f, 0x04),
            tab_inactive: Color::Rgb(0x6f, 0x5b, 0x46),
        }
    }

    /// Blueprint Dark — cyanotype: prussian ground, drafting blue.
    pub fn blueprint_dark() -> Self {
        Self {
            bg: Color::Rgb(0x10, 0x1a, 0x2b),
            frame: Color::Rgb(0x0b, 0x13, 0x20),
            surface: Color::Rgb(0x17, 0x25, 0x3a),
            surface2: Color::Rgb(0x20, 0x31, 0x4a),
            border: Color::Rgb(0x25, 0x33, 0x4a),
            border_focus: Color::Rgb(0x6e, 0xa8, 0xff),
            text: Color::Rgb(0xe6, 0xee, 0xfb),
            text_dim: Color::Rgb(0x93, 0xa5, 0xc2),
            faint: Color::Rgb(0x5b, 0x6d, 0x8c),
            accent: Color::Rgb(0x6e, 0xa8, 0xff),
            info: Color::Rgb(0x57, 0xc7, 0xe8),
            success: Color::Rgb(0x5e, 0xcf, 0x9a),
            warning: Color::Rgb(0xe8, 0xc3, 0x4a),
            error: Color::Rgb(0xff, 0x7a, 0x85),
            highlight: blend((0x6e, 0xa8, 0xff), (0x10, 0x1a, 0x2b), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0x6e, 0xa8, 0xff),
            progress_empty: Color::Rgb(0x20, 0x31, 0x4a),
            tab_active: Color::Rgb(0x6e, 0xa8, 0xff),
            tab_inactive: Color::Rgb(0x93, 0xa5, 0xc2),
        }
    }

    /// Blueprint Light — drafting table, cool daylight.
    pub fn blueprint_light() -> Self {
        Self {
            bg: Color::Rgb(0xf4, 0xf7, 0xfb),
            frame: Color::Rgb(0xe9, 0xef, 0xf7),
            surface: Color::Rgb(0xff, 0xff, 0xff),
            surface2: Color::Rgb(0xdc, 0xe6, 0xf4),
            border: Color::Rgb(0xcf, 0xdc, 0xec),
            border_focus: Color::Rgb(0x1d, 0x4e, 0xd8),
            text: Color::Rgb(0x0d, 0x1b, 0x2e),
            text_dim: Color::Rgb(0x52, 0x62, 0x7a),
            faint: Color::Rgb(0x94, 0xa5, 0xbb),
            accent: Color::Rgb(0x1d, 0x4e, 0xd8),
            info: Color::Rgb(0x1d, 0x4e, 0xd8),
            success: Color::Rgb(0x0f, 0x7b, 0x5f),
            warning: Color::Rgb(0x8a, 0x52, 0x00),
            error: Color::Rgb(0xc0, 0x2b, 0x34),
            highlight: blend((0x1d, 0x4e, 0xd8), (0xf4, 0xf7, 0xfb), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0x1d, 0x4e, 0xd8),
            progress_empty: Color::Rgb(0xdc, 0xe6, 0xf4),
            tab_active: Color::Rgb(0x1d, 0x4e, 0xd8),
            tab_inactive: Color::Rgb(0x52, 0x62, 0x7a),
        }
    }

    /// Graphite Dark — true neutral greys, warm signal.
    pub fn graphite_dark() -> Self {
        Self {
            bg: Color::Rgb(0x1b, 0x1c, 0x1e),
            frame: Color::Rgb(0x14, 0x15, 0x17),
            surface: Color::Rgb(0x23, 0x25, 0x28),
            surface2: Color::Rgb(0x2e, 0x31, 0x34),
            border: Color::Rgb(0x2c, 0x2e, 0x32),
            border_focus: Color::Rgb(0xf0, 0xa2, 0x2e),
            text: Color::Rgb(0xec, 0xed, 0xef),
            text_dim: Color::Rgb(0x8a, 0x8e, 0x93),
            faint: Color::Rgb(0x67, 0x6b, 0x70),
            accent: Color::Rgb(0xf0, 0xa2, 0x2e),
            info: Color::Rgb(0x6b, 0xa8, 0xe5),
            success: Color::Rgb(0x58, 0xc0, 0x8a),
            warning: Color::Rgb(0xe8, 0xc3, 0x4a),
            error: Color::Rgb(0xe5, 0x65, 0x4b),
            highlight: blend((0xf0, 0xa2, 0x2e), (0x1b, 0x1c, 0x1e), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0xf0, 0xa2, 0x2e),
            progress_empty: Color::Rgb(0x2e, 0x31, 0x34),
            tab_active: Color::Rgb(0xf0, 0xa2, 0x2e),
            tab_inactive: Color::Rgb(0x8a, 0x8e, 0x93),
        }
    }

    /// Graphite Light — high-key neutrals (shipped as Porcelain).
    pub fn graphite_light() -> Self {
        Self {
            bg: Color::Rgb(0xfb, 0xfb, 0xfd),
            frame: Color::Rgb(0xf3, 0xf3, 0xf6),
            surface: Color::Rgb(0xff, 0xff, 0xff),
            surface2: Color::Rgb(0xe6, 0xe6, 0xeb),
            border: Color::Rgb(0xe1, 0xe1, 0xe7),
            border_focus: Color::Rgb(0x0f, 0x76, 0x6e),
            text: Color::Rgb(0x16, 0x17, 0x1a),
            text_dim: Color::Rgb(0x63, 0x66, 0x6d),
            faint: Color::Rgb(0xa0, 0xa3, 0xab),
            accent: Color::Rgb(0x0f, 0x76, 0x6e),
            info: Color::Rgb(0x25, 0x63, 0xa8),
            success: Color::Rgb(0x17, 0x79, 0x5a),
            warning: Color::Rgb(0x8a, 0x52, 0x00),
            error: Color::Rgb(0xc0, 0x39, 0x2b),
            highlight: blend((0x0f, 0x76, 0x6e), (0xfb, 0xfb, 0xfd), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0x0f, 0x76, 0x6e),
            progress_empty: Color::Rgb(0xe6, 0xe6, 0xeb),
            tab_active: Color::Rgb(0x0f, 0x76, 0x6e),
            tab_inactive: Color::Rgb(0x63, 0x66, 0x6d),
        }
    }

    /// Nebula Dark — oxblood panels, hot crimson.
    pub fn nebula_dark() -> Self {
        Self {
            bg: Color::Rgb(0x1b, 0x0c, 0x12),
            frame: Color::Rgb(0x11, 0x09, 0x0d),
            surface: Color::Rgb(0x24, 0x10, 0x18),
            surface2: Color::Rgb(0x33, 0x16, 0x1f),
            border: Color::Rgb(0x35, 0x2d, 0x2f),
            border_focus: Color::Rgb(0xff, 0x31, 0x5d),
            text: Color::Rgb(0xff, 0xf8, 0xee),
            text_dim: Color::Rgb(0xa9, 0x8d, 0x8f),
            faint: Color::Rgb(0x7a, 0x62, 0x66),
            accent: Color::Rgb(0xff, 0x31, 0x5d),
            info: Color::Rgb(0xd0, 0x8f, 0x92),
            success: Color::Rgb(0x86, 0xc7, 0x9a),
            warning: Color::Rgb(0xff, 0xb0, 0x5c),
            error: Color::Rgb(0xff, 0x6c, 0x76),
            highlight: blend((0xff, 0x31, 0x5d), (0x1b, 0x0c, 0x12), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0xff, 0x31, 0x5d),
            progress_empty: Color::Rgb(0x33, 0x16, 0x1f),
            tab_active: Color::Rgb(0xff, 0x31, 0x5d),
            tab_inactive: Color::Rgb(0xa9, 0x8d, 0x8f),
        }
    }

    /// Nebula Light — bone paper, oxblood ink.
    pub fn nebula_light() -> Self {
        Self {
            bg: Color::Rgb(0xfb, 0xf6, 0xf2),
            frame: Color::Rgb(0xf3, 0xeb, 0xe5),
            surface: Color::Rgb(0xff, 0xff, 0xff),
            surface2: Color::Rgb(0xec, 0xdf, 0xd8),
            border: Color::Rgb(0xe2, 0xd3, 0xca),
            border_focus: Color::Rgb(0xb3, 0x12, 0x2f),
            text: Color::Rgb(0x2a, 0x0f, 0x16),
            text_dim: Color::Rgb(0x78, 0x55, 0x5a),
            faint: Color::Rgb(0xb0, 0x9a, 0x9d),
            accent: Color::Rgb(0xb3, 0x12, 0x2f),
            info: Color::Rgb(0x2f, 0x5f, 0x9e),
            success: Color::Rgb(0x1c, 0x7a, 0x52),
            warning: Color::Rgb(0x8a, 0x52, 0x00),
            error: Color::Rgb(0xc0, 0x39, 0x2b),
            highlight: blend((0xb3, 0x12, 0x2f), (0xfb, 0xf6, 0xf2), ACCENT_TINT_PCT),
            progress_fill: Color::Rgb(0xb3, 0x12, 0x2f),
            progress_empty: Color::Rgb(0xec, 0xdf, 0xd8),
            tab_active: Color::Rgb(0xb3, 0x12, 0x2f),
            tab_inactive: Color::Rgb(0x78, 0x55, 0x5a),
        }
    }
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
    fn default_preset_is_mocha_dark() {
        assert_eq!(ThemePreset::default(), ThemePreset::MochaDark);
        assert_eq!(Theme::default().bg, Theme::mocha_dark().bg);
    }

    #[test]
    fn unknown_slug_falls_back_to_the_default() {
        assert_eq!(ThemePreset::from_slug("🐠"), ThemePreset::MochaDark);
        assert_eq!(ThemePreset::from_slug(""), ThemePreset::MochaDark);
    }

    #[test]
    fn ships_exactly_the_ten_design_system_themes() {
        // One vocabulary across every surface: these slugs are the GUI's
        // ThemeId set, and nothing else ships. The eleven-preset palette this
        // replaced had five themes with no counterpart anywhere else, and a
        // `Mocha` that was Catppuccin's rather than the app's.
        let slugs: Vec<&str> = ThemePreset::ALL.iter().map(|p| p.slug()).collect();
        assert_eq!(slugs.len(), 10);
        for expected in [
            "mocha-dark",
            "mocha-light",
            "safelight-dark",
            "safelight-light",
            "blueprint-dark",
            "blueprint-light",
            "graphite-dark",
            "graphite-light",
            "nebula-dark",
            "nebula-light",
        ] {
            assert!(
                slugs.contains(&expected),
                "theme `{expected}` is missing from ThemePreset::ALL",
            );
        }
        for retired in [
            "studio-dark",
            "studio-light",
            "ristretto",
            "gruvbox",
            "tokyo",
            "nord",
            "dracula",
        ] {
            assert!(!slugs.contains(&retired), "`{retired}` should have retired");
        }
    }

    /// Every colour a preset builds, read back out of the shared token map.
    fn tokens_for(slug: &str) -> std::collections::HashMap<String, Color> {
        let css = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ui/tokens.css"),
        )
        .expect("ui/tokens.css is the shared token source");
        let needle = format!("[data-theme=\"{slug}\"] {{");
        let start = css
            .find(&needle)
            .unwrap_or_else(|| panic!("no map for {slug}"));
        let body = &css[start + needle.len()..];
        let body = &body[..body.find('}').expect("map is closed")];

        let mut map = std::collections::HashMap::new();
        for line in body.lines() {
            let Some((key, value)) = line.trim().split_once(':') else {
                continue;
            };
            let key = key.trim().trim_start_matches("--mold-");
            let value = value.trim().trim_end_matches(';').trim();
            if let Some(hex) = value.strip_prefix('#') {
                if hex.len() == 6 {
                    let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
                    map.insert(key.to_string(), Color::Rgb(byte(0), byte(2), byte(4)));
                }
            }
        }
        map
    }

    #[test]
    fn theme_presets_match_the_design_system() {
        // The guard the TUI never had. Its palette used to be an independent
        // set of hexes with no link to ui/tokens.css, which is how its `Mocha`
        // came to be a different Mocha from the app's. Every field here is a
        // stated derivation, checked against the one token source.
        for preset in ThemePreset::ALL {
            let t = preset.build();
            let m = tokens_for(preset.slug());
            let tok = |k: &str| {
                *m.get(k)
                    .unwrap_or_else(|| panic!("{k} in {}", preset.slug()))
            };

            assert_eq!(t.bg, tok("bg"), "{preset:?} bg");
            assert_eq!(t.frame, tok("bg-deep"), "{preset:?} frame");
            assert_eq!(t.surface, tok("surface"), "{preset:?} surface");
            assert_eq!(t.surface2, tok("surface-2"), "{preset:?} surface2");
            assert_eq!(t.border, tok("border"), "{preset:?} border");
            assert_eq!(t.text, tok("text"), "{preset:?} text");
            assert_eq!(t.text_dim, tok("text-dim"), "{preset:?} text_dim");
            assert_eq!(t.faint, tok("text-faint"), "{preset:?} faint");
            assert_eq!(t.accent, tok("blue"), "{preset:?} accent");
            // The dual-accent model's second hue.
            assert_eq!(t.info, tok("sapphire"), "{preset:?} info");
            assert_eq!(t.success, tok("success"), "{preset:?} success");
            assert_eq!(t.warning, tok("warning"), "{preset:?} warning");
            assert_eq!(t.error, tok("error"), "{preset:?} error");
            assert_eq!(t.progress_empty, tok("surface-2"), "{preset:?} trough");
        }
    }

    #[test]
    fn every_family_ships_both_tones_and_pairs_with_itself() {
        // The regression the whole contract exists for: a tone flip must keep
        // the theme. Nebula used to become Porcelain in daylight.
        for preset in ThemePreset::ALL {
            let light = preset.with_tone(true);
            let dark = preset.with_tone(false);
            assert!(light.is_light(), "{preset:?} light tone");
            assert!(!dark.is_light(), "{preset:?} dark tone");
            assert_eq!(light.label(), preset.label(), "{preset:?} keeps its name");
            assert_eq!(dark.label(), preset.label(), "{preset:?} keeps its name");
            // Idempotent.
            assert_eq!(light.with_tone(true), light);
            assert_eq!(dark.with_tone(false), dark);
        }
        assert_eq!(ThemePreset::ALL.iter().filter(|p| p.is_light()).count(), 5);
    }

    #[test]
    fn every_retired_slug_still_resolves() {
        // A saved `tui.theme` must never fail to load or silently reset.
        for (retired, expected) in [
            ("studio-dark", ThemePreset::MochaDark),
            ("studio", ThemePreset::MochaDark),
            ("dracula", ThemePreset::MochaDark),
            ("studio-light", ThemePreset::MochaLight),
            ("latte", ThemePreset::MochaLight),
            ("mocha", ThemePreset::MochaDark),
            ("safelight", ThemePreset::SafelightDark),
            ("gruvbox", ThemePreset::SafelightDark),
            ("tokyo", ThemePreset::BlueprintDark),
            ("tokyo-night", ThemePreset::BlueprintDark),
            ("nord", ThemePreset::BlueprintDark),
            ("blueprint", ThemePreset::BlueprintLight),
            ("graphite", ThemePreset::GraphiteDark),
            ("porcelain", ThemePreset::GraphiteLight),
            ("nebula", ThemePreset::NebulaDark),
            ("ristretto", ThemePreset::NebulaDark),
        ] {
            assert_eq!(ThemePreset::from_slug(retired), expected, "{retired}");
        }
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
    fn selection_tint_is_the_pre_blended_accent_tint() {
        // `--mold-accent-tint` is 13 % of the accent; a terminal cell has no
        // alpha, so it is pre-blended over the theme background.
        assert_eq!(ACCENT_TINT_PCT, 13);
        assert_eq!(
            blend((0xff, 0x31, 0x5d), (0x1b, 0x0c, 0x12), 13),
            ThemePreset::NebulaDark.build().highlight
        );
        assert_eq!(
            blend((0x1a, 0x5a, 0xc4), (0xef, 0xf1, 0xf5), 13),
            ThemePreset::MochaLight.build().highlight
        );
    }

    #[test]
    fn a_theme_card_names_the_theme_and_never_its_tone() {
        // Tone is the Appearance control's job; a card that also said "dark"
        // is what made the two read as contradicting pickers.
        for preset in ThemePreset::ALL {
            let label = preset.label().to_ascii_lowercase();
            assert!(!label.contains("dark"), "{preset:?} label");
            // "Safelight" is a name, not a tone — match a whole word only.
            assert!(
                !label
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|w| w == "light"),
                "{preset:?} label",
            );
            assert!(!preset.description().is_empty(), "{preset:?} description");
            let description = preset.description().to_ascii_lowercase();
            assert!(!description.contains("dark"), "{preset:?} description");
        }
        assert_eq!(ThemePreset::FAMILIES.len(), 5);
    }
}
