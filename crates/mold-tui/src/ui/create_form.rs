//! Create-form model — the essentials rows plus the Advanced accordion.
//!
//! This module owns the *shape* of the Create parameters panel: which rows
//! are visible (`visible_rows`), what each accordion section summarises
//! (`section_summary`), how many advanced values differ from their defaults
//! (`advanced_active_count`), and the pure display helpers for the Size and
//! Detail essentials (`size_presets`, `detail_dots`). Rendering lives in
//! `ui::param_form`; key handling lives in `app.rs`. Keeping the state
//! machine here keeps `app.rs` from growing.

use mold_core::OutputFormat;

use crate::app::{GenerateParams, ParamField};
use crate::model_info::ModelCapabilities;

/// Default steps range mapped onto the 8 Detail dots. Values outside the
/// range clamp — the dots are a coarse gauge, not a precise scale.
pub(crate) const DETAIL_MIN: u32 = 6;
pub(crate) const DETAIL_MAX: u32 = 50;
const DETAIL_DOT_COUNT: u32 = 8;

/// The Advanced accordion sections, in display order. `Video` is a
/// capability-gated TUI addition beyond the spec's six image sections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvSection {
    Sampling,
    Negative,
    Source,
    Lora,
    Upscale,
    Output,
    Video,
}

impl AdvSection {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sampling => "Scheduler & sampling",
            Self::Negative => "Negative prompt",
            Self::Source => "Source image",
            Self::Lora => "LoRA",
            Self::Upscale => "Upscale after generate",
            Self::Output => "Output format",
            Self::Video => "Video",
        }
    }

    /// Stable slug for persistence (`tui.advanced_section`).
    pub fn slug(self) -> &'static str {
        match self {
            Self::Sampling => "sampling",
            Self::Negative => "negative",
            Self::Source => "source",
            Self::Lora => "lora",
            Self::Upscale => "upscale",
            Self::Output => "output",
            Self::Video => "video",
        }
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        match slug {
            "sampling" => Some(Self::Sampling),
            "negative" => Some(Self::Negative),
            "source" => Some(Self::Source),
            "lora" => Some(Self::Lora),
            "upscale" => Some(Self::Upscale),
            "output" => Some(Self::Output),
            "video" => Some(Self::Video),
            _ => None,
        }
    }
}

/// Accordion disclosure state. At most one section is expanded; `expanded`
/// is remembered (and persisted) even while the disclosure is closed so
/// re-opening restores the user's place.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct AdvancedState {
    /// Whether the `A` disclosure is open.
    pub open: bool,
    /// The one expanded section, if any.
    pub expanded: Option<AdvSection>,
}

impl AdvancedState {
    /// Load the persisted accordion state (`tui.advanced_open` +
    /// `tui.advanced_section`). Defaults to closed/none when the DB is
    /// unavailable.
    pub fn load() -> Self {
        let Ok(Some(db)) = mold_db::open_default() else {
            return Self::default();
        };
        let s = mold_db::Settings::new(&db);
        let open = s
            .get_bool(mold_db::settings::TUI_ADVANCED_OPEN)
            .unwrap_or(None)
            .unwrap_or(false);
        let expanded = s
            .get_str(mold_db::settings::TUI_ADVANCED_SECTION)
            .unwrap_or(None)
            .as_deref()
            .and_then(AdvSection::from_slug);
        Self { open, expanded }
    }

    /// Persist the accordion state. Silent no-op when the DB is disabled.
    pub fn save(&self) {
        let Ok(Some(db)) = mold_db::open_default() else {
            return;
        };
        let s = mold_db::Settings::new(&db);
        let _ = s.set_bool(mold_db::settings::TUI_ADVANCED_OPEN, self.open);
        let _ = s.set_str(
            mold_db::settings::TUI_ADVANCED_SECTION,
            self.expanded.map(|sec| sec.slug()).unwrap_or(""),
        );
    }
}

/// One navigable row of the Create parameters panel — the flat traversal
/// model `GenerateState.param_index` indexes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateRow {
    /// An essentials field (Model, Size, Steps/Detail, Guidance, Seed, Batch).
    Field(ParamField),
    /// The `▸ Advanced` disclosure row.
    AdvancedHeader,
    /// A section row inside the open accordion.
    Section(AdvSection),
    /// A field row inside an expanded section.
    SectionField(AdvSection, ParamField),
    /// The inline negative-prompt editor (3 rows, Negative expanded only).
    NegativeEditor,
    /// The bottom `↺ Reset to model defaults` action row.
    ResetDefaults,
}

/// Sections available for the given model capabilities, in display order.
pub fn advanced_sections(caps: &ModelCapabilities) -> Vec<AdvSection> {
    let mut sections = vec![AdvSection::Sampling];
    if caps.supports_negative_prompt {
        sections.push(AdvSection::Negative);
    }
    if caps.supports_source_image
        || caps.supports_strength
        || caps.supports_mask
        || caps.supports_controlnet
    {
        sections.push(AdvSection::Source);
    }
    if caps.supports_lora {
        sections.push(AdvSection::Lora);
    }
    sections.push(AdvSection::Upscale);
    sections.push(AdvSection::Output);
    if caps.supports_video {
        sections.push(AdvSection::Video);
    }
    sections
}

/// Field rows inside a section, filtered by capabilities. `Negative` has
/// no fields — its body is the inline editor row.
pub fn section_fields(sec: AdvSection, caps: &ModelCapabilities) -> Vec<ParamField> {
    match sec {
        AdvSection::Sampling => {
            let mut fields = Vec::new();
            if caps.supports_scheduler {
                fields.push(ParamField::Scheduler);
            }
            fields.push(ParamField::Expand);
            fields.push(ParamField::Offload);
            fields
        }
        AdvSection::Negative => Vec::new(),
        AdvSection::Source => {
            let mut fields = Vec::new();
            if caps.supports_source_image {
                fields.push(ParamField::SourceImage);
            }
            if caps.supports_strength {
                fields.push(ParamField::Strength);
            }
            if caps.supports_mask {
                fields.push(ParamField::MaskImage);
            }
            if caps.supports_controlnet {
                fields.push(ParamField::ControlImage);
                fields.push(ParamField::ControlModel);
                fields.push(ParamField::ControlScale);
            }
            fields
        }
        AdvSection::Lora => vec![ParamField::Lora],
        AdvSection::Upscale => vec![ParamField::Upscale],
        AdvSection::Output => vec![ParamField::Format],
        AdvSection::Video => {
            let mut fields = vec![ParamField::Frames, ParamField::Fps];
            if caps.supports_audio {
                fields.push(ParamField::Audio);
            }
            fields
        }
    }
}

/// Build the flat row list for the current capabilities + accordion state.
/// This replaces the old `ParamField::visible_fields` — the capability
/// gating moved here intact; the `InferenceMode` parameter is gone because
/// routing now comes from the Machines generation target.
pub fn visible_rows(caps: &ModelCapabilities, adv: &AdvancedState) -> Vec<CreateRow> {
    let mut rows = vec![
        CreateRow::Field(ParamField::Model),
        CreateRow::Field(ParamField::Size),
        CreateRow::Field(ParamField::Steps),
        CreateRow::Field(ParamField::Guidance),
        CreateRow::Field(ParamField::Seed),
        CreateRow::Field(ParamField::Batch),
        CreateRow::AdvancedHeader,
    ];
    if adv.open {
        for sec in advanced_sections(caps) {
            rows.push(CreateRow::Section(sec));
            if adv.expanded == Some(sec) {
                if sec == AdvSection::Negative {
                    rows.push(CreateRow::NegativeEditor);
                } else {
                    for field in section_fields(sec, caps) {
                        rows.push(CreateRow::SectionField(sec, field));
                    }
                }
            }
        }
    }
    rows.push(CreateRow::ResetDefaults);
    rows
}

fn file_name_of(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

/// The dim right-aligned summary on a collapsed section row (mock-exact
/// defaults: `default` / `off` / `none` / `png`).
pub fn section_summary(sec: AdvSection, params: &GenerateParams, negative_empty: bool) -> String {
    match sec {
        AdvSection::Sampling => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(s) = params.scheduler {
                parts.push(s.to_string());
            }
            if params.expand {
                parts.push("expand".into());
            }
            if params.offload {
                parts.push("offload".into());
            }
            if parts.is_empty() {
                "default".into()
            } else {
                parts.join(" \u{00b7} ")
            }
        }
        AdvSection::Negative => {
            if negative_empty {
                "off".into()
            } else {
                "on".into()
            }
        }
        AdvSection::Source => params
            .source_image_path
            .as_deref()
            .or(params.control_image_path.as_deref())
            .map(file_name_of)
            .unwrap_or_else(|| "off".into()),
        AdvSection::Lora => params
            .lora_path
            .as_deref()
            .map(|p| format!("{} \u{00b7}{:.2}", file_name_of(p), params.lora_scale))
            .unwrap_or_else(|| "none".into()),
        AdvSection::Upscale => params.upscale_model.clone().unwrap_or_else(|| "off".into()),
        AdvSection::Output => format!("{:?}", params.format).to_lowercase(),
        AdvSection::Video => {
            let mut summary = format!("{}f \u{00b7} {}fps", params.frames, params.fps);
            if let Some(enabled) = params.enable_audio {
                summary.push_str(if enabled {
                    " \u{00b7} audio on"
                } else {
                    " \u{00b7} audio off"
                });
            }
            summary
        }
    }
}

/// How many advanced values differ from their defaults — the accordion
/// header badge. Counts scheduler, negative prompt, source image, LoRA,
/// upscale, non-PNG format, offload, and expand individually.
pub fn advanced_active_count(params: &GenerateParams, negative_empty: bool) -> usize {
    let mut count = 0;
    if params.scheduler.is_some() {
        count += 1;
    }
    if !negative_empty {
        count += 1;
    }
    if params.source_image_path.is_some()
        || params.mask_image_path.is_some()
        || params.control_image_path.is_some()
    {
        count += 1;
    }
    if params.lora_path.is_some() {
        count += 1;
    }
    if params.upscale_model.is_some() {
        count += 1;
    }
    if params.format != OutputFormat::Png {
        count += 1;
    }
    if params.offload {
        count += 1;
    }
    if params.expand {
        count += 1;
    }
    if params.enable_audio.is_some() {
        count += 1;
    }
    count
}

/// The dim hint on the Advanced header row: a vocabulary strip while
/// collapsed, a section count while open.
pub fn advanced_header_hint(caps: &ModelCapabilities, adv: &AdvancedState) -> String {
    if adv.open {
        format!("{} sections", advanced_sections(caps).len())
    } else {
        "scheduler \u{00b7} negative \u{00b7} img2img \u{00b7} lora \u{00b7} upscale \u{00b7} format"
            .to_string()
    }
}

/// Aspect-ratio size presets (1:1, 3:2, 2:3, 16:9, 9:16) fitted to the
/// model's default pixel area and aligned to `align` (64 for every current
/// family). `◀▶` on the Size essentials row cycles through these.
pub fn size_presets(default_w: u32, default_h: u32, align: u32) -> Vec<(u32, u32)> {
    let align = align.max(1);
    let area = (default_w.max(align) as f64) * (default_h.max(align) as f64);
    const RATIOS: [(f64, f64); 5] = [(1.0, 1.0), (3.0, 2.0), (2.0, 3.0), (16.0, 9.0), (9.0, 16.0)];
    RATIOS
        .iter()
        .map(|&(rw, rh)| {
            let w = (area * rw / rh).sqrt();
            let h = area / w;
            (round_align(w, align), round_align(h, align))
        })
        .collect()
}

fn round_align(v: f64, align: u32) -> u32 {
    let a = align as f64;
    (((v / a).round().max(1.0)) * a) as u32
}

/// `●●●●○○○○`-style gauge for the Detail (steps) essentials row: 8 dots,
/// at least one filled, all filled at `max`.
pub fn detail_dots(steps: u32, min: u32, max: u32) -> String {
    let (min, max) = if min < max {
        (min, max)
    } else {
        (0, min.max(1))
    };
    let steps = steps.clamp(min, max);
    let span = max - min;
    let filled = ((DETAIL_DOT_COUNT * (steps - min)) / span.max(1)).clamp(1, DETAIL_DOT_COUNT);
    let mut out = String::new();
    for _ in 0..filled {
        out.push('\u{25cf}');
    }
    for _ in filled..DETAIL_DOT_COUNT {
        out.push('\u{25cb}');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_info::capabilities_for_family;
    use mold_core::Config;

    fn fresh_params() -> GenerateParams {
        GenerateParams::from_config(&Config::default())
    }

    fn open_state(expanded: Option<AdvSection>) -> AdvancedState {
        AdvancedState {
            open: true,
            expanded,
        }
    }

    // ── visible_rows ────────────────────────────────────────────

    #[test]
    fn visible_rows_collapsed_hides_all_sections() {
        let caps = capabilities_for_family("sd15");
        let rows = visible_rows(&caps, &AdvancedState::default());
        assert!(
            !rows
                .iter()
                .any(|r| matches!(r, CreateRow::Section(_) | CreateRow::SectionField(..))),
            "collapsed accordion must not emit section rows: {rows:?}"
        );
        // Essentials + header + reset, in order.
        assert_eq!(
            rows,
            vec![
                CreateRow::Field(ParamField::Model),
                CreateRow::Field(ParamField::Size),
                CreateRow::Field(ParamField::Steps),
                CreateRow::Field(ParamField::Guidance),
                CreateRow::Field(ParamField::Seed),
                CreateRow::Field(ParamField::Batch),
                CreateRow::AdvancedHeader,
                CreateRow::ResetDefaults,
            ]
        );
    }

    #[test]
    fn visible_rows_expanding_one_section_collapses_other() {
        let caps = capabilities_for_family("sd15");
        let rows = visible_rows(&caps, &open_state(Some(AdvSection::Sampling)));
        assert!(rows.contains(&CreateRow::SectionField(
            AdvSection::Sampling,
            ParamField::Scheduler
        )));
        // Only the Sampling section carries field rows.
        assert!(
            rows.iter().all(|r| match r {
                CreateRow::SectionField(sec, _) => *sec == AdvSection::Sampling,
                _ => true,
            }),
            "expanding one section must collapse every other: {rows:?}"
        );

        // Switching the expansion moves the fields, exclusively.
        let rows = visible_rows(&caps, &open_state(Some(AdvSection::Output)));
        assert!(rows.contains(&CreateRow::SectionField(
            AdvSection::Output,
            ParamField::Format
        )));
        assert!(rows.iter().all(|r| match r {
            CreateRow::SectionField(sec, _) => *sec == AdvSection::Output,
            _ => true,
        }));
    }

    #[test]
    fn visible_rows_negative_section_expands_to_inline_editor() {
        let caps = capabilities_for_family("sd15");
        assert!(caps.supports_negative_prompt);
        let rows = visible_rows(&caps, &open_state(Some(AdvSection::Negative)));
        let sec_idx = rows
            .iter()
            .position(|r| *r == CreateRow::Section(AdvSection::Negative))
            .unwrap();
        assert_eq!(
            rows[sec_idx + 1],
            CreateRow::NegativeEditor,
            "the Negative section body is the inline editor row"
        );
    }

    #[test]
    fn video_section_only_when_caps_support_video() {
        let video_caps = capabilities_for_family("ltx-video");
        assert!(video_caps.supports_video);
        let rows = visible_rows(&video_caps, &open_state(None));
        assert!(rows.contains(&CreateRow::Section(AdvSection::Video)));

        let image_caps = capabilities_for_family("flux");
        assert!(!image_caps.supports_video);
        let rows = visible_rows(&image_caps, &open_state(None));
        assert!(!rows.contains(&CreateRow::Section(AdvSection::Video)));
    }

    #[test]
    fn ltx2_video_section_exposes_audio_without_leaking_to_legacy_video() {
        let ltx2 = capabilities_for_family("ltx2");
        assert!(section_fields(AdvSection::Video, &ltx2).contains(&ParamField::Audio));

        let legacy = capabilities_for_family("ltx-video");
        assert!(!section_fields(AdvSection::Video, &legacy).contains(&ParamField::Audio));
    }

    #[test]
    fn negative_section_gated_on_capability() {
        let caps = capabilities_for_family("flux");
        if !caps.supports_negative_prompt {
            let rows = visible_rows(&caps, &open_state(None));
            assert!(!rows.contains(&CreateRow::Section(AdvSection::Negative)));
        }
        let sd = capabilities_for_family("sd15");
        let rows = visible_rows(&sd, &open_state(None));
        assert!(rows.contains(&CreateRow::Section(AdvSection::Negative)));
    }

    #[test]
    fn scheduler_row_gated_on_capability() {
        // The capability gating moved intact from ParamField::visible_fields.
        let sd = capabilities_for_family("sd15");
        assert!(section_fields(AdvSection::Sampling, &sd).contains(&ParamField::Scheduler));
        let flux = capabilities_for_family("flux");
        assert!(!section_fields(AdvSection::Sampling, &flux).contains(&ParamField::Scheduler));
        // LoRA section gating.
        let rows = visible_rows(&flux, &open_state(None));
        assert!(rows.contains(&CreateRow::Section(AdvSection::Lora)));
        let wuerstchen = capabilities_for_family("wuerstchen");
        let rows = visible_rows(&wuerstchen, &open_state(None));
        assert!(!rows.contains(&CreateRow::Section(AdvSection::Lora)));
    }

    #[test]
    fn every_field_row_has_a_nonempty_label() {
        let caps = capabilities_for_family("sd15");
        for sec in advanced_sections(&caps) {
            let rows = visible_rows(&caps, &open_state(Some(sec)));
            for row in rows {
                if let CreateRow::Field(f) | CreateRow::SectionField(_, f) = row {
                    assert!(!f.label().is_empty(), "field {f:?} has empty label");
                }
            }
        }
    }

    // ── section summaries + active count ────────────────────────

    #[test]
    fn section_summaries_match_mock_defaults() {
        let params = fresh_params();
        assert_eq!(
            section_summary(AdvSection::Sampling, &params, true),
            "default"
        );
        assert_eq!(section_summary(AdvSection::Negative, &params, true), "off");
        assert_eq!(section_summary(AdvSection::Source, &params, true), "off");
        assert_eq!(section_summary(AdvSection::Lora, &params, true), "none");
        assert_eq!(section_summary(AdvSection::Upscale, &params, true), "off");
        assert_eq!(section_summary(AdvSection::Output, &params, true), "png");
    }

    #[test]
    fn section_summaries_reflect_set_values() {
        let mut params = fresh_params();
        params.scheduler = Some(mold_core::Scheduler::Ddim);
        params.offload = true;
        assert_eq!(
            section_summary(AdvSection::Sampling, &params, true),
            "ddim \u{00b7} offload"
        );
        params.source_image_path = Some("/tmp/cat.png".into());
        assert_eq!(
            section_summary(AdvSection::Source, &params, true),
            "cat.png"
        );
        params.lora_path = Some("/loras/pixel.safetensors".into());
        params.lora_scale = 0.75;
        assert_eq!(
            section_summary(AdvSection::Lora, &params, true),
            "pixel.safetensors \u{00b7}0.75"
        );
        params.upscale_model = Some("real-esrgan-x4plus:fp16".into());
        assert_eq!(
            section_summary(AdvSection::Upscale, &params, true),
            "real-esrgan-x4plus:fp16"
        );
        assert_eq!(section_summary(AdvSection::Negative, &params, false), "on");

        params.enable_audio = Some(true);
        assert_eq!(
            section_summary(AdvSection::Video, &params, true),
            "25f \u{00b7} 24fps \u{00b7} audio on"
        );
        params.enable_audio = Some(false);
        assert_eq!(
            section_summary(AdvSection::Video, &params, true),
            "25f \u{00b7} 24fps \u{00b7} audio off"
        );
    }

    #[test]
    fn active_count_zero_on_fresh_params() {
        assert_eq!(advanced_active_count(&fresh_params(), true), 0);
    }

    #[test]
    fn active_count_counts_scheduler_negative_lora_upscale_format_offload_expand() {
        let mut params = fresh_params();
        params.scheduler = Some(mold_core::Scheduler::Ddim);
        params.lora_path = Some("/loras/pixel.safetensors".into());
        params.upscale_model = Some("real-esrgan-x2:fp16".into());
        params.format = OutputFormat::Jpeg;
        params.offload = true;
        params.expand = true;
        // scheduler + negative + lora + upscale + format + offload + expand
        assert_eq!(advanced_active_count(&params, false), 7);
        // Source counts once no matter how many source-ish paths are set.
        params.source_image_path = Some("/tmp/a.png".into());
        params.mask_image_path = Some("/tmp/m.png".into());
        assert_eq!(advanced_active_count(&params, false), 8);

        params.enable_audio = Some(false);
        assert_eq!(
            advanced_active_count(&params, false),
            9,
            "an explicit audio override must count even when it disables audio"
        );
    }

    #[test]
    fn advanced_header_hint_collapsed_lists_vocabulary() {
        let caps = capabilities_for_family("sd15");
        let hint = advanced_header_hint(&caps, &AdvancedState::default());
        assert_eq!(
            hint,
            "scheduler \u{00b7} negative \u{00b7} img2img \u{00b7} lora \u{00b7} upscale \u{00b7} format"
        );
        let open = advanced_header_hint(&caps, &open_state(None));
        assert_eq!(open, format!("{} sections", advanced_sections(&caps).len()));
    }

    // ── size presets ────────────────────────────────────────────

    #[test]
    fn size_presets_preserve_area_within_tolerance() {
        for &(w, h) in &[(1024u32, 1024u32), (1360, 768), (512, 512)] {
            let area = (w * h) as f64;
            for (pw, ph) in size_presets(w, h, 64) {
                let p_area = (pw * ph) as f64;
                let drift = (p_area - area).abs() / area;
                assert!(
                    drift < 0.15,
                    "preset {pw}x{ph} drifts {:.1}% from the {w}x{h} area",
                    drift * 100.0
                );
            }
        }
    }

    #[test]
    fn size_presets_are_64_aligned() {
        for (pw, ph) in size_presets(1024, 1024, 64) {
            assert_eq!(pw % 64, 0, "{pw} not 64-aligned");
            assert_eq!(ph % 64, 0, "{ph} not 64-aligned");
        }
    }

    #[test]
    fn size_presets_cover_the_five_ratios() {
        let presets = size_presets(1024, 1024, 64);
        assert_eq!(presets.len(), 5);
        // 1:1 on a square default is exact.
        assert_eq!(presets[0], (1024, 1024));
        // Landscape ratios are wider than tall, portrait the reverse.
        assert!(presets[1].0 > presets[1].1, "3:2 must be landscape");
        assert!(presets[2].0 < presets[2].1, "2:3 must be portrait");
        assert!(presets[3].0 > presets[3].1, "16:9 must be landscape");
        assert!(presets[4].0 < presets[4].1, "9:16 must be portrait");
    }

    // ── detail dots ─────────────────────────────────────────────

    #[test]
    fn detail_dots_min_is_one_dot() {
        assert_eq!(detail_dots(DETAIL_MIN, DETAIL_MIN, DETAIL_MAX), "●○○○○○○○");
    }

    #[test]
    fn detail_dots_max_is_eight() {
        assert_eq!(detail_dots(DETAIL_MAX, DETAIL_MIN, DETAIL_MAX), "●●●●●●●●");
    }

    #[test]
    fn detail_dots_monotonic() {
        let mut last = 0;
        for steps in DETAIL_MIN..=DETAIL_MAX {
            let dots = detail_dots(steps, DETAIL_MIN, DETAIL_MAX);
            let filled = dots.chars().filter(|c| *c == '\u{25cf}').count();
            assert!(filled >= last, "dots must never decrease as steps rise");
            last = filled;
        }
        assert_eq!(last, 8);
    }

    #[test]
    fn detail_dots_out_of_range_clamps() {
        assert_eq!(
            detail_dots(1, DETAIL_MIN, DETAIL_MAX),
            detail_dots(DETAIL_MIN, DETAIL_MIN, DETAIL_MAX)
        );
        assert_eq!(
            detail_dots(999, DETAIL_MIN, DETAIL_MAX),
            detail_dots(DETAIL_MAX, DETAIL_MIN, DETAIL_MAX)
        );
        // Degenerate range must not panic.
        assert_eq!(detail_dots(10, 10, 10).chars().count(), 8);
    }

    // ── slug round-trip + persistence ───────────────────────────

    #[test]
    fn adv_section_slug_round_trips() {
        for sec in [
            AdvSection::Sampling,
            AdvSection::Negative,
            AdvSection::Source,
            AdvSection::Lora,
            AdvSection::Upscale,
            AdvSection::Output,
            AdvSection::Video,
        ] {
            assert_eq!(AdvSection::from_slug(sec.slug()), Some(sec));
        }
        assert_eq!(AdvSection::from_slug("nope"), None);
    }

    #[test]
    #[serial_test::serial(mold_env)]
    fn accordion_state_round_trips_through_db() {
        crate::test_env::with_isolated_env(|_home| {
            let state = AdvancedState {
                open: true,
                expanded: Some(AdvSection::Lora),
            };
            state.save();
            assert_eq!(AdvancedState::load(), state);

            // Clearing the expansion persists the empty slug.
            let closed = AdvancedState {
                open: false,
                expanded: None,
            };
            closed.save();
            assert_eq!(AdvancedState::load(), closed);
        });
    }
}
