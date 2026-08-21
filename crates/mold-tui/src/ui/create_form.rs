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
/// capability-gated TUI addition beyond the spec's six image sections;
/// `Identity` is the PuLID-FLUX face-reference section — gated on the
/// selected checkpoint's advertised `/api/models[].supports_identity`, never
/// on the family or on a locally compiled feature; and `Filing` is the
/// creation-time "File under" section, which is about where the print lands
/// in the Library rather than how it renders, so it sits last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvSection {
    Sampling,
    Negative,
    Source,
    Identity,
    Lora,
    Upscale,
    Output,
    Video,
    Filing,
}

impl AdvSection {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sampling => "Scheduler & sampling",
            Self::Negative => "Negative prompt",
            Self::Source => "Source image",
            Self::Identity => "Identity photo",
            Self::Lora => "LoRA",
            Self::Upscale => "Upscale after generate",
            Self::Output => "Output format",
            Self::Video => "Video",
            Self::Filing => "File under",
        }
    }

    /// Stable slug for persistence (`tui.advanced_section`).
    pub fn slug(self) -> &'static str {
        match self {
            Self::Sampling => "sampling",
            Self::Negative => "negative",
            Self::Source => "source",
            Self::Identity => "identity",
            Self::Lora => "lora",
            Self::Upscale => "upscale",
            Self::Output => "output",
            Self::Video => "video",
            Self::Filing => "filing",
        }
    }

    pub fn from_slug(slug: &str) -> Option<Self> {
        match slug {
            "sampling" => Some(Self::Sampling),
            "negative" => Some(Self::Negative),
            "source" => Some(Self::Source),
            "identity" => Some(Self::Identity),
            "lora" => Some(Self::Lora),
            "upscale" => Some(Self::Upscale),
            "output" => Some(Self::Output),
            "video" => Some(Self::Video),
            "filing" => Some(Self::Filing),
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
        || caps.supports_references
        || caps.supports_strength
        || caps.supports_mask
        || caps.supports_controlnet
    {
        sections.push(AdvSection::Source);
    }
    // Identity sits beside Source because it is the other conditioning
    // reference, and above LoRA because `mold_core::identity` refuses the
    // two together — the neighbouring rows say so without a warning.
    if caps.supports_identity {
        sections.push(AdvSection::Identity);
    }
    if caps.supports_lora {
        sections.push(AdvSection::Lora);
    }
    sections.push(AdvSection::Upscale);
    sections.push(AdvSection::Output);
    if caps.supports_video {
        sections.push(AdvSection::Video);
    }
    // Creation-time filing is not a generation parameter and has no
    // capability gate: every model produces a print, and every print can be
    // named and filed.
    sections.push(AdvSection::Filing);
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
            if caps.supports_references {
                fields.push(ParamField::References);
            }
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
        AdvSection::Identity => vec![
            ParamField::IdentityImage,
            ParamField::IdentityWeight,
            ParamField::IdentityStartStep,
        ],
        AdvSection::Lora => vec![ParamField::Lora],
        AdvSection::Upscale => vec![ParamField::Upscale],
        AdvSection::Output => vec![ParamField::Format],
        AdvSection::Video => {
            let mut fields = vec![ParamField::Frames, ParamField::Fps];
            if caps.supports_video_upscale {
                fields.push(ParamField::Pipeline);
            }
            if caps.supports_audio && !caps.audio_required {
                fields.push(ParamField::Audio);
            }
            if caps.supports_video_upscale {
                fields.push(ParamField::SpatialUpscale);
                fields.push(ParamField::TemporalUpscale);
                fields.push(ParamField::StgScale);
                fields.push(ParamField::StgBlocks);
                fields.push(ParamField::RescaleScale);
                fields.push(ParamField::ModalityScale);
                fields.push(ParamField::GuidanceSkip);
            }
            if caps.supports_flow_shift {
                fields.push(ParamField::SampleShift);
            }
            fields
        }
        AdvSection::Filing => vec![ParamField::Title, ParamField::Tags, ParamField::Collection],
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
    ];
    if caps.supports_video {
        rows.push(CreateRow::Field(ParamField::Duration));
    }
    rows.extend([
        CreateRow::Field(ParamField::Seed),
        CreateRow::Field(ParamField::Batch),
        CreateRow::AdvancedHeader,
    ]);
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
        AdvSection::Source if !params.reference_paths.is_empty() => {
            format!("{} ordered", params.reference_paths.len())
        }
        AdvSection::Source => params
            .source_image_path
            .as_deref()
            .or(params.control_image_path.as_deref())
            .map(file_name_of)
            .unwrap_or_else(|| "off".into()),
        AdvSection::Identity => params
            .identity_image_path
            .as_deref()
            .map(|path| {
                format!(
                    "{} \u{00b7} {:.2} \u{00b7} step {}",
                    file_name_of(path),
                    params.id_weight,
                    params.id_start_step
                )
            })
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
            if let Some(pipeline) = params.pipeline {
                summary.push_str(&format!(" \u{00b7} {pipeline}"));
            }
            if let Some(enabled) = params.enable_audio {
                summary.push_str(if enabled {
                    " \u{00b7} audio on"
                } else {
                    " \u{00b7} audio off"
                });
            }
            if let Some(upscale) = params.spatial_upscale {
                summary.push_str(match upscale {
                    mold_core::Ltx2SpatialUpscale::X1_5 => " \u{00b7} spatial 1.5×",
                    mold_core::Ltx2SpatialUpscale::X2 => " \u{00b7} spatial 2×",
                });
            }
            if params.temporal_upscale.is_some() {
                summary.push_str(" \u{00b7} temporal 2×");
            }
            let guidance_count = guidance_override_count(params);
            if guidance_count > 0 {
                summary.push_str(&format!(" \u{00b7} guidance {guidance_count}"));
            }
            if let Some(shift) = params.sample_shift {
                summary.push_str(&format!(" \u{00b7} shift {shift:.1}"));
            }
            summary
        }
        AdvSection::Filing => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(title) = params.title.as_deref() {
                parts.push(format!("\u{201c}{title}\u{201d}"));
            }
            match params.tags.len() {
                0 => {}
                1 => parts.push("1 tag".into()),
                count => parts.push(format!("{count} tags")),
            }
            if let Some(collection) = params.collection.as_deref() {
                parts.push(format!("in {collection}"));
            }
            // A tag the user did not type must be visible before Generate,
            // not discovered later in the Library — the same disclosure the
            // CLI prints on stderr.
            if let Some(slug) = auto_tag_disclosure(params) {
                parts.push(format!("auto: {slug}"));
            }
            if parts.is_empty() {
                "none".into()
            } else {
                parts.join(" \u{00b7} ")
            }
        }
    }
}

fn guidance_override_count(params: &GenerateParams) -> usize {
    let overrides = &params.guidance_overrides;
    usize::from(overrides.stg_scale.is_some())
        + usize::from(overrides.stg_blocks.is_some())
        + usize::from(overrides.rescale_scale.is_some())
        + usize::from(overrides.modality_scale.is_some())
        + usize::from(overrides.skip_step.is_some())
}

/// How many advanced values differ from their defaults — the accordion
/// header badge. Counts scheduler, negative prompt, source image, LoRA,
/// upscale, video guidance, non-PNG format, offload, and expand individually.
pub fn advanced_active_count(params: &GenerateParams, negative_empty: bool) -> usize {
    let mut count = 0;
    if params.scheduler.is_some() {
        count += 1;
    }
    if !negative_empty {
        count += 1;
    }
    if !params.reference_paths.is_empty()
        || params.source_image_path.is_some()
        || params.mask_image_path.is_some()
        || params.control_image_path.is_some()
    {
        count += 1;
    }
    // The identity photo counts once; each knob counts only when it differs
    // from the value `mold_core::identity` would apply anyway, so an attached
    // photo on stock settings reads as one active control, not three.
    if params.identity_image_path.is_some() {
        count += 1;
    }
    if params.id_weight != mold_core::identity::ID_WEIGHT_DEFAULT {
        count += 1;
    }
    if params.id_start_step != mold_core::identity::ID_START_STEP_DEFAULT {
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
    if params.pipeline.is_some() {
        count += 1;
    }
    if params.spatial_upscale.is_some() {
        count += 1;
    }
    if params.temporal_upscale.is_some() {
        count += 1;
    }
    if params.sample_shift.is_some() {
        count += 1;
    }
    // Creation-time filing counts one per touched field. The auto tag is
    // derived from the title and is not counted twice.
    if params.title.is_some() {
        count += 1;
    }
    if !params.tags.is_empty() {
        count += 1;
    }
    if params.collection.is_some() {
        count += 1;
    }
    count += guidance_override_count(params);
    count
}

// ── creation-time filing ("File under") ─────────────────────────────
//
// The rules themselves are `mold_core::organization`'s — admission has to
// refuse a bad tag before any model work is paid for, so mold-core owns
// them and every client delegates. What lives here is the *editor* shape:
// the comma-separated string the one-line Tags row speaks, the decision of
// which errors the popup shows, and the auto-tag disclosure the section
// summary renders.

/// Split a comma-separated tag entry into raw names. Normalization,
/// case-insensitive deduplication, and the request cap all belong to
/// [`mold_core::normalize_request_tags`]; this is only the CSV shape.
pub fn parse_tag_input(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

/// Render a stored tag list back into the editor's comma-separated form.
pub fn format_tag_input(tags: &[String]) -> String {
    tags.join(", ")
}

/// Compose the tags this form will submit, disclosing a tag added from the
/// title.
///
/// Delegates to [`mold_core::compose_client_tags`] — the shared client
/// policy the CLI uses — and only rewrites its one cap-overflow message,
/// which names `--no-auto-tag`, a flag the TUI has not got.
pub fn compose_filing_tags(
    tags: &[String],
    title: Option<&str>,
    auto_tag_title: bool,
) -> Result<mold_core::ComposedClientTags, String> {
    // Explicit-list failures (control characters, over-long names, more than
    // the cap of typed tags) carry the shared wording, which reads correctly
    // on every surface.
    mold_core::normalize_request_tags(tags)?;
    mold_core::compose_client_tags(tags, title, auto_tag_title).map_err(|_| {
        format!(
            "the title's tag would exceed the {}-tag limit \u{2014} drop a tag, clear the title, \
             or turn off Settings \u{25b8} Library \u{25b8} Tag by title",
            mold_core::MAX_REQUEST_TAGS
        )
    })
}

/// The tag this form would add from its title, when it would add one.
/// `None` whenever the setting is off, the title is unset or slugless, the
/// user already typed that tag, or the list is too full to take it.
pub fn auto_tag_disclosure(params: &GenerateParams) -> Option<String> {
    compose_filing_tags(&params.tags, params.title.as_deref(), params.auto_tag_title)
        .ok()
        .and_then(|composed| composed.auto_tagged)
}

/// One committed edit from a File-under editor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilingEdit {
    Title(Option<String>),
    Tags(Vec<String>),
    Collection(Option<String>),
}

/// Validate one File-under editor's text against the rest of the form,
/// returning either the value to store or the message the popup keeps
/// showing. Nothing invalid ever reaches a generation request.
///
/// Title and Tags are validated *against each other* — the title may add a
/// tag — so the form can never reach a state whose only symptom is a 422 at
/// submit time.
pub fn commit_filing_input(
    field: ParamField,
    raw: &str,
    params: &GenerateParams,
) -> Result<FilingEdit, String> {
    match field {
        ParamField::Title => {
            let title = mold_core::validate_print_title(raw)?;
            compose_filing_tags(&params.tags, title.as_deref(), params.auto_tag_title)?;
            Ok(FilingEdit::Title(title))
        }
        ParamField::Tags => {
            let parsed = parse_tag_input(raw);
            compose_filing_tags(&parsed, params.title.as_deref(), params.auto_tag_title)?;
            // Store the normalized *explicit* list only. The auto tag stays
            // derived, so clearing the title (or the setting) removes it
            // instead of leaving a tag the user never typed behind.
            Ok(FilingEdit::Tags(mold_core::normalize_request_tags(
                &parsed,
            )?))
        }
        ParamField::Collection => {
            if raw.trim().is_empty() {
                // An emptied editor is an explicit clear, exactly like the
                // Seed row's.
                return Ok(FilingEdit::Collection(None));
            }
            let (name, _slug) = mold_core::validate_collection_name(raw)?;
            Ok(FilingEdit::Collection(Some(name)))
        }
        other => Err(format!("{} is not a File under field", other.label())),
    }
}

/// Wire value for the Negative editor given the model's advertised default
/// negative (`/api/models[].default_negative_prompt`, wan today; empty when
/// the model has none). Mirrors `studio/lib/negativePrompt.ts` exactly so
/// every surface serializes the same tri-state:
///
/// - text equal to the advertised default → `None` (absent keeps the default
///   server-side and preserves today's behavior against older servers);
/// - cleared while a default is advertised → `Some("")`, the explicit empty
///   uncond the engine honors as an opt-out;
/// - anything else non-empty → `Some(text)`; empty with no default → `None`,
///   unless `explicit_clear` marks the empty editor as a restored explicit
///   opt-out (#787 round 3) — then the `""` still ships even while no
///   default is known, so absence cannot re-enable the engine fallback.
pub fn negative_prompt_wire_value(
    text: &str,
    advertised_default: &str,
    supports_negative: bool,
    explicit_clear: bool,
) -> Option<String> {
    if !supports_negative {
        return None;
    }
    let text = text.trim();
    let default = advertised_default.trim();
    if explicit_clear && text.is_empty() {
        return Some(String::new());
    }
    if default.is_empty() {
        return (!text.is_empty()).then(|| text.to_string());
    }
    (text != default).then(|| text.to_string())
}

/// The model's *effective* default negative: the advertised additive field
/// when present, else the family constant for a family whose engine
/// substitutes one anyway (wan, via
/// `mold_core::manifest::default_negative_prompt_for_family`). A known
/// default must survive additive-field absence — resolving the same wan
/// model against an older server that omits `default_negative_prompt` would
/// otherwise collapse the tracked default to `""`, at which point an
/// explicit `""` opt-out serializes as absence and silently re-enables the
/// engine fallback. Mirrors
/// `studio/lib/negativePrompt.ts::effectiveNegativeDefault`; the parity test
/// below pins the two constants byte-for-byte.
pub fn effective_negative_default(advertised: Option<&str>, family: &str) -> String {
    match advertised.map(str::trim).filter(|text| !text.is_empty()) {
        Some(text) => text.to_string(),
        None => mold_core::manifest::default_negative_prompt_for_family(family)
            .unwrap_or_default()
            .to_string(),
    }
}

/// Prefill decision when the advertised default changes (model switch or a
/// fresher catalog). `Some(next)` replaces the editor only while it still
/// shows the previous default (both empty included — that is how the default
/// first appears); a user-typed value, or an explicit clear while a default
/// was advertised, is authority and returns `None`. `explicit_clear` extends
/// that authority to an empty editor restored as an explicit `""` opt-out
/// before any default was known (#787 round 3) — without it the deferred
/// clear is indistinguishable from "untouched" and would take the prefill.
/// Mirrors `studio/lib/negativePrompt.ts::negativePromptOnDefaultChange`.
pub fn negative_prompt_on_default_change(
    current: &str,
    previous_default: &str,
    next_default: &str,
    explicit_clear: bool,
) -> Option<String> {
    if explicit_clear && current.trim().is_empty() {
        return None;
    }
    (current.trim() == previous_default.trim() && current.trim() != next_default.trim())
        .then(|| next_default.trim().to_string())
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

    // ── negative-prompt default contract (#787) ─────────────────
    //
    // These four tests are the TUI half of the cross-surface parity
    // contract; `studio/lib/negativePrompt.test.ts` pins the identical
    // cases for web, desktop, and iPhone.

    const WAN_DEFAULT: &str = mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT;

    #[test]
    fn negative_wire_value_untouched_default_stays_absent() {
        assert_eq!(
            negative_prompt_wire_value(WAN_DEFAULT, WAN_DEFAULT, true, false),
            None
        );
        // No advertised default: empty stays absent (today's behavior).
        assert_eq!(negative_prompt_wire_value("", "", true, false), None);
        // Unsupported family never serializes one.
        assert_eq!(negative_prompt_wire_value("blurry", "", false, false), None);
        assert_eq!(
            negative_prompt_wire_value(WAN_DEFAULT, WAN_DEFAULT, false, false),
            None
        );
    }

    #[test]
    fn negative_wire_value_cleared_is_the_explicit_empty_opt_out() {
        assert_eq!(
            negative_prompt_wire_value("", WAN_DEFAULT, true, false).as_deref(),
            Some("")
        );
        assert_eq!(
            negative_prompt_wire_value("   ", WAN_DEFAULT, true, false).as_deref(),
            Some(""),
            "whitespace-only is a clear, matching the engine's trim"
        );
    }

    #[test]
    fn negative_wire_value_typed_text_replaces_the_default() {
        assert_eq!(
            negative_prompt_wire_value("blurry", WAN_DEFAULT, true, false).as_deref(),
            Some("blurry")
        );
        assert_eq!(
            negative_prompt_wire_value("blurry", "", true, false).as_deref(),
            Some("blurry")
        );
    }

    #[test]
    fn negative_default_change_replaces_only_an_untouched_editor() {
        // The default first appears: empty editor, no previous default.
        assert_eq!(
            negative_prompt_on_default_change("", "", WAN_DEFAULT, false).as_deref(),
            Some(WAN_DEFAULT)
        );
        // Still showing the old default → follow the new model.
        assert_eq!(
            negative_prompt_on_default_change(WAN_DEFAULT, WAN_DEFAULT, "", false).as_deref(),
            Some("")
        );
        // User-typed text is authority.
        assert_eq!(
            negative_prompt_on_default_change("blurry", WAN_DEFAULT, "", false),
            None
        );
        // An explicit clear (opt-out) survives a wan→wan model switch.
        assert_eq!(
            negative_prompt_on_default_change("", WAN_DEFAULT, WAN_DEFAULT, false),
            None
        );
        // No change → no textarea rebuild.
        assert_eq!(
            negative_prompt_on_default_change(WAN_DEFAULT, WAN_DEFAULT, WAN_DEFAULT, false),
            None
        );
    }

    /// #787 round 3: an explicit `""` restored before any default was known
    /// (gallery reuse of an opted-out print, session cold start) is user
    /// authority even though it is visually identical to "untouched". The
    /// marker keeps the clear across the default arriving and keeps the wire
    /// shipping `""`; the identical cases are pinned for the browsers in
    /// `studio/lib/negativePrompt.test.ts`.
    #[test]
    fn deferred_explicit_clear_marker_preserves_the_restored_opt_out() {
        // The default resolves after restore: the marked clear stays.
        assert_eq!(
            negative_prompt_on_default_change("", "", WAN_DEFAULT, true),
            None
        );
        // The wire ships the opt-out even while the default is unknown.
        assert_eq!(
            negative_prompt_wire_value("", "", true, true).as_deref(),
            Some("")
        );
        // The marker is scoped to the empty editor: typed text and an editor
        // still showing the previous default keep the ordinary rules.
        assert_eq!(
            negative_prompt_on_default_change("blurry", "", WAN_DEFAULT, true),
            None
        );
        assert_eq!(
            negative_prompt_wire_value("blurry", "", true, true).as_deref(),
            Some("blurry")
        );
        assert_eq!(
            negative_prompt_on_default_change(WAN_DEFAULT, WAN_DEFAULT, "next", true).as_deref(),
            Some("next")
        );
        // Unsupported family still serializes nothing.
        assert_eq!(negative_prompt_wire_value("", "", false, true), None);
    }

    #[test]
    fn effective_default_prefers_the_advertisement_then_the_family_constant() {
        // Advertised row value wins, trimmed.
        assert_eq!(
            effective_negative_default(Some(" custom "), "wan"),
            "custom"
        );
        // An older server omits the additive field: the known family default
        // survives so the "" opt-out keeps serializing as Some("").
        assert_eq!(effective_negative_default(None, "wan"), WAN_DEFAULT);
        assert_eq!(effective_negative_default(Some("  "), "wan"), WAN_DEFAULT);
        assert_eq!(
            negative_prompt_wire_value("", &effective_negative_default(None, "wan"), true, false)
                .as_deref(),
            Some("")
        );
        // Families without an engine fallback stay empty.
        assert_eq!(effective_negative_default(None, "sdxl"), "");
        assert_eq!(effective_negative_default(None, ""), "");
    }

    /// The Studio surfaces cannot read `mold_core`, so
    /// `studio/lib/negativePrompt.ts` carries its own copy of wan's tuned
    /// default for the older-server fallback. A drifted copy would make web
    /// and TUI disagree about what "untouched" means on the wire.
    #[test]
    fn effective_default_ts_mirror_pins_the_wan_constant() {
        let workspace = env!("CARGO_MANIFEST_DIR")
            .strip_suffix("/crates/mold-tui")
            .or_else(|| env!("CARGO_MANIFEST_DIR").strip_suffix("crates/mold-tui"))
            .unwrap_or(env!("CARGO_MANIFEST_DIR"));
        let path = format!("{workspace}/studio/lib/negativePrompt.ts");
        let source =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        assert!(
            source.contains(&format!("\"{WAN_DEFAULT}\"")),
            "studio/lib/negativePrompt.ts must pin WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT \
             to mold_core::manifest::WAN_DEFAULT_NEGATIVE_PROMPT"
        );
        assert!(
            source.contains("export const WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT"),
            "studio/lib/negativePrompt.ts must export WAN_FAMILY_DEFAULT_NEGATIVE_PROMPT"
        );
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
    fn h3_ref2va_source_section_exposes_only_the_ordered_reference_editor() {
        let caps = crate::model_info::capabilities_for_model(
            mold_core::minimax_h3::FAMILY,
            mold_core::minimax_h3::REF2VA_COMFY,
            None,
            None,
            None,
            None,
        );
        assert_eq!(
            section_fields(AdvSection::Source, &caps),
            vec![ParamField::References]
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
        assert!(rows.contains(&CreateRow::Field(ParamField::Duration)));

        let image_caps = capabilities_for_family("flux");
        assert!(!image_caps.supports_video);
        let rows = visible_rows(&image_caps, &open_state(None));
        assert!(!rows.contains(&CreateRow::Section(AdvSection::Video)));
        assert!(!rows.contains(&CreateRow::Field(ParamField::Duration)));
    }

    #[test]
    fn ltx2_video_section_exposes_pipeline_audio_and_upscalers_without_leaking_to_legacy_video() {
        let ltx2 = capabilities_for_family("ltx2");
        let ltx2_fields = section_fields(AdvSection::Video, &ltx2);
        assert!(ltx2_fields.contains(&ParamField::Pipeline));
        assert!(ltx2_fields.contains(&ParamField::Audio));
        assert!(ltx2_fields.contains(&ParamField::SpatialUpscale));
        assert!(ltx2_fields.contains(&ParamField::TemporalUpscale));
        assert!(ltx2_fields.contains(&ParamField::StgScale));
        assert!(ltx2_fields.contains(&ParamField::StgBlocks));
        assert!(ltx2_fields.contains(&ParamField::RescaleScale));
        assert!(ltx2_fields.contains(&ParamField::ModalityScale));
        assert!(ltx2_fields.contains(&ParamField::GuidanceSkip));

        let legacy = capabilities_for_family("ltx-video");
        let legacy_fields = section_fields(AdvSection::Video, &legacy);
        assert!(!legacy_fields.contains(&ParamField::Pipeline));
        assert!(!legacy_fields.contains(&ParamField::Audio));
        assert!(!legacy_fields.contains(&ParamField::SpatialUpscale));
        assert!(!legacy_fields.contains(&ParamField::TemporalUpscale));
        assert!(!legacy_fields.contains(&ParamField::StgScale));
        assert!(!legacy_fields.contains(&ParamField::StgBlocks));
        assert!(!legacy_fields.contains(&ParamField::RescaleScale));
        assert!(!legacy_fields.contains(&ParamField::ModalityScale));
        assert!(!legacy_fields.contains(&ParamField::GuidanceSkip));
    }

    /// #782: the flow-shift row is wan's alone — the family upstream calls
    /// `--sample_shift` its primary quality knob — and never leaks to the LTX
    /// families or H3.
    #[test]
    fn wan_video_section_exposes_flow_shift_without_leaking_to_other_families() {
        let wan = capabilities_for_family("wan");
        let wan_fields = section_fields(AdvSection::Video, &wan);
        assert!(wan_fields.contains(&ParamField::SampleShift));
        assert!(!wan_fields.contains(&ParamField::StgScale));

        for family in ["ltx2", "ltx-video", mold_core::minimax_h3::FAMILY, "flux"] {
            let caps = capabilities_for_family(family);
            assert!(
                !section_fields(AdvSection::Video, &caps).contains(&ParamField::SampleShift),
                "{family} must not offer the wan flow-shift row"
            );
        }

        // Absent-until-touched: a fresh wan form shows no shift in the
        // summary or the active count; a touched one shows both.
        let mut params = fresh_params();
        assert_eq!(advanced_active_count(&params, true), 0);
        assert!(!section_summary(AdvSection::Video, &params, true).contains("shift"));
        params.sample_shift = Some(12.0);
        assert_eq!(advanced_active_count(&params, true), 1);
        assert!(section_summary(AdvSection::Video, &params, true).contains("shift 12.0"));
    }

    #[test]
    fn h3_video_section_does_not_offer_an_audio_disable_control() {
        let h3 = capabilities_for_family(mold_core::minimax_h3::FAMILY);
        let fields = section_fields(AdvSection::Video, &h3);

        assert!(fields.contains(&ParamField::Frames));
        assert!(fields.contains(&ParamField::Fps));
        assert!(!fields.contains(&ParamField::Audio));
        assert!(!fields.contains(&ParamField::Pipeline));
        assert!(!fields.contains(&ParamField::SpatialUpscale));
        assert!(!fields.contains(&ParamField::TemporalUpscale));
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

    // ── identity (PuLID-FLUX) ───────────────────────────────────

    /// The Identity section is advertised-only. `capabilities_for_model`
    /// reads `/api/models[].supports_identity`; the family says nothing, an
    /// absent field is "no", and an explicit `false` is also "no" — so the
    /// row can never be offered against a server that would refuse it.
    #[test]
    fn identity_section_appears_only_for_an_advertising_checkpoint() {
        for advertised in [None, Some(false)] {
            let caps = crate::model_info::capabilities_for_model(
                "flux",
                "flux-dev:q8",
                None,
                None,
                None,
                advertised,
            );
            assert!(!caps.supports_identity, "advertised: {advertised:?}");
            let rows = visible_rows(&caps, &open_state(None));
            assert!(
                !rows.contains(&CreateRow::Section(AdvSection::Identity)),
                "an unadvertised checkpoint must not offer the Identity section"
            );
        }

        let caps = crate::model_info::capabilities_for_model(
            "flux",
            "flux-dev:q8",
            None,
            None,
            None,
            Some(true),
        );
        assert!(caps.supports_identity);
        let rows = visible_rows(&caps, &open_state(Some(AdvSection::Identity)));
        assert!(rows.contains(&CreateRow::Section(AdvSection::Identity)));
        assert_eq!(
            section_fields(AdvSection::Identity, &caps),
            vec![
                ParamField::IdentityImage,
                ParamField::IdentityWeight,
                ParamField::IdentityStartStep,
            ]
        );
        for field in section_fields(AdvSection::Identity, &caps) {
            assert!(
                rows.contains(&CreateRow::SectionField(AdvSection::Identity, field)),
                "{field:?} must be visible while Identity is expanded"
            );
        }
    }

    /// Identity is a conditioning reference, so it sits next to Source and
    /// above LoRA — the pairing `mold_core::identity` refuses.
    #[test]
    fn identity_section_sits_between_source_and_lora() {
        let caps = crate::model_info::capabilities_for_model(
            "flux",
            "flux-dev:q8",
            None,
            None,
            None,
            Some(true),
        );
        let sections = advanced_sections(&caps);
        let index = |sec| sections.iter().position(|entry| *entry == sec).unwrap();
        assert!(index(AdvSection::Source) < index(AdvSection::Identity));
        assert!(index(AdvSection::Identity) < index(AdvSection::Lora));
    }

    #[test]
    fn identity_summary_and_active_count_track_the_photo_and_its_knobs() {
        let mut params = fresh_params();
        assert_eq!(section_summary(AdvSection::Identity, &params, true), "off");
        assert_eq!(advanced_active_count(&params, true), 0);

        params.identity_image_path = Some("/photos/ada.png".into());
        assert_eq!(
            section_summary(AdvSection::Identity, &params, true),
            "ada.png \u{00b7} 1.00 \u{00b7} step 0"
        );
        assert_eq!(
            advanced_active_count(&params, true),
            1,
            "a photo on stock knobs is one active control, not three"
        );

        params.id_weight = 1.5;
        params.id_start_step = 3;
        assert_eq!(
            section_summary(AdvSection::Identity, &params, true),
            "ada.png \u{00b7} 1.50 \u{00b7} step 3"
        );
        assert_eq!(advanced_active_count(&params, true), 3);
    }

    /// The defaults the fresh form starts on are `mold_core::identity`'s, not
    /// a second copy — a drifted default would record provenance that does
    /// not describe what rendered.
    #[test]
    fn identity_defaults_come_from_mold_core() {
        let params = fresh_params();
        assert_eq!(params.id_weight, mold_core::identity::ID_WEIGHT_DEFAULT);
        assert_eq!(
            params.id_start_step,
            mold_core::identity::ID_START_STEP_DEFAULT
        );
        assert!(mold_core::identity::validate_id_weight(params.id_weight).is_ok());
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
        params.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStageHq);
        assert_eq!(
            section_summary(AdvSection::Video, &params, true),
            "25f \u{00b7} 24fps \u{00b7} two-stage-hq \u{00b7} audio on"
        );
        params.enable_audio = Some(false);
        assert_eq!(
            section_summary(AdvSection::Video, &params, true),
            "25f \u{00b7} 24fps \u{00b7} two-stage-hq \u{00b7} audio off"
        );
        params.spatial_upscale = Some(mold_core::Ltx2SpatialUpscale::X1_5);
        params.temporal_upscale = Some(mold_core::Ltx2TemporalUpscale::X2);
        params.guidance_overrides = mold_core::Ltx2GuidanceOverrides {
            stg_scale: Some(1.5),
            stg_blocks: Some(vec![28, 29]),
            rescale_scale: Some(0.7),
            modality_scale: Some(3.0),
            skip_step: Some(2),
        };
        assert_eq!(
            section_summary(AdvSection::Video, &params, true),
            "25f \u{00b7} 24fps \u{00b7} two-stage-hq \u{00b7} audio off \u{00b7} spatial 1.5× \u{00b7} temporal 2× \u{00b7} guidance 5"
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

        params.pipeline = Some(mold_core::Ltx2PipelineMode::TwoStage);
        assert_eq!(
            advanced_active_count(&params, false),
            10,
            "an explicit pipeline override must count"
        );

        params.spatial_upscale = Some(mold_core::Ltx2SpatialUpscale::X1_5);
        params.temporal_upscale = Some(mold_core::Ltx2TemporalUpscale::X2);
        assert_eq!(
            advanced_active_count(&params, false),
            12,
            "each explicit latent upscale override must count"
        );
        params.guidance_overrides = mold_core::Ltx2GuidanceOverrides {
            stg_scale: Some(1.5),
            stg_blocks: Some(vec![28, 29]),
            rescale_scale: Some(0.7),
            modality_scale: Some(3.0),
            skip_step: Some(2),
        };
        assert_eq!(
            advanced_active_count(&params, false),
            17,
            "each explicit guidance override must count"
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
            AdvSection::Identity,
            AdvSection::Lora,
            AdvSection::Upscale,
            AdvSection::Output,
            AdvSection::Video,
            AdvSection::Filing,
        ] {
            assert_eq!(AdvSection::from_slug(sec.slug()), Some(sec));
        }
        assert_eq!(AdvSection::from_slug("nope"), None);
    }

    // ── File under (creation-time filing) ───────────────────────

    /// Filing has no capability gate: every model produces a print, and
    /// every print can be named and filed. It sits last, after the
    /// generation parameters.
    #[test]
    fn filing_section_is_available_for_every_family_and_lists_its_three_fields() {
        for family in ["sd15", "flux", "ltx2", "wan", mold_core::minimax_h3::FAMILY] {
            let caps = capabilities_for_family(family);
            let sections = advanced_sections(&caps);
            assert_eq!(
                sections.last(),
                Some(&AdvSection::Filing),
                "{family} must end the accordion with File under"
            );
            assert_eq!(
                section_fields(AdvSection::Filing, &caps),
                vec![ParamField::Title, ParamField::Tags, ParamField::Collection],
                "{family}"
            );
        }
    }

    #[test]
    fn visible_rows_expanding_filing_shows_title_tags_and_collection() {
        let caps = capabilities_for_family("sd15");

        // Collapsed: the section row exists, its fields do not.
        let rows = visible_rows(&caps, &open_state(None));
        assert!(rows.contains(&CreateRow::Section(AdvSection::Filing)));
        assert!(!rows
            .iter()
            .any(|row| matches!(row, CreateRow::SectionField(AdvSection::Filing, _))));

        let rows = visible_rows(&caps, &open_state(Some(AdvSection::Filing)));
        let sec_idx = rows
            .iter()
            .position(|r| *r == CreateRow::Section(AdvSection::Filing))
            .unwrap();
        assert_eq!(
            &rows[sec_idx + 1..sec_idx + 4],
            &[
                CreateRow::SectionField(AdvSection::Filing, ParamField::Title),
                CreateRow::SectionField(AdvSection::Filing, ParamField::Tags),
                CreateRow::SectionField(AdvSection::Filing, ParamField::Collection),
            ]
        );
    }

    /// Absent-until-touched: an untouched form has nothing filed, so the
    /// summary says so and the accordion badge stays at zero.
    #[test]
    fn filing_summary_and_active_count_are_empty_until_touched() {
        let params = fresh_params();
        assert_eq!(section_summary(AdvSection::Filing, &params, true), "none");
        assert_eq!(advanced_active_count(&params, true), 0);
        assert_eq!(auto_tag_disclosure(&params), None);
    }

    #[test]
    fn filing_summary_names_title_tags_collection_and_discloses_the_auto_tag() {
        let mut params = fresh_params();
        params.auto_tag_title = true;

        params.title = Some("Smurf Village".into());
        assert_eq!(
            section_summary(AdvSection::Filing, &params, true),
            "\u{201c}Smurf Village\u{201d} \u{00b7} auto: smurf-village",
            "a tag the user did not type must be visible before Generate"
        );

        params.tags = vec!["village".into()];
        params.collection = Some("Blue Period".into());
        assert_eq!(
            section_summary(AdvSection::Filing, &params, true),
            "\u{201c}Smurf Village\u{201d} \u{00b7} 1 tag \u{00b7} in Blue Period \
             \u{00b7} auto: smurf-village"
        );

        params.tags = vec!["village".into(), "blue".into()];
        assert!(section_summary(AdvSection::Filing, &params, true).contains("2 tags"));

        // The preference off removes only the derived tag.
        params.auto_tag_title = false;
        let summary = section_summary(AdvSection::Filing, &params, true);
        assert!(!summary.contains("auto:"), "{summary}");
        assert!(
            summary.contains("\u{201c}Smurf Village\u{201d}"),
            "{summary}"
        );
    }

    #[test]
    fn filing_contributes_one_active_count_per_touched_field() {
        let mut params = fresh_params();
        params.title = Some("Smurf Village".into());
        assert_eq!(
            advanced_active_count(&params, true),
            1,
            "the derived auto tag must not be counted a second time"
        );
        params.tags = vec!["village".into()];
        assert_eq!(advanced_active_count(&params, true), 2);
        params.collection = Some("Blue Period".into());
        assert_eq!(advanced_active_count(&params, true), 3);
    }

    /// The disclosure is exactly `compose_client_tags`' decision, so a tag
    /// the user already typed is theirs and is never announced as added.
    #[test]
    fn auto_tag_disclosure_stays_silent_when_the_user_already_typed_that_tag() {
        let mut params = fresh_params();
        params.auto_tag_title = true;
        params.title = Some("Smurf Village".into());
        params.tags = vec!["Smurf-Village".into()];
        assert_eq!(auto_tag_disclosure(&params), None);
        assert_eq!(
            section_summary(AdvSection::Filing, &params, true),
            "\u{201c}Smurf Village\u{201d} \u{00b7} 1 tag"
        );
    }

    // ── the editors ────────────────────────────────────────────

    #[test]
    fn tag_input_round_trips_through_the_comma_separated_editor() {
        assert_eq!(
            parse_tag_input(" smurfs , village ,, "),
            vec!["smurfs".to_string(), "village".to_string()]
        );
        assert!(parse_tag_input("   ").is_empty());
        assert_eq!(
            format_tag_input(&["smurfs".into(), "village".into()]),
            "smurfs, village"
        );
        assert_eq!(
            parse_tag_input(&format_tag_input(&["smurfs".into(), "village".into()])),
            vec!["smurfs".to_string(), "village".to_string()]
        );
    }

    #[test]
    fn commit_title_validates_and_treats_an_emptied_editor_as_a_clear() {
        let mut params = fresh_params();
        assert_eq!(
            commit_filing_input(ParamField::Title, "  Smurf Village  ", &params),
            Ok(FilingEdit::Title(Some("Smurf Village".into())))
        );
        assert_eq!(
            commit_filing_input(ParamField::Title, "   ", &params),
            Ok(FilingEdit::Title(None))
        );
        assert!(commit_filing_input(ParamField::Title, "nul\0", &params).is_err());
        let long = "x".repeat(mold_core::PRINT_TITLE_MAX_CHARS + 1);
        assert!(commit_filing_input(ParamField::Title, &long, &params).is_err());
        // Untouched by an unrelated field's state.
        params.tags = vec!["village".into()];
        assert!(commit_filing_input(ParamField::Title, "Smurf Village", &params).is_ok());
    }

    #[test]
    fn commit_tags_normalizes_dedupes_and_refuses_what_admission_would() {
        let params = fresh_params();
        assert_eq!(
            commit_filing_input(ParamField::Tags, "Smurfs, smurfs,  village ", &params),
            Ok(FilingEdit::Tags(vec![
                "Smurfs".to_string(),
                "village".to_string()
            ])),
            "case-insensitive duplicates collapse, first spelling wins"
        );
        assert_eq!(
            commit_filing_input(ParamField::Tags, "  ", &params),
            Ok(FilingEdit::Tags(Vec::new())),
            "an emptied editor clears the tags"
        );
        assert!(commit_filing_input(ParamField::Tags, "nul\0", &params).is_err());
        let over: Vec<String> = (0..mold_core::MAX_REQUEST_TAGS + 1)
            .map(|i| format!("t{i}"))
            .collect();
        assert!(commit_filing_input(ParamField::Tags, &over.join(","), &params).is_err());
    }

    #[test]
    fn commit_collection_normalizes_clears_and_refuses_a_slugless_name() {
        let params = fresh_params();
        assert_eq!(
            commit_filing_input(ParamField::Collection, "  Smurf   Village  ", &params),
            Ok(FilingEdit::Collection(Some("Smurf Village".into()))),
            "whitespace runs collapse so the same name merges across hosts"
        );
        assert_eq!(
            commit_filing_input(ParamField::Collection, "   ", &params),
            Ok(FilingEdit::Collection(None))
        );
        // No ASCII alphanumeric survives, so there is no slug to merge on.
        assert!(
            commit_filing_input(ParamField::Collection, "\u{65e5}\u{672c}\u{8a9e}", &params)
                .is_err()
        );
        assert!(commit_filing_input(ParamField::Collection, "nul\0", &params).is_err());
    }

    #[test]
    fn compose_filing_tags_adds_the_title_slug_only_while_the_preference_is_on() {
        let composed =
            compose_filing_tags(&["village".into()], Some("Smurf Village"), true).unwrap();
        assert_eq!(
            composed.tags,
            vec!["village".to_string(), "smurf-village".to_string()]
        );
        assert_eq!(composed.auto_tagged.as_deref(), Some("smurf-village"));

        let composed =
            compose_filing_tags(&["village".into()], Some("Smurf Village"), false).unwrap();
        assert_eq!(composed.tags, vec!["village".to_string()]);
        assert_eq!(composed.auto_tagged, None);

        // No title, nothing derived.
        let composed = compose_filing_tags(&["village".into()], None, true).unwrap();
        assert_eq!(composed.tags, vec!["village".to_string()]);
    }

    /// The shared helper's cap-overflow message names `--no-auto-tag`, a
    /// flag the TUI has not got. Both editors refuse the state instead, in
    /// words that name a control the TUI actually has.
    #[test]
    fn filing_editors_refuse_the_auto_tag_overflow_without_naming_a_cli_flag() {
        let full: Vec<String> = (0..mold_core::MAX_REQUEST_TAGS)
            .map(|i| format!("t{i}"))
            .collect();

        let error = compose_filing_tags(&full, Some("Smurf Village"), true).unwrap_err();
        assert!(!error.contains("--no-auto-tag"), "{error}");
        assert!(error.contains("Tag by title"), "{error}");

        // Setting the title on an already-full form is refused…
        let mut params = fresh_params();
        params.auto_tag_title = true;
        params.tags = full.clone();
        assert!(commit_filing_input(ParamField::Title, "Smurf Village", &params).is_err());
        // …and so is filling the tag list on an already-titled form.
        let mut params = fresh_params();
        params.auto_tag_title = true;
        params.title = Some("Smurf Village".into());
        assert!(commit_filing_input(ParamField::Tags, &full.join(","), &params).is_err());

        // With the preference off the same list is fine both ways.
        let mut params = fresh_params();
        params.auto_tag_title = false;
        params.tags = full.clone();
        assert!(commit_filing_input(ParamField::Title, "Smurf Village", &params).is_ok());
        params.tags.clear();
        params.title = Some("Smurf Village".into());
        assert!(commit_filing_input(ParamField::Tags, &full.join(","), &params).is_ok());
    }

    /// The stored list is the explicit one: the auto tag stays derived, so
    /// clearing the title (or the preference) takes it away again rather
    /// than leaving behind a tag the user never typed.
    #[test]
    fn committing_tags_stores_only_the_explicit_list() {
        let mut params = fresh_params();
        params.auto_tag_title = true;
        params.title = Some("Smurf Village".into());
        assert_eq!(
            commit_filing_input(ParamField::Tags, "village", &params),
            Ok(FilingEdit::Tags(vec!["village".to_string()])),
            "the title's slug must not be frozen into the typed list"
        );
    }

    #[test]
    fn commit_filing_input_rejects_a_field_that_is_not_a_filing_row() {
        let params = fresh_params();
        assert!(commit_filing_input(ParamField::Seed, "1", &params).is_err());
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
