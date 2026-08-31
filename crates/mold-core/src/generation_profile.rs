//! Canonical, versioned generation-control profiles.
//!
//! A profile is the one model/recipe authority consumed by admission, Rust
//! clients, and `/api/models`. Browser clients receive the same fully-resolved
//! recipes and never reconstruct family policy. The legacy flattened model
//! defaults remain a derived compatibility view for one release.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    validation, GuidanceCapabilities, Ltx2PipelineMode, OutputFormat, Scheduler,
    SourceImageCapability,
};

pub const GENERATION_PROFILE_SCHEMA_VERSION: u32 = 1;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "kebab-case")]
pub enum ResolutionDomain {
    Dynamic,
    Buckets,
    SourceDriven,
    None,
}

/// What an off-bucket resolution means for a `Buckets`-domain recipe.
///
/// `Reject` keeps the historical fail-closed behaviour (H3's reviewed
/// runtime genuinely refuses off-bucket shapes). `Warn` admits any size
/// that clears the alignment/limit/aspect gates — the buckets are the
/// trained sizes the model is optimized for, not the only runnable ones —
/// and the advisory dimension-warning channel tells the user results may
/// vary. Absent on the wire means `Reject`, so older profiles keep today's
/// semantics and clients fail closed against older servers.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "kebab-case")]
#[ts(rename_all = "kebab-case")]
pub enum OffBucketPolicy {
    #[default]
    Reject,
    Warn,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "kebab-case")]
pub enum ControlMode {
    Adjustable,
    Fixed,
    Hidden,
}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS,
)]
#[serde(rename_all = "kebab-case")]
pub enum ProvenanceKind {
    Upstream,
    MoldPolicy,
    Derived,
    DeliveryLimit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct ProfileProvenance {
    pub kind: ProvenanceKind,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub qualified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

/// Server-side qualification record for upstream resolution candidates.
///
/// `qualified` means the dimensions may be presented as recommendations; it
/// is not a per-size runtime-performance claim. A dynamic family can qualify
/// a pinned upstream oracle when Mold's alignment, pixel admission, and image
/// delivery paths are resolution-generic. Bucketed or size-sensitive families
/// additionally need a checked-in exact-size generation-and-delivery campaign.
/// The generator keeps the evidence visible so those two qualification paths
/// cannot be conflated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolutionQualificationRecord {
    pub family: &'static str,
    pub source: &'static str,
    pub revision: &'static str,
    pub qualified: bool,
    pub evidence: &'static str,
    pub candidates: &'static [(u32, u32)],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
#[ts(rename = "ProfileResolutionPreset")]
pub struct ResolutionPreset {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub tier: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
#[ts(rename = "ProfileAspectGroup")]
pub struct AspectGroup {
    pub id: String,
    pub label: String,
    pub presets: Vec<ResolutionPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct ResolutionProfile {
    pub domain: ResolutionDomain,
    pub alignment: u32,
    pub min_width: u32,
    pub min_height: u32,
    #[ts(type = "number")]
    pub max_pixels: u64,
    /// Optional source-image canvas ceiling. Source-driven models can accept
    /// a larger output canvas than their conditioning encoder should ingest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(type = "number | null")]
    pub source_max_pixels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_axis_pixels: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_aspect_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_aspect_ratio: Option<f64>,
    /// Only meaningful for the `Buckets` domain; see [`OffBucketPolicy`].
    /// Absent on the wire (older servers and profiles) means `Reject`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off_bucket: Option<OffBucketPolicy>,
    pub aspect_groups: Vec<AspectGroup>,
}

/// The one human sentence explaining why a control cannot be changed.
///
/// Authored at the single place the fixedness is decided, so no client ever
/// composes copy for a value it did not choose. Absent for every adjustable
/// control and for a fixed control with nothing worth saying — a client that
/// finds no note renders nothing rather than inventing a sentence, which is
/// exactly what an older server's response deserializes to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct IntegerControl {
    pub default: u32,
    pub min: u32,
    pub max: u32,
    pub step: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recommended: Vec<u32>,
    pub mode: ControlMode,
    /// See [`IntegerControl`]'s note on fixed-control copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct FloatControl {
    pub default: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub mode: ControlMode,
    /// See [`IntegerControl`]'s note on fixed-control copy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
#[serde(tag = "mode", rename_all = "kebab-case")]
#[ts(rename = "ProfileFpsControl")]
pub enum FpsControl {
    Fixed {
        value: u32,
    },
    Adjustable {
        default: u32,
        min: u32,
        max: u32,
        step: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct TemporalProfile {
    pub frames: IntegerControl,
    pub frame_offset: u32,
    pub fps: FpsControl,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_duration_seconds: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct GenerationDefaultsProfile {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct RecipeSelector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
}

/// A non-numeric request field's complete UI/admission contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct FeatureControlProfile {
    pub mode: ControlMode,
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// A repeatable adapter input and its immutable stack limit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct AdapterControlProfile {
    pub mode: ControlMode,
    pub max_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct OutputCapabilitiesProfile {
    pub default_format: OutputFormat,
    pub formats: Vec<OutputFormat>,
    pub audio_requires_mp4: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delivery_reason: Option<String>,
}

/// Delivery encoders linked into a concrete Mold binary.
///
/// The authored registry describes the complete Mold-qualified contract. Each
/// executable narrows that contract to what it can actually deliver before it
/// advertises, renders, or validates a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GenerationDeliveryCapabilities {
    pub mp4: bool,
    pub webp: bool,
}

impl GenerationDeliveryCapabilities {
    pub const fn new(mp4: bool, webp: bool) -> Self {
        Self { mp4, webp }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct WanRecipeCapabilitiesProfile {
    pub mode: ControlMode,
    pub supports_distill_strength: bool,
    pub supports_first_last_frame: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_last_frame_min_frames: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct GenerationCapabilitiesProfile {
    pub guidance: GuidanceCapabilities,
    pub negative_prompt: FeatureControlProfile,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image: Option<SourceImageCapability>,
    pub supports_lora: bool,
    pub supports_controlnet: bool,
    /// Face-identity conditioning (`GenerateRequest.id_image`). True only for
    /// an identity-qualified checkpoint on a binary that links the identity
    /// adapter, so a client never offers a control this server would refuse.
    #[serde(default)]
    pub supports_identity: bool,
    pub supports_sequence: bool,
    pub supports_extend: bool,
    pub supports_audio: bool,
    pub source_video: FeatureControlProfile,
    pub mask: FeatureControlProfile,
    pub keyframes: FeatureControlProfile,
    pub audio: FeatureControlProfile,
    pub lora: AdapterControlProfile,
    pub controlnet: AdapterControlProfile,
    pub output: OutputCapabilitiesProfile,
    pub wan_recipe: WanRecipeCapabilitiesProfile,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedulers: Vec<Scheduler>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct GenerationRecipeProfile {
    pub id: String,
    pub label: String,
    pub request_selector: RecipeSelector,
    pub defaults: GenerationDefaultsProfile,
    pub resolution: ResolutionProfile,
    pub steps: IntegerControl,
    pub guidance: FloatControl,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalProfile>,
    pub capabilities: GenerationCapabilitiesProfile,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub provenance: Vec<ProfileProvenance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema, ts_rs::TS)]
pub struct GenerationProfileSet {
    pub schema_version: u32,
    pub profile_id: String,
    pub profile_hash: String,
    pub default_recipe_id: String,
    pub recipes: Vec<GenerationRecipeProfile>,
}

impl GenerationProfileSet {
    pub fn default_recipe(&self) -> Option<&GenerationRecipeProfile> {
        self.recipes
            .iter()
            .find(|recipe| recipe.id == self.default_recipe_id)
    }

    /// Whether any selectable recipe of this model advertises face-identity
    /// conditioning. `/api/models[].supports_identity` reads this rather than
    /// re-deriving the predicate.
    pub fn supports_identity(&self) -> bool {
        self.recipes
            .iter()
            .any(|recipe| recipe.capabilities.supports_identity)
    }

    pub fn recipe_for_pipeline(
        &self,
        pipeline: Option<Ltx2PipelineMode>,
    ) -> Option<&GenerationRecipeProfile> {
        match pipeline {
            Some(pipeline) => self
                .recipes
                .iter()
                .find(|recipe| recipe.request_selector.pipeline == Some(pipeline)),
            None => self.default_recipe(),
        }
    }

    /// Recompute the content address after a server-side runtime probe refines
    /// an advertised capability. The profile hash must describe the exact
    /// contract on the wire, not the pre-probe manifest approximation.
    pub fn refresh_hash(&mut self) {
        self.profile_hash.clear();
        let encoded = serde_json::to_vec(self).expect("generation profile must serialize");
        self.profile_hash = format!("{:x}", Sha256::digest(encoded));
    }
}

/// Narrow an authored profile to the delivery encoders in a concrete binary.
///
/// This is intentionally shared by server and local surfaces so neither can
/// advertise or accept a format the executing binary cannot encode. Recipes
/// with no viable delivery format remain unavailable; they are never silently
/// redirected to an unrelated container.
pub fn qualify_generation_profile_delivery(
    profile: &mut GenerationProfileSet,
    delivery: GenerationDeliveryCapabilities,
) {
    for recipe in &mut profile.recipes {
        recipe
            .capabilities
            .output
            .formats
            .retain(|format| match format {
                OutputFormat::Mp4 => delivery.mp4,
                OutputFormat::Webp => delivery.webp,
                _ => true,
            });
        if !recipe
            .capabilities
            .output
            .formats
            .contains(&recipe.capabilities.output.default_format)
        {
            if let Some(format) = recipe.capabilities.output.formats.first().copied() {
                recipe.capabilities.output.default_format = format;
            }
        }
        if recipe.capabilities.output.audio_requires_mp4 && !delivery.mp4 {
            recipe.capabilities.supports_audio = false;
        }
    }

    profile
        .recipes
        .retain(|recipe| !recipe.capabilities.output.formats.is_empty());
    if !profile
        .recipes
        .iter()
        .any(|recipe| recipe.id == profile.default_recipe_id)
    {
        profile.default_recipe_id = profile
            .recipes
            .first()
            .map(|recipe| recipe.id.clone())
            .unwrap_or_default();
    }
    profile.refresh_hash();
}

/// Resolve the profile-owned output default for a concrete request selector.
pub fn generation_profile_default_output_format(
    profile: &GenerationProfileSet,
    pipeline: Option<Ltx2PipelineMode>,
) -> Result<OutputFormat, String> {
    let recipe = profile.recipe_for_pipeline(pipeline).ok_or_else(|| {
        if let Some(pipeline) = pipeline {
            format!("pipeline '{pipeline}' is not available for this model")
        } else {
            format!(
                "generation profile '{}' has no default recipe",
                profile.profile_id
            )
        }
    })?;
    Ok(recipe.capabilities.output.default_format)
}

/// Fill an omitted request output from its resolved recipe contract.
pub fn materialize_generation_profile_output_default(
    profile: &GenerationProfileSet,
    request: &mut crate::GenerateRequest,
) -> Result<(), String> {
    if request.output_format.is_none() {
        request.output_format = Some(generation_profile_default_output_format(
            profile,
            request.pipeline,
        )?);
    }
    Ok(())
}

/// Validate model-owned request fields against the exact resolved recipe.
/// Family validation may still perform engine-specific structural checks, but
/// it must not widen these advertised controls.
pub fn validate_request_against_generation_profile(
    profile: &GenerationProfileSet,
    request: &crate::GenerateRequest,
) -> Result<(), String> {
    let recipe = if let Some(pipeline) = request.pipeline {
        profile
            .recipes
            .iter()
            .find(|recipe| recipe.request_selector.pipeline == Some(pipeline))
            .ok_or_else(|| format!("pipeline '{}' is not available for this model", pipeline))?
    } else {
        profile.default_recipe().ok_or_else(|| {
            format!(
                "generation profile '{}' has no default recipe",
                profile.profile_id
            )
        })?
    };
    validate_request_against_recipe(recipe, request)
}

pub fn validate_request_against_recipe(
    recipe: &GenerationRecipeProfile,
    request: &crate::GenerateRequest,
) -> Result<(), String> {
    validate_integer("steps", request.steps, &recipe.steps)?;
    validate_float("guidance", request.guidance, &recipe.guidance)?;
    if let Some(scheduler) = request.scheduler {
        let advertised = &recipe.capabilities.schedulers;
        if !advertised.contains(&scheduler) {
            return Err(format!(
                "scheduler '{scheduler}' is not available for this recipe"
            ));
        }
    }
    if let Some(output_format) = request.output_format {
        if !recipe.capabilities.output.formats.contains(&output_format) {
            return Err(format!(
                "output format '{}' is not available for this recipe",
                output_format.extension()
            ));
        }
    }

    let resolution = &recipe.resolution;
    if resolution.domain != ResolutionDomain::None {
        validate_resolution(resolution, request.width, request.height)?;
    }

    if let Some(temporal) = &recipe.temporal {
        let frames = request.frames.unwrap_or(temporal.frames.default);
        let effective_fps = request.fps.unwrap_or(match temporal.fps {
            FpsControl::Fixed { value } => value,
            FpsControl::Adjustable { default, .. } => default,
        });
        let mut effective_frames = temporal.frames.clone();
        if let Some(seconds) = temporal.max_duration_seconds {
            let raw_duration_cap = seconds
                .saturating_mul(effective_fps.max(1))
                .saturating_add(temporal.frame_offset);
            let grid_cap = raw_duration_cap.saturating_sub(temporal.frame_offset)
                / temporal.frames.step
                * temporal.frames.step
                + temporal.frame_offset;
            effective_frames.max = effective_frames.max.min(grid_cap);
        }
        validate_integer("frames", frames, &effective_frames)?;
        match temporal.fps {
            FpsControl::Fixed { value } => {
                if request.fps.is_some_and(|fps| fps != value) {
                    return Err(format!("fps is fixed at {value} for this recipe"));
                }
            }
            FpsControl::Adjustable { min, max, step, .. } => {
                if let Some(fps) = request.fps {
                    if !(min..=max).contains(&fps) || !(fps - min).is_multiple_of(step) {
                        return Err(format!(
                            "fps must be {min} through {max} in steps of {step}"
                        ));
                    }
                }
            }
        }
    } else if request.frames.is_some() || request.fps.is_some() {
        return Err("frames and fps are not supported by this recipe".to_string());
    }
    Ok(())
}

pub fn validate_dimensions_against_recipe(
    recipe: &GenerationRecipeProfile,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if recipe.resolution.domain == ResolutionDomain::None {
        return Err("resolution is not available for this recipe".to_string());
    }
    validate_resolution(&recipe.resolution, width, height)
}

/// Advisory counterpart to the `Warn` off-bucket policy: the size is admitted
/// (it cleared every hard gate), but the model is not tuned for it. `None`
/// for exact buckets, for `Reject`-policy recipes (they refuse instead), and
/// for sizes the recipe refuses outright.
pub fn off_bucket_resolution_warning(
    recipe: &GenerationRecipeProfile,
    width: u32,
    height: u32,
) -> Option<String> {
    let profile = &recipe.resolution;
    if profile.domain != ResolutionDomain::Buckets
        || profile.off_bucket.unwrap_or_default() != OffBucketPolicy::Warn
        || validate_resolution(profile, width, height).is_err()
    {
        return None;
    }
    let exact = profile
        .aspect_groups
        .iter()
        .flat_map(|group| &group.presets)
        .any(|preset| preset.width == width && preset.height == height);
    (!exact).then(|| format!("This model isn't optimized for {width}x{height} — results may vary."))
}

/// Client-surface advisory for a custom size: the server (or forced-local
/// engine) is the admission authority, so a recipe refusal is reported as a
/// warning rather than blocking entry — the request still submits and the
/// authoritative refusal comes back as the job's own error. Falls through to
/// the warn-policy off-bucket advisory for admitted sizes.
pub fn resolution_advisory(
    recipe: &GenerationRecipeProfile,
    width: u32,
    height: u32,
) -> Option<String> {
    match validate_dimensions_against_recipe(recipe, width, height) {
        Err(error) => Some(format!("{error} — the server may reject this size")),
        Ok(()) => off_bucket_resolution_warning(recipe, width, height),
    }
}

fn validate_resolution(profile: &ResolutionProfile, width: u32, height: u32) -> Result<(), String> {
    if width < profile.min_width || height < profile.min_height {
        return Err(format!(
            "width and height must each be at least {}x{} for this recipe",
            profile.min_width, profile.min_height
        ));
    }
    if !width.is_multiple_of(profile.alignment) || !height.is_multiple_of(profile.alignment) {
        return Err(format!(
            "width and height must be multiples of {} for this recipe",
            profile.alignment
        ));
    }
    let pixels = u64::from(width) * u64::from(height);
    if pixels > profile.max_pixels {
        return Err(format!(
            "resolution {width}x{height} exceeds this recipe's {} pixel limit",
            profile.max_pixels
        ));
    }
    if let Some(max_axis) = profile.max_axis_pixels {
        if width > max_axis || height > max_axis {
            return Err(format!(
                "width and height must not exceed {max_axis} for this recipe"
            ));
        }
    }
    let aspect = f64::from(width) / f64::from(height);
    if profile
        .min_aspect_ratio
        .is_some_and(|minimum| aspect < minimum)
        || profile
            .max_aspect_ratio
            .is_some_and(|maximum| aspect > maximum)
    {
        return Err(format!(
            "resolution {width}x{height} is outside this recipe's aspect-ratio range"
        ));
    }
    if profile.domain == ResolutionDomain::Buckets
        && profile.off_bucket.unwrap_or_default() == OffBucketPolicy::Reject
        && !profile
            .aspect_groups
            .iter()
            .flat_map(|group| &group.presets)
            .any(|preset| preset.width == width && preset.height == height)
    {
        return Err(format!(
            "resolution {width}x{height} is not an available bucket for this recipe"
        ));
    }
    Ok(())
}

fn validate_integer(name: &str, value: u32, control: &IntegerControl) -> Result<(), String> {
    if control.mode == ControlMode::Fixed && value != control.default {
        return Err(format!(
            "{name} is fixed at {} for this recipe",
            control.default
        ));
    }
    if !(control.min..=control.max).contains(&value)
        || !(value - control.min).is_multiple_of(control.step)
    {
        return Err(format!(
            "{name} must be {} through {} in steps of {}",
            control.min, control.max, control.step
        ));
    }
    Ok(())
}

fn validate_float(name: &str, value: f64, control: &FloatControl) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite"));
    }
    if control.mode == ControlMode::Fixed && (value - control.default).abs() > f64::EPSILON {
        return Err(format!(
            "{name} is fixed at {} for this recipe",
            control.default
        ));
    }
    if value < control.min || value > control.max {
        return Err(format!(
            "{name} must be {} through {}",
            control.min, control.max
        ));
    }
    if control.step <= 0.0 || !control.step.is_finite() {
        return Err(format!("{name} has an invalid profile step"));
    }
    let steps = (value - control.min) / control.step;
    let tolerance = f64::EPSILON * 16.0 * steps.abs().max(1.0);
    if (steps - steps.round()).abs() > tolerance {
        return Err(format!(
            "{name} must be {} through {} in steps of {}",
            control.min, control.max, control.step
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct GenerationProfileInput<'a> {
    pub model: &'a str,
    pub family: &'a str,
    pub sub_family: Option<&'a str>,
    pub default_width: u32,
    pub default_height: u32,
    pub default_steps: u32,
    pub default_guidance: f64,
    pub default_frames: Option<u32>,
    pub default_fps: Option<u32>,
    pub default_negative_prompt: Option<String>,
    pub source_image: Option<SourceImageCapability>,
    pub supports_sequence: bool,
    pub supports_extend: bool,
    pub supports_audio: bool,
}

/// Resolve the canonical shipped profile for a built-in manifest.
///
/// Catalog advertisement, generated documentation, and registry invariants
/// all call this function so they cannot independently reinterpret manifest
/// defaults or capabilities.
pub fn generation_profile_for_manifest(
    manifest: &crate::manifest::ModelManifest,
) -> GenerationProfileSet {
    let family = manifest.family.as_str();
    generation_profile_for_manifest_with_defaults(
        manifest,
        GenerationDefaultsProfile {
            width: manifest.defaults.width,
            height: manifest.defaults.height,
            steps: manifest.defaults.steps,
            guidance: manifest.defaults.guidance,
            frames: manifest.defaults.frames,
            fps: manifest.defaults.fps,
            negative_prompt: crate::manifest::default_negative_prompt_for_family(family)
                .map(str::to_string),
        },
    )
}

/// Resolve a built-in manifest while preserving validated local default
/// overlays. Identity and capabilities still come exclusively from the
/// manifest; callers may replace only user-owned defaults.
pub fn generation_profile_for_manifest_with_defaults(
    manifest: &crate::manifest::ModelManifest,
    defaults: GenerationDefaultsProfile,
) -> GenerationProfileSet {
    let family = manifest.family.as_str();
    resolve_generation_profile(GenerationProfileInput {
        model: &manifest.name,
        family,
        sub_family: None,
        default_width: defaults.width,
        default_height: defaults.height,
        default_steps: defaults.steps,
        default_guidance: defaults.guidance,
        default_frames: defaults.frames,
        default_fps: defaults.fps,
        default_negative_prompt: defaults.negative_prompt,
        source_image: manifest.defaults.source_image,
        supports_sequence: crate::catalog::chain_capable_family(family),
        supports_extend: crate::catalog::extend_capable_model(
            family,
            manifest.defaults.source_image,
        ),
        // A family that declares its own audio contract answers here (H3
        // always renders synchronized audio); LTX-2 keeps the optimistic
        // family answer the server's per-checkpoint probe then narrows.
        supports_audio: crate::catalog::declared_audio_capability(family, &manifest.name)
            .unwrap_or(family == "ltx2"),
    })
}

const SD15: &[(u32, u32)] = &[(512, 512), (512, 768), (768, 512), (384, 512), (512, 384)];
const SDXL: &[(u32, u32)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
];
const SD3: &[(u32, u32)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
];
const FLUX: &[(u32, u32)] = &[
    (1024, 1024),
    (1024, 768),
    (768, 1024),
    (1024, 576),
    (576, 1024),
    (768, 768),
];
/// Official, runtime-qualified Z-Image-Turbo 1024-tier presets.
const Z_IMAGE_UPSTREAM_CANDIDATES: &[(u32, u32)] = &[
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1152, 864),
    (864, 1152),
    (1248, 832),
    (832, 1248),
    (1280, 720),
    (720, 1280),
    (1344, 576),
    (576, 1344),
];
/// Current official Qwen-Image standard aspect-ratio presets.
const QWEN_UPSTREAM_CANDIDATES: &[(u32, u32)] = &[
    (1328, 1328),
    (1664, 928),
    (928, 1664),
    (1472, 1104),
    (1104, 1472),
    (1584, 1056),
    (1056, 1584),
];
const WUERSTCHEN: &[(u32, u32)] = &[(1024, 1024)];
const LTX_VIDEO: &[(u32, u32)] = &[
    (704, 480),
    (768, 512),
    (512, 512),
    (1024, 576),
    (1216, 704),
    (576, 1024),
    (768, 768),
    (512, 768),
];
const LTX2: &[(u32, u32)] = &[
    (704, 480),
    (768, 512),
    (512, 512),
    (1024, 576),
    (1216, 704),
    (704, 1216),
    (576, 1024),
    (768, 768),
    (512, 768),
    (1536, 1024),
    (1024, 1536),
    (1920, 1088),
    (1088, 1920),
];
const WAN_480: &[(u32, u32)] = &[(832, 480), (480, 832)];
const WAN_480_720: &[(u32, u32)] = &[(832, 480), (480, 832), (1280, 720), (720, 1280)];
const WAN_TI2V: &[(u32, u32)] = &[(1280, 704), (704, 1280)];
const H3: &[(u32, u32)] = &[
    (1536, 672),
    (1344, 768),
    (1024, 768),
    (768, 768),
    (768, 1024),
    (768, 1344),
];
/// The compact stack's RECOMMENDED canvases. Its admission rule is
/// `minimax_h3::is_admitted_compact_canvas` — any 32-aligned canvas inside the
/// campaign's own area ceiling — so these are a ladder to pick from rather
/// than the only sizes that run. Restating the list here is what would let it
/// drift from the runtime that enforces the rule.
const H3_COMPACT: &[(u32, u32)] = crate::minimax_h3::REVIEWED_COMPACT_CANVASES;

const Z_IMAGE_QUALIFICATION: ResolutionQualificationRecord =
    ResolutionQualificationRecord {
        family: "z-image",
        source: "https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/768cb50d847cdbba97c89533ae976be69cf5a5b8/app.py",
        revision: "768cb50d847cdbba97c89533ae976be69cf5a5b8",
        qualified: true,
        evidence: "docs/qualification/z-image-1024-tier-metal-q4.json: exact-size Q4 Metal generation and decoded PNG delivery for every 1024-tier candidate",
        candidates: Z_IMAGE_UPSTREAM_CANDIDATES,
    };

const QWEN_IMAGE_QUALIFICATION: ResolutionQualificationRecord =
    ResolutionQualificationRecord {
        family: "qwen-image",
        source: "https://github.com/QwenLM/Qwen-Image/blob/6b5e1f5cec987d404be5ac6657db3b9aacb56a89/README.md",
        revision: "6b5e1f5cec987d404be5ac6657db3b9aacb56a89",
        qualified: true,
        evidence: "contract qualification: pinned upstream README.md aspect_ratios oracle; Mold dynamic /16 admission and common decoded-image delivery are resolution-generic; no per-size runtime-performance claim",
        candidates: QWEN_UPSTREAM_CANDIDATES,
    };

/// Return the pinned upstream dimension record and its Mold qualification
/// status for a family with an authored aspect set.
pub fn resolution_qualification_record(
    family: &str,
) -> Option<&'static ResolutionQualificationRecord> {
    match canonical_family(family) {
        "z-image" => Some(&Z_IMAGE_QUALIFICATION),
        "qwen-image" => Some(&QWEN_IMAGE_QUALIFICATION),
        _ => None,
    }
}

pub fn family_presets(family: &str) -> &'static [(u32, u32)] {
    match canonical_family(family) {
        "sd15" => SD15,
        "sdxl" => SDXL,
        "sd3" => SD3,
        "flux" | "flux2" => FLUX,
        "z-image" => Z_IMAGE_UPSTREAM_CANDIDATES,
        "qwen-image" | "qwen-image-edit" => QWEN_UPSTREAM_CANDIDATES,
        "wuerstchen" => WUERSTCHEN,
        "ltx-video" => LTX_VIDEO,
        "ltx2" => LTX2,
        "wan" => WAN_480_720,
        "minimax-h3" => H3,
        _ => &[],
    }
}

/// Resolve the authored UI grouping for the conservative legacy family
/// adapter. The dimensions and their display grouping therefore come from the
/// same registry path as versioned profiles.
pub fn family_aspect_groups(family: &str) -> Vec<AspectGroup> {
    let family = canonical_family(family);
    aspect_groups(family, family_presets(family))
}

pub fn presets_for_identity<'a>(
    model: &str,
    family: &str,
    sub_family: Option<&str>,
) -> &'a [(u32, u32)] {
    let family = canonical_family(family);
    if family == "minimax-h3" {
        // Only the hidden official BF16 references keep the flexible ladder.
        // Every compact identity — and any identity the layout resolver does
        // not recognize — takes the fixed reviewed envelope, the same
        // fail-toward-the-stricter-contract rule as `source_fit_dimensions`.
        return match crate::minimax_h3::layout_for_model(model) {
            Some(crate::minimax_h3::Layout::OfficialBf16) => H3,
            _ => H3_COMPACT,
        };
    }
    if family != "wan" {
        return family_presets(family);
    }
    let identity = format!(
        "{} {}",
        crate::manifest::resolve_model_name(model).to_ascii_lowercase(),
        sub_family.unwrap_or_default().to_ascii_lowercase()
    );
    if identity.contains("ti2v-5b") {
        WAN_TI2V
    } else if identity.contains("1.3b") {
        WAN_480
    } else {
        WAN_480_720
    }
}

pub fn canonical_family(family: &str) -> &str {
    match family.trim() {
        "ltx-2" => "ltx2",
        "flux.2" | "flux-2" => "flux2",
        "minimax_h3" | "minimaxh3" => "minimax-h3",
        other => other,
    }
}

pub fn resolve_generation_profile(input: GenerationProfileInput<'_>) -> GenerationProfileSet {
    let family = canonical_family(input.family);
    let profile_id = input
        .sub_family
        .map(|sub| format!("{family}.{sub}"))
        .unwrap_or_else(|| format!("{family}.{}", crate::manifest::model_base_name(input.model)));
    let mut recipes = if family == "ltx2" {
        let mut recipes = vec![recipe(&input, "auto", "Auto", None)];
        for pipeline in Ltx2PipelineMode::ALL {
            recipes.push(recipe(
                &input,
                pipeline.as_str(),
                &pipeline_label(pipeline),
                Some(pipeline),
            ));
        }
        recipes
    } else {
        vec![recipe(&input, "default", "Default", None)]
    };
    recipes.retain(|recipe| {
        !(family == "ltx2"
            && recipe.request_selector.pipeline == Some(Ltx2PipelineMode::T2a)
            && !input.supports_audio)
    });
    let mut set = GenerationProfileSet {
        schema_version: GENERATION_PROFILE_SCHEMA_VERSION,
        profile_id,
        profile_hash: String::new(),
        default_recipe_id: if family == "ltx2" { "auto" } else { "default" }.to_string(),
        recipes,
    };
    set.refresh_hash();
    set
}

/// The sentence explaining a fixed guidance control.
///
/// Authored beside the decision that fixes the scale, so the wording can never
/// describe a value the recipe did not pin. H3 does not run a guided branch at
/// all, so its copy names that rather than a distilled CFG the user could
/// escape by picking a Dev checkpoint.
fn fixed_guidance_note(
    family: &str,
    guidance_caps: GuidanceCapabilities,
    scale: f64,
) -> Option<String> {
    if guidance_caps.adjustable {
        return None;
    }
    if family == "minimax-h3" {
        return Some(
            "MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0.".to_string(),
        );
    }
    Some(format!(
        "Distilled recipe fixes CFG at {scale:.1}. Choose a Dev checkpoint with Auto or a guided pipeline to adjust it."
    ))
}

/// The sentence explaining a Turbo tier's fixed step count.
///
/// A reviewed Turbo tier's `steps` is terminal-inclusive — the published
/// N-step schedule has N denoise intervals and therefore N+1 sampler grid
/// points ([`crate::minimax_h3::TurboManifestTier::steps`]) — so the field
/// shows 9 for the 8-step tier. Saying so is the whole point of the note.
fn fixed_turbo_steps_note(tier: &crate::minimax_h3::TurboManifestTier) -> String {
    let intervals = tier.steps.saturating_sub(1);
    let points = tier.steps;
    format!(
        "Fixed by the {intervals}-step Turbo tier: {points} terminal-inclusive sampler grid points ({intervals} denoise intervals)."
    )
}

fn recipe(
    input: &GenerationProfileInput<'_>,
    id: &str,
    label: &str,
    pipeline: Option<Ltx2PipelineMode>,
) -> GenerationRecipeProfile {
    let family = canonical_family(input.family);
    // The reviewed compact H3 stack has one canvas, one step count, and one
    // frame count. All three are derived from the identity, never from
    // `input.default_*` — those are laundered through user `model_prefs` in
    // `build_model_catalog` and can carry a stale off-envelope value.
    let h3_compact = crate::minimax_h3::uses_reviewed_compact_envelope(family, input.model);
    // A Turbo tier's step count is a property of the distilled adapter, not
    // a qualification pin: its schedule has exactly that many grid points, so
    // it stays a fixed control. The base compact tag takes a range.
    let h3_compact_turbo = crate::minimax_h3::turbo_tier_for_model(input.model);
    let h3_compact_turbo_steps = h3_compact_turbo.map(|tier| tier.steps);
    let h3_compact_steps = h3_compact_turbo_steps.unwrap_or(crate::minimax_h3::COMFY_DEFAULT_STEPS);
    let audio_only = pipeline == Some(Ltx2PipelineMode::T2a);
    let source_driven = family == "qwen-image-edit"
        || matches!(
            pipeline,
            Some(Ltx2PipelineMode::Retake | Ltx2PipelineMode::LipDub)
        );
    let composed = family == "ltx2"
        && pipeline
            .map(Ltx2PipelineMode::refines_spatially)
            .unwrap_or_else(|| {
                validation::ltx2_spatial_composition(input.model, None)
                    == validation::Ltx2SpatialComposition::TiledTwoStage
            });
    let composition = if composed {
        validation::Ltx2SpatialComposition::TiledTwoStage
    } else {
        validation::Ltx2SpatialComposition::SinglePass
    };
    let alignment = if composed {
        validation::LTX2_TWO_STAGE_ALIGNMENT
    } else if family == "wan" {
        identity_alignment(input.model, family, input.sub_family)
    } else {
        validation::dimension_alignment_for_family(Some(family))
    };
    let mut dimensions = presets_for_identity(input.model, family, input.sub_family).to_vec();
    if composed {
        for rung in validation::LTX2_OUTPUT_RUNGS
            .iter()
            .filter(|rung| rung.requires_tiled_stage2())
        {
            dimensions.push((rung.width, rung.height));
            dimensions.push((rung.height, rung.width));
        }
    }
    dimensions.retain(|(width, height)| {
        width.is_multiple_of(alignment)
            && height.is_multiple_of(alignment)
            && validation::validate_generation_dimensions_composed(
                *width,
                *height,
                Some(family),
                composition,
            )
            .is_ok()
    });
    let resolution = if audio_only {
        ResolutionProfile {
            domain: ResolutionDomain::None,
            alignment: 1,
            min_width: 0,
            min_height: 0,
            max_pixels: 0,
            source_max_pixels: None,
            max_axis_pixels: None,
            min_aspect_ratio: None,
            max_aspect_ratio: None,
            off_bucket: None,
            aspect_groups: Vec::new(),
        }
    } else {
        ResolutionProfile {
            domain: if source_driven {
                ResolutionDomain::SourceDriven
            } else if family == "wan" {
                ResolutionDomain::Buckets
            } else {
                // The compact H3 stack is a RANGE, not a bucket set: any
                // 32-aligned canvas inside its area ceiling is admitted and
                // the memory estimate decides what fits. Its presets survive
                // as `aspect_groups` recommendations.
                ResolutionDomain::Dynamic
            },
            alignment,
            min_width: if h3_compact {
                crate::minimax_h3::reviewed_compact_min_axis_pixels()
            } else {
                alignment.max(64)
            },
            min_height: if h3_compact {
                crate::minimax_h3::reviewed_compact_min_axis_pixels()
            } else {
                alignment.max(64)
            },
            // The compact stack's ceilings come from its own canvas rule. The
            // family constants are the *official* BF16 ladder's headroom and
            // sit above the compact area ceiling, which lets a client that
            // reads the ceiling offer a size admission refuses.
            max_pixels: if h3_compact {
                crate::minimax_h3::reviewed_compact_max_pixels()
            } else {
                validation::max_pixels_for_family_composed(Some(family), composition)
            },
            source_max_pixels: (family == "qwen-image-edit")
                .then_some(validation::QWEN_IMAGE_EDIT_SOURCE_MAX_PIXELS),
            max_axis_pixels: if h3_compact {
                Some(crate::minimax_h3::reviewed_compact_max_axis_pixels())
            } else {
                validation::max_axis_pixels_for_family_composed(Some(family), composition)
            },
            min_aspect_ratio: (family == "minimax-h3")
                .then_some(crate::minimax_h3::MIN_ASPECT_RATIO),
            max_aspect_ratio: (family == "minimax-h3")
                .then_some(crate::minimax_h3::MAX_ASPECT_RATIO),
            // Wan's buckets are the trained sizes, not the only runnable
            // ones — a deliberate off-bucket request is admitted and the
            // advisory warning channel says results may vary. H3's compact
            // presets are recommendations on a continuous range, so there is
            // no off-bucket question to answer at all.
            off_bucket: (family == "wan").then_some(OffBucketPolicy::Warn),
            aspect_groups: aspect_groups(family, &dimensions),
        }
    };

    let guidance_caps = if family == "minimax-h3" {
        GuidanceCapabilities {
            adjustable: false,
            supports_negative_prompt: false,
            fixed_scale: Some(0.0),
        }
    } else {
        GuidanceCapabilities::for_recipe(family, input.model, pipeline)
    };
    let effective_guidance = guidance_caps.fixed_scale.unwrap_or(input.default_guidance);
    let mut temporal = temporal_profile(input, family);
    if let Some(temporal) = temporal.as_mut().filter(|_| h3_compact) {
        // The compact stack takes the FAMILY frame grid. It was pinned to one
        // clip length because the runtime envelope validated `frames` by
        // equality; the envelope is now minted per request and the memory
        // estimate refuses what does not fit.
        temporal.frames = IntegerControl {
            default: crate::minimax_h3::DEFAULT_COMPACT_FRAMES,
            min: crate::minimax_h3::MIN_FRAMES,
            max: crate::minimax_h3::MAX_FRAMES,
            step: crate::minimax_h3::FRAME_STEP,
            recommended: vec![
                crate::minimax_h3::MIN_FRAMES,
                crate::minimax_h3::DEFAULT_COMPACT_FRAMES,
                226,
                crate::minimax_h3::MAX_FRAMES,
            ],
            mode: ControlMode::Adjustable,
            note: None,
        };
    }
    let flux2_dev = family == "flux2"
        && crate::manifest::resolve_model_name(input.model)
            .to_ascii_lowercase()
            .contains("dev");
    let wan = family == "wan";
    let normalized_model = crate::manifest::resolve_model_name(input.model).to_ascii_lowercase();
    // BFL's own FP8 Flux.2 conversions store `weight / weight_scale`, and a
    // LoRA merge widens the weight it patches — dropping the scale on exactly
    // the layers the adapter touches. `Flux2Engine::load_transformer` refuses
    // the pair, so the control must not be offered. Same shape of refusal as
    // wan's fp8-scaled expert tier below.
    let flux2_fp8 = family == "flux2" && normalized_model.ends_with(":fp8");
    // Checkpoints whose own loader refuses an adapter stack, whatever the
    // family supports: FLUX.2 [dev]'s reference protocol, the FP8 Flux.2
    // conversions, and wan's fp8-scaled expert pair.
    let checkpoint_refuses_lora =
        flux2_dev || flux2_fp8 || (wan && normalized_model.ends_with("a14b:fp8"));
    let lora_supported = validation::family_supports_lora(family) && !checkpoint_refuses_lora;
    let source_video_required = matches!(
        pipeline,
        Some(Ltx2PipelineMode::IcLora | Ltx2PipelineMode::Retake | Ltx2PipelineMode::LipDub)
    );
    let source_video_supported = family == "ltx2" && !audio_only;
    let keyframes_required = pipeline == Some(Ltx2PipelineMode::Keyframe);
    let keyframes_supported = family == "ltx2" || wan;
    let audio_input_required = pipeline == Some(Ltx2PipelineMode::A2Vid);
    let audio_input_supported = family == "ltx2" && !audio_only;
    let mask_supported = !audio_only
        && !matches!(
            family,
            "ltx-video" | "ltx2" | "wan" | "qwen-image-edit" | "minimax-h3"
        )
        && !flux2_dev
        && input.source_image != Some(SourceImageCapability::Unsupported);
    let controlnet_supported = family == "sd15";
    // Identity conditioning is advertised only when this binary can actually
    // execute it AND the checkpoint is one of the qualified ones. Both halves
    // belong to `crate::identity`: `identity_runtime_available` is the feature
    // AND the landed runtime adapter, never a bare `cfg!(feature = "pulid")`,
    // which would advertise a control the worker cannot honour.
    let identity_supported = crate::identity::identity_runtime_available()
        && crate::identity::identity_qualified_model_with_family(input.model, Some(family));
    let output = if audio_only {
        OutputCapabilitiesProfile {
            default_format: OutputFormat::Wav,
            formats: vec![OutputFormat::Wav],
            audio_requires_mp4: false,
            delivery_reason: Some("Audio-only delivery uses WAV.".to_string()),
        }
    } else if family == "minimax-h3" {
        OutputCapabilitiesProfile {
            default_format: OutputFormat::Mp4,
            formats: vec![OutputFormat::Mp4],
            audio_requires_mp4: true,
            delivery_reason: Some("Synchronized H3 audio/video delivery requires MP4.".to_string()),
        }
    } else if temporal.is_some() {
        OutputCapabilitiesProfile {
            default_format: OutputFormat::Mp4,
            formats: vec![
                OutputFormat::Mp4,
                OutputFormat::Gif,
                OutputFormat::Apng,
                OutputFormat::Webp,
            ],
            audio_requires_mp4: family == "ltx2",
            delivery_reason: (family == "ltx2")
                .then(|| "Audio-enabled video delivery requires MP4.".to_string()),
        }
    } else {
        OutputCapabilitiesProfile {
            default_format: OutputFormat::Png,
            formats: vec![OutputFormat::Png, OutputFormat::Jpeg, OutputFormat::Webp],
            audio_requires_mp4: false,
            delivery_reason: None,
        }
    };
    let default_width = if h3_compact {
        crate::minimax_h3::DEFAULT_WIDTH
    } else {
        input.default_width
    };
    let default_height = if h3_compact {
        crate::minimax_h3::DEFAULT_HEIGHT
    } else {
        input.default_height
    };
    let default_steps = if h3_compact {
        h3_compact_steps
    } else {
        input.default_steps
    };
    let defaults = GenerationDefaultsProfile {
        width: if audio_only { 0 } else { default_width },
        height: if audio_only { 0 } else { default_height },
        steps: default_steps,
        guidance: effective_guidance,
        frames: temporal.as_ref().map(|profile| profile.frames.default),
        fps: temporal.as_ref().map(|profile| match profile.fps {
            FpsControl::Fixed { value } => value,
            FpsControl::Adjustable { default, .. } => default,
        }),
        negative_prompt: input.default_negative_prompt.clone(),
    };
    GenerationRecipeProfile {
        id: id.to_string(),
        label: label.to_string(),
        request_selector: RecipeSelector { pipeline },
        defaults,
        resolution,
        steps: IntegerControl {
            default: default_steps,
            min: match (h3_compact_turbo_steps, family) {
                (Some(steps), _) => steps,
                (None, "minimax-h3") => crate::minimax_h3::COMPACT_MIN_STEPS,
                _ => 1,
            },
            max: match (h3_compact_turbo_steps, h3_compact) {
                (Some(steps), _) => steps,
                (None, true) => crate::minimax_h3::COMPACT_MAX_STEPS,
                (None, false) => 100,
            },
            step: 1,
            recommended: vec![default_steps],
            mode: if h3_compact_turbo_steps.is_some() {
                ControlMode::Fixed
            } else {
                ControlMode::Adjustable
            },
            note: h3_compact_turbo.map(fixed_turbo_steps_note),
        },
        guidance: FloatControl {
            default: effective_guidance,
            min: if guidance_caps.adjustable {
                0.0
            } else {
                effective_guidance
            },
            max: if guidance_caps.adjustable {
                100.0
            } else {
                effective_guidance
            },
            step: 0.1,
            mode: if guidance_caps.adjustable {
                ControlMode::Adjustable
            } else {
                ControlMode::Fixed
            },
            note: fixed_guidance_note(family, guidance_caps, effective_guidance),
        },
        temporal,
        capabilities: GenerationCapabilitiesProfile {
            guidance: guidance_caps,
            negative_prompt: feature_control(
                guidance_caps.supports_negative_prompt,
                false,
                "This recipe does not encode a negative prompt.",
            ),
            source_image: input.source_image,
            supports_lora: lora_supported,
            supports_controlnet: controlnet_supported,
            supports_identity: identity_supported,
            supports_sequence: input.supports_sequence && !audio_only,
            supports_extend: input.supports_extend && !audio_only,
            supports_audio: input.supports_audio || audio_only,
            source_video: feature_control(
                source_video_supported,
                source_video_required,
                "This recipe does not accept a source video.",
            ),
            mask: feature_control(
                mask_supported,
                false,
                "This model does not accept an inpainting mask.",
            ),
            keyframes: feature_control(
                keyframes_supported,
                keyframes_required,
                "This model does not accept keyframes.",
            ),
            audio: feature_control(
                audio_input_supported,
                audio_input_required,
                "This recipe does not accept source audio.",
            ),
            lora: AdapterControlProfile {
                mode: if lora_supported {
                    ControlMode::Adjustable
                } else {
                    ControlMode::Hidden
                },
                max_count: if lora_supported {
                    if pipeline == Some(Ltx2PipelineMode::IcLora) {
                        3
                    } else {
                        4
                    }
                } else {
                    0
                },
                reason: (!lora_supported)
                    .then(|| "This model does not accept LoRA adapters.".to_string()),
            },
            controlnet: AdapterControlProfile {
                mode: if controlnet_supported {
                    ControlMode::Adjustable
                } else {
                    ControlMode::Hidden
                },
                max_count: u32::from(controlnet_supported),
                reason: (!controlnet_supported)
                    .then(|| "ControlNet generation is available for SD1.5 models.".to_string()),
            },
            output,
            wan_recipe: WanRecipeCapabilitiesProfile {
                mode: if wan {
                    ControlMode::Adjustable
                } else {
                    ControlMode::Hidden
                },
                supports_distill_strength: wan
                    && (normalized_model.ends_with("a14b:q4")
                        || normalized_model.ends_with("a14b:q5")),
                supports_first_last_frame: wan
                    && input.source_image != Some(SourceImageCapability::Unsupported),
                first_last_frame_min_frames: wan.then_some(validation::WAN_TI2V_FLF_MIN_FRAMES),
                reason: (!wan)
                    .then(|| "Wan sampler controls apply only to Wan models.".to_string()),
            },
            schedulers: match family {
                "sdxl" if normalized_model.starts_with("playground-v2.5") => {
                    vec![Scheduler::EdmDpmPp2m]
                }
                "sd15" | "sdxl" => {
                    vec![Scheduler::Ddim, Scheduler::EulerAncestral, Scheduler::UniPc]
                }
                "wan" => vec![Scheduler::UniPc, Scheduler::Euler, Scheduler::DpmPp],
                _ => Vec::new(),
            },
        },
        provenance: provenance(family),
    }
}

fn feature_control(
    supported: bool,
    required: bool,
    unsupported_reason: &'static str,
) -> FeatureControlProfile {
    FeatureControlProfile {
        mode: if supported {
            ControlMode::Adjustable
        } else {
            ControlMode::Hidden
        },
        required: supported && required,
        reason: (!supported).then(|| unsupported_reason.to_string()),
    }
}

fn temporal_profile(input: &GenerationProfileInput<'_>, family: &str) -> Option<TemporalProfile> {
    let step = validation::frame_step_for_family(family)?;
    let offset = validation::frame_offset_for_family(family).unwrap_or(1);
    let fps = input.default_fps.unwrap_or(validation::LTX2_DEFAULT_FPS);
    let max = if family == "minimax-h3" {
        345
    } else if family == "ltx2" {
        // The absolute resource guard (604) is not itself on LTX-2's 8n+1
        // request grid. Advertise the largest requestable value; admission
        // applies the lower duration-derived cap for the selected FPS.
        validation::max_frames_for_family_at_fps(family, 120)?
    } else {
        validation::max_frames_for_family_at_fps(family, fps)?
    };
    let min = validation::min_frames_for_family(family).unwrap_or(offset);
    let default = input.default_frames.unwrap_or(min).clamp(min, max);
    Some(TemporalProfile {
        frames: IntegerControl {
            default,
            min,
            max,
            step,
            recommended: vec![default],
            mode: ControlMode::Adjustable,
            note: None,
        },
        frame_offset: offset,
        fps: if let Some(fixed) = validation::fixed_fps_for_family(family) {
            FpsControl::Fixed { value: fixed }
        } else {
            FpsControl::Adjustable {
                default: fps,
                min: 1,
                max: 120,
                step: 1,
            }
        },
        max_duration_seconds: validation::max_runtime_seconds_for_family(family),
    })
}

fn aspect_groups(family: &str, dimensions: &[(u32, u32)]) -> Vec<AspectGroup> {
    let mut groups: Vec<AspectGroup> = Vec::new();
    for &(width, height) in dimensions {
        let id = authored_aspect_label(family, width, height);
        let preset = ResolutionPreset {
            id: format!("{width}x{height}"),
            width,
            height,
            tier: "recommended".to_string(),
        };
        if let Some(group) = groups.iter_mut().find(|group| group.id == id) {
            group.presets.push(preset);
        } else {
            groups.push(AspectGroup {
                label: id.clone(),
                id,
                presets: vec![preset],
            });
        }
    }
    for group in &mut groups {
        group
            .presets
            .sort_by_key(|preset| preset.width * preset.height);
    }
    groups
}

fn authored_aspect_label(family: &str, width: u32, height: u32) -> String {
    match (canonical_family(family), width, height) {
        ("qwen-image" | "qwen-image-edit", 1664, 928) => "≈16:9".to_string(),
        ("qwen-image" | "qwen-image-edit", 928, 1664) => "≈9:16".to_string(),
        _ => {
            let divisor = gcd(width, height);
            format!("{}:{}", width / divisor, height / divisor)
        }
    }
}

fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left.max(1)
}

fn identity_alignment(model: &str, family: &str, sub_family: Option<&str>) -> u32 {
    if family == "wan"
        && format!("{} {}", model, sub_family.unwrap_or_default())
            .to_ascii_lowercase()
            .contains("ti2v-5b")
    {
        32
    } else {
        validation::dimension_alignment_for_model(model, Some(family))
    }
}

fn pipeline_label(pipeline: Ltx2PipelineMode) -> String {
    pipeline
        .as_str()
        .split('-')
        .map(|part| {
            let mut chars = part.chars();
            chars
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
                .unwrap_or_default()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn provenance(family: &str) -> Vec<ProfileProvenance> {
    if canonical_family(family) == "qwen-image-edit" {
        return vec![ProfileProvenance {
            kind: ProvenanceKind::MoldPolicy,
            source: "Mold source-driven Qwen Image Edit guidance".to_string(),
            revision: None,
            qualified: true,
            evidence: Some(
                "source fitting preserves the input aspect on the dynamic /16 canvas and caps edit inputs at upstream's 1024x1024 VAE area; optional shape presets reuse Mold's qualified Qwen Image aspect set"
                    .to_string(),
            ),
        }];
    }
    if let Some(record) = resolution_qualification_record(family) {
        return vec![ProfileProvenance {
            kind: ProvenanceKind::Upstream,
            source: record.source.to_string(),
            revision: Some(record.revision.to_string()),
            qualified: record.qualified,
            evidence: Some(record.evidence.to_string()),
        }];
    }
    let (source, revision, evidence) = match family {
        "ltx-video" => (
            "https://github.com/Lightricks/LTX-Video",
            Some("4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"),
            "mold.generation-profile.v1",
        ),
        "ltx2" => (
            "https://github.com/Lightricks/LTX-2",
            Some("4f8905737aac86a554637cac86c178877a39c744"),
            "mold.generation-profile.v1",
        ),
        "wan" => (
            "https://github.com/Wan-Video/Wan2.2",
            Some("42bf4cfaa384bc21833865abc2f9e6c0e67233dc"),
            "mold.generation-profile.v1",
        ),
        "minimax-h3" => (
            "https://github.com/MiniMax-AI/MiniMax-H3",
            Some("fa6891ff7cdaaa03fa4497e89ac64ff169219acf"),
            "mold.generation-profile.v1",
        ),
        _ => (
            "mold-qualified compatibility profile",
            None,
            "mold.generation-profile.v1",
        ),
    };
    vec![ProfileProvenance {
        kind: if source.starts_with("http") {
            ProvenanceKind::Upstream
        } else {
            ProvenanceKind::MoldPolicy
        },
        source: source.to_string(),
        revision: revision.map(str::to_string),
        qualified: true,
        evidence: Some(evidence.to_string()),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(model: &'a str, family: &'a str) -> GenerationProfileInput<'a> {
        GenerationProfileInput {
            model,
            family,
            sub_family: None,
            default_width: 1024,
            default_height: 1024,
            default_steps: 20,
            default_guidance: 3.5,
            default_frames: None,
            default_fps: None,
            default_negative_prompt: None,
            source_image: None,
            supports_sequence: false,
            supports_extend: false,
            supports_audio: false,
        }
    }

    /// `Flux2Engine::load_transformer` refuses an adapter over an FP8-scaled
    /// checkpoint, because the merge widens the patched weight and drops its
    /// `weight_scale`. A profile that still advertised the control would
    /// offer a load that always fails.
    #[test]
    fn flux2_fp8_tiers_do_not_advertise_lora() {
        for model in ["flux2-klein:fp8", "flux2-klein-9b:fp8"] {
            let profile = resolve_generation_profile(input(model, "flux2"));
            let lora = profile.default_recipe().unwrap().capabilities.lora.mode;
            assert_eq!(lora, ControlMode::Hidden, "{model} must not offer LoRA");
        }
        // Every other Flux.2 tier still does.
        for model in ["flux2-klein:q8", "flux2-klein-base:q8", "flux2-klein:bf16"] {
            let profile = resolve_generation_profile(input(model, "flux2"));
            let lora = profile.default_recipe().unwrap().capabilities.lora.mode;
            assert_ne!(lora, ControlMode::Hidden, "{model} must still offer LoRA");
        }
    }

    #[test]
    fn identity_capability_is_off_in_a_build_that_cannot_execute_it() {
        // A binary that cannot execute identity conditioning must never
        // advertise the control, however qualified the checkpoint is. That
        // is the feature AND the landed runtime adapter, not the feature
        // alone.
        let profile = resolve_generation_profile(input("flux-dev:q8", "flux"));
        let advertised = profile
            .default_recipe()
            .unwrap()
            .capabilities
            .supports_identity;
        assert_eq!(advertised, crate::identity::identity_runtime_available());
        assert_eq!(
            profile.supports_identity(),
            crate::identity::identity_runtime_available()
        );
    }

    /// The capability arm must not ship a promise the worker cannot execute
    /// (the failure `CLAUDE.md` records for wan's `supports_sequence`). While
    /// the runtime adapter is pending, NO build advertises identity — the
    /// `pulid` feature included.
    #[test]
    fn identity_capability_is_unadvertised_while_the_runtime_adapter_is_pending() {
        if crate::identity::IDENTITY_RUNTIME_READY {
            return;
        }
        for model in crate::identity::identity_qualified_models() {
            let family = crate::identity::identity_family(model)
                .expect("qualified model has an identity family")
                .family();
            let profile = resolve_generation_profile(input(model, family));
            assert!(
                !profile.supports_identity(),
                "{model} must not advertise identity while the adapter is pending \
                 (feature pulid = {})",
                cfg!(feature = "pulid")
            );
            assert!(
                !profile
                    .default_recipe()
                    .unwrap()
                    .capabilities
                    .supports_identity,
                "{model}: the default recipe must not advertise it either"
            );
        }
    }

    #[test]
    fn identity_capability_is_never_advertised_outside_supported_families_or_for_turbo() {
        for (model, family) in [
            ("flux2-klein", "flux2"),
            ("sdxl-turbo:fp16", "sdxl"),
            ("z-image-turbo:q4", "z-image"),
        ] {
            let profile = resolve_generation_profile(input(model, family));
            assert!(
                !profile.supports_identity(),
                "{model} must not advertise identity conditioning"
            );
        }
    }

    #[cfg(feature = "pulid")]
    #[test]
    fn identity_capability_is_advertised_for_qualified_checkpoints_once_the_adapter_lands() {
        if !crate::identity::IDENTITY_RUNTIME_READY {
            // Pinned by `identity_capability_is_unadvertised_while_the_runtime_adapter_is_pending`
            // until #1221 flips the constant.
            return;
        }
        for model in crate::identity::identity_qualified_models() {
            let family = crate::identity::identity_family(model)
                .expect("qualified model has an identity family")
                .family();
            let profile = resolve_generation_profile(input(model, family));
            assert!(
                profile.supports_identity(),
                "{model} must advertise identity conditioning"
            );
        }
        for (model, family) in [("cv:123", "flux"), ("hf:owner/sdxl-finetune", "sdxl")] {
            let profile = resolve_generation_profile(input(model, family));
            assert!(
                profile.supports_identity(),
                "opaque catalog model {model} must inherit {family} identity support"
            );
        }
    }

    #[test]
    fn qualified_z_image_candidates_are_profile_recommendations() {
        let profile = resolve_generation_profile(input("z-image-turbo:q4", "z-image"));
        let recipe = profile.default_recipe().unwrap();
        let presets = recipe
            .resolution
            .aspect_groups
            .iter()
            .flat_map(|group| &group.presets)
            .map(|preset| (preset.width, preset.height))
            .collect::<std::collections::HashSet<_>>();
        let candidates = resolution_qualification_record("z-image").unwrap();
        assert!(candidates.qualified);
        assert_eq!(presets.len(), candidates.candidates.len());
        assert!(presets.contains(&(1280, 720)));
        assert!(presets.contains(&(720, 1280)));
    }

    #[test]
    fn wan_subfamily_selects_exact_checkpoint_contract() {
        let mut wan = input("cv:opaque", "wan");
        wan.sub_family = Some("wan22-ti2v-5b");
        let recipe = resolve_generation_profile(wan)
            .default_recipe()
            .unwrap()
            .clone();
        assert_eq!(recipe.resolution.alignment, 32);
        assert_eq!(recipe.resolution.aspect_groups.len(), 2);
        assert!(recipe.resolution.aspect_groups.iter().all(|group| {
            group
                .presets
                .iter()
                .all(|preset| preset.width == 1280 || preset.height == 1280)
        }));
    }

    #[test]
    fn h3_temporal_ceiling_is_valid_on_both_grids() {
        let mut h3 = input("minimax-h3-fl2va:official-bf16", "minimax-h3");
        h3.default_frames = Some(crate::minimax_h3::MIN_FRAMES);
        h3.default_fps = Some(24);
        let temporal = resolve_generation_profile(h3)
            .default_recipe()
            .unwrap()
            .temporal
            .clone()
            .unwrap();
        assert_eq!(temporal.frames.max, 345);
        assert_eq!(
            (temporal.frames.max - temporal.frame_offset) % temporal.frames.step,
            0
        );
        assert!(temporal.frames.max <= 15 * 24);
    }

    #[test]
    fn profile_hash_is_stable_and_content_addressed() {
        let left = resolve_generation_profile(input("flux-dev:q4", "flux"));
        let right = resolve_generation_profile(input("flux-dev:q4", "flux"));
        assert_eq!(left.profile_hash, right.profile_hash);
        assert_eq!(left.profile_hash.len(), 64);
    }

    #[test]
    fn advanced_and_delivery_controls_are_recipe_owned() {
        let sd15 = resolve_generation_profile(input("sd15-base:q4", "sd15"));
        let sd15_caps = &sd15.default_recipe().unwrap().capabilities;
        assert_eq!(sd15_caps.controlnet.mode, ControlMode::Adjustable);
        assert_eq!(sd15_caps.controlnet.max_count, 1);
        assert_eq!(
            sd15_caps.output.formats,
            vec![OutputFormat::Png, OutputFormat::Jpeg, OutputFormat::Webp]
        );

        let wan = resolve_generation_profile(input("wan22-t2v-a14b:fp8", "wan"));
        let wan_caps = &wan.default_recipe().unwrap().capabilities;
        assert_eq!(wan_caps.lora.mode, ControlMode::Hidden);
        assert_eq!(wan_caps.controlnet.mode, ControlMode::Hidden);
        assert_eq!(wan_caps.wan_recipe.mode, ControlMode::Adjustable);
        assert!(!wan_caps.wan_recipe.supports_distill_strength);
        assert_eq!(
            wan_caps.wan_recipe.first_last_frame_min_frames,
            Some(validation::WAN_TI2V_FLF_MIN_FRAMES)
        );
    }

    #[test]
    fn explicit_pipeline_lookup_never_falls_back_to_default_recipe() {
        let mut ltx = input("ltx2-distilled:q4", "ltx2");
        ltx.default_frames = Some(121);
        ltx.default_fps = Some(24);
        let profile = resolve_generation_profile(ltx);
        assert!(profile.recipe_for_pipeline(None).is_some());
        assert!(profile
            .recipe_for_pipeline(Some(Ltx2PipelineMode::T2a))
            .is_none());
    }

    #[test]
    fn t2a_dimensionless_profile_reaches_family_admission_with_zero_canvas() {
        let mut ltx = input("ltx-2.3-22b-dev:fp8", "ltx2");
        ltx.default_frames = Some(97);
        ltx.default_fps = Some(24);
        ltx.supports_audio = true;
        let profile = resolve_generation_profile(ltx);
        let recipe = profile
            .recipe_for_pipeline(Some(Ltx2PipelineMode::T2a))
            .unwrap();
        assert_eq!(recipe.resolution.domain, ResolutionDomain::None);
        let request: crate::GenerateRequest = serde_json::from_value(serde_json::json!({
            "prompt": "rain on a tin roof",
            "model": "ltx-2.3-22b-dev:fp8",
            "width": 0,
            "height": 0,
            "steps": recipe.defaults.steps,
            "guidance": recipe.defaults.guidance,
            "frames": recipe.defaults.frames,
            "fps": recipe.defaults.fps,
            "pipeline": "t2a",
            "output_format": "wav"
        }))
        .unwrap();
        validate_request_against_generation_profile(&profile, &request).unwrap();
        crate::validation::validate_generate_request_with_family(&request, Some("ltx2")).unwrap();
    }

    #[test]
    fn scheduler_contract_matches_engine_solver_families() {
        let sd = resolve_generation_profile(input("sdxl-base:q4", "sdxl"));
        assert_eq!(
            sd.default_recipe().unwrap().capabilities.schedulers,
            vec![Scheduler::Ddim, Scheduler::EulerAncestral, Scheduler::UniPc]
        );

        let mut wan_input = input("wan22-t2v-a14b:q5", "wan");
        wan_input.default_frames = Some(81);
        wan_input.default_fps = Some(16);
        let wan = resolve_generation_profile(wan_input);
        assert_eq!(
            wan.default_recipe().unwrap().capabilities.schedulers,
            vec![Scheduler::UniPc, Scheduler::Euler, Scheduler::DpmPp]
        );
        let mut request = request_for(&wan, 1280, 720);
        request.scheduler = Some(Scheduler::Ddim);
        assert!(validate_request_against_generation_profile(&wan, &request)
            .unwrap_err()
            .contains("not available"));

        let flux = resolve_generation_profile(input("flux-dev:q4", "flux"));
        let mut flux_request = request_for(&flux, 1024, 1024);
        flux_request.scheduler = Some(Scheduler::Euler);
        assert!(
            validate_request_against_generation_profile(&flux, &flux_request)
                .unwrap_err()
                .contains("not available")
        );
    }

    #[test]
    fn z_and_qwen_provenance_is_pinned_and_qualification_is_explicit() {
        for (model, family, revision, qualified, evidence_fragment) in [
            (
                "z-image-turbo:q4",
                "z-image",
                "768cb50d847cdbba97c89533ae976be69cf5a5b8",
                true,
                "docs/qualification/z-image-1024-tier-metal-q4.json",
            ),
            (
                "qwen-image:q4",
                "qwen-image",
                "6b5e1f5cec987d404be5ac6657db3b9aacb56a89",
                true,
                "no per-size runtime-performance claim",
            ),
        ] {
            let profile = resolve_generation_profile(input(model, family));
            let provenance = &profile.default_recipe().unwrap().provenance[0];
            assert_eq!(provenance.qualified, qualified);
            assert_eq!(provenance.revision.as_deref(), Some(revision));
            assert!(provenance.source.contains(revision));
            let evidence = provenance.evidence.as_deref().unwrap();
            assert!(evidence.contains(evidence_fragment));
        }
    }

    #[test]
    fn qwen_image_edit_presets_are_mold_source_fitting_guidance() {
        let profile =
            resolve_generation_profile(input("qwen-image-edit-2511:q4", "qwen-image-edit"));
        let recipe = profile.default_recipe().unwrap();
        assert_eq!(recipe.resolution.domain, ResolutionDomain::SourceDriven);
        assert_eq!(
            recipe.resolution.source_max_pixels,
            Some(validation::QWEN_IMAGE_EDIT_SOURCE_MAX_PIXELS)
        );
        assert_eq!(recipe.provenance[0].kind, ProvenanceKind::MoldPolicy);
        assert!(recipe.provenance[0].source.contains("source-driven"));
        assert!(resolution_qualification_record("qwen-image-edit").is_none());
    }

    #[test]
    fn qwen_candidates_are_profile_recommendations() {
        let profile = resolve_generation_profile(input("qwen-image:q4", "qwen-image"));
        let presets = profile
            .default_recipe()
            .unwrap()
            .resolution
            .aspect_groups
            .iter()
            .flat_map(|group| &group.presets)
            .map(|preset| (preset.width, preset.height))
            .collect::<std::collections::HashSet<_>>();
        let candidates = resolution_qualification_record("qwen-image").unwrap();
        assert!(candidates.qualified);
        assert_eq!(candidates.candidates, QWEN_UPSTREAM_CANDIDATES);
        assert_eq!(presets.len(), candidates.candidates.len());
        assert!(presets.contains(&(1664, 928)));
        assert!(presets.contains(&(928, 1664)));
    }

    #[test]
    fn adjustable_float_controls_enforce_the_advertised_step() {
        let profile = resolve_generation_profile(input("flux-dev:q4", "flux"));
        let mut request = request_for(&profile, 1024, 1024);
        request.guidance = 3.6;
        validate_request_against_generation_profile(&profile, &request).unwrap();
        request.guidance = 3.65;
        assert!(
            validate_request_against_generation_profile(&profile, &request)
                .unwrap_err()
                .contains("steps of 0.1")
        );
    }

    #[test]
    fn ltx2_frame_ceiling_tracks_the_requested_fps() {
        let mut ltx = input("ltx2-distilled:q4", "ltx2");
        ltx.default_width = 768;
        ltx.default_height = 512;
        ltx.default_frames = Some(121);
        ltx.default_fps = Some(24);
        let profile = resolve_generation_profile(ltx);
        assert_eq!(
            profile
                .default_recipe()
                .unwrap()
                .temporal
                .as_ref()
                .unwrap()
                .frames
                .max,
            validation::LTX2_MAX_FRAMES_ABSOLUTE - 3
        );

        let mut request = request_for(&profile, 768, 512);
        request.fps = Some(12);
        request.frames = Some(241);
        validate_request_against_generation_profile(&profile, &request).unwrap();
        request.frames = Some(249);
        assert!(
            validate_request_against_generation_profile(&profile, &request)
                .unwrap_err()
                .contains("frames must be")
        );

        request.fps = Some(120);
        request.frames = Some(601);
        validate_request_against_generation_profile(&profile, &request).unwrap();
    }

    #[test]
    fn every_shipped_manifest_profile_is_internally_admissible() {
        for manifest in crate::manifest::known_manifests()
            .iter()
            .filter(|manifest| manifest.is_generation_model())
        {
            let profile = generation_profile_for_manifest(manifest);
            assert_eq!(profile.schema_version, GENERATION_PROFILE_SCHEMA_VERSION);
            assert_eq!(profile.profile_hash.len(), 64, "{}", manifest.name);
            assert!(profile.default_recipe().is_some(), "{}", manifest.name);

            for recipe in &profile.recipes {
                let context = format!("{} recipe {}", manifest.name, recipe.id);
                assert!(
                    (recipe.steps.min..=recipe.steps.max).contains(&recipe.steps.default),
                    "{context}: step default is outside its control"
                );
                assert!(
                    (recipe.guidance.min..=recipe.guidance.max).contains(&recipe.guidance.default),
                    "{context}: guidance default is outside its control"
                );

                if let Some(temporal) = &recipe.temporal {
                    assert!(
                        (temporal.frames.min..=temporal.frames.max)
                            .contains(&temporal.frames.default),
                        "{context}: frame default is outside its control"
                    );
                    assert_eq!(
                        (temporal.frames.default - temporal.frame_offset) % temporal.frames.step,
                        0,
                        "{context}: frame default is off-grid"
                    );
                }

                if recipe.resolution.domain == ResolutionDomain::None {
                    assert!(recipe.resolution.aspect_groups.is_empty(), "{context}");
                    continue;
                }
                assert_resolution(
                    &context,
                    recipe,
                    recipe.defaults.width,
                    recipe.defaults.height,
                );
                let mut preset_ids = std::collections::HashSet::new();
                for group in &recipe.resolution.aspect_groups {
                    for preset in &group.presets {
                        assert!(preset_ids.insert(&preset.id), "{context}: duplicate preset");
                        assert_resolution(&context, recipe, preset.width, preset.height);
                    }
                }
                if recipe.resolution.domain == ResolutionDomain::Buckets {
                    assert!(
                        recipe
                            .resolution
                            .aspect_groups
                            .iter()
                            .flat_map(|group| &group.presets)
                            .any(|preset| {
                                preset.width == recipe.defaults.width
                                    && preset.height == recipe.defaults.height
                            }),
                        "{context}: bucket default is not advertised"
                    );
                }
            }
        }
    }

    fn assert_resolution(context: &str, recipe: &GenerationRecipeProfile, width: u32, height: u32) {
        let resolution = &recipe.resolution;
        assert!(width >= resolution.min_width, "{context}: width too small");
        assert!(
            height >= resolution.min_height,
            "{context}: height too small"
        );
        assert_eq!(width % resolution.alignment, 0, "{context}: width off-grid");
        assert_eq!(
            height % resolution.alignment,
            0,
            "{context}: height off-grid"
        );
        assert!(
            u64::from(width) * u64::from(height) <= resolution.max_pixels,
            "{context}: pixel ceiling exceeded"
        );
        if let Some(max_axis) = resolution.max_axis_pixels {
            assert!(
                width <= max_axis && height <= max_axis,
                "{context}: axis exceeded"
            );
        }
        let aspect = f64::from(width) / f64::from(height);
        if let Some(min) = resolution.min_aspect_ratio {
            assert!(aspect >= min, "{context}: aspect below minimum");
        }
        if let Some(max) = resolution.max_aspect_ratio {
            assert!(aspect <= max, "{context}: aspect above maximum");
        }
    }

    fn request_for(
        profile: &GenerationProfileSet,
        width: u32,
        height: u32,
    ) -> crate::GenerateRequest {
        let recipe = profile.default_recipe().unwrap();
        serde_json::from_value(serde_json::json!({
            "prompt": "test",
            "model": "test",
            "width": width,
            "height": height,
            "steps": recipe.defaults.steps,
            "guidance": recipe.defaults.guidance,
            "frames": recipe.defaults.frames,
            "fps": recipe.defaults.fps
        }))
        .unwrap()
    }

    #[test]
    fn profile_admission_accepts_z_wide_and_rejects_non_bucket_wan() {
        let z = resolve_generation_profile(input("z-image-turbo:q4", "z-image"));
        let z_request = request_for(&z, 1280, 720);
        validate_request_against_generation_profile(&z, &z_request).unwrap();

        let mut wan_input = input("wan22-t2v-a14b:q5", "wan");
        wan_input.default_width = 1280;
        wan_input.default_height = 720;
        wan_input.default_frames = Some(81);
        wan_input.default_fps = Some(16);
        let wan = resolve_generation_profile(wan_input);
        let valid = request_for(&wan, 1280, 720);
        validate_request_against_generation_profile(&wan, &valid).unwrap();

        // Wan's buckets are the trained sizes, not the only runnable ones: a
        // deliberate aligned off-bucket request is admitted (the advisory
        // dimension-warning channel says results may vary)...
        let off_bucket = request_for(&wan, 1024, 768);
        validate_request_against_generation_profile(&wan, &off_bucket).unwrap();

        // ...while a Reject-policy bucket profile (H3's reviewed bridge, and
        // every profile serialized before the field existed) still refuses.
        let mut strict = wan.clone();
        for recipe in &mut strict.recipes {
            recipe.resolution.off_bucket = Some(OffBucketPolicy::Reject);
        }
        assert!(
            validate_request_against_generation_profile(&strict, &off_bucket)
                .unwrap_err()
                .contains("not an available bucket")
        );

        // The advisory helper (the TUI's warning source) fires exactly for the
        // admitted off-bucket size — never for exact buckets, never for a
        // Reject-policy recipe, never for a refused shape.
        let recipe = wan.default_recipe().unwrap();
        assert!(off_bucket_resolution_warning(recipe, 1024, 768)
            .unwrap()
            .contains("results may vary"));
        assert!(off_bucket_resolution_warning(recipe, 1280, 720).is_none());
        assert!(off_bucket_resolution_warning(recipe, 1023, 768).is_none());
        let strict_recipe = strict.default_recipe().unwrap();
        assert!(off_bucket_resolution_warning(strict_recipe, 1024, 768).is_none());
    }

    #[test]
    fn resolution_advisory_downgrades_refusals_for_client_surfaces() {
        // The TUI (like the other shells) never blocks a custom size: a
        // recipe refusal becomes an advisory naming the server as authority,
        // an admitted warn-policy off-bucket keeps the softer message, and an
        // exact preset stays silent.
        let mut wan_input = input("wan22-t2v-a14b:q5", "wan");
        wan_input.default_width = 1280;
        wan_input.default_height = 720;
        wan_input.default_frames = Some(81);
        wan_input.default_fps = Some(16);
        let wan = resolve_generation_profile(wan_input);
        let recipe = wan.default_recipe().unwrap();
        let refused = resolution_advisory(recipe, 1023, 768).unwrap();
        assert!(refused.contains("server may reject"));
        assert!(resolution_advisory(recipe, 1024, 768)
            .unwrap()
            .contains("results may vary"));
        assert!(resolution_advisory(recipe, 1280, 720).is_none());
    }

    #[test]
    fn profile_admission_rejects_unadvertised_output_format() {
        let profile = resolve_generation_profile(input("flux-dev:q4", "flux"));
        let mut request = request_for(&profile, 1024, 1024);
        request.output_format = Some(OutputFormat::Mp4);
        assert!(
            validate_request_against_generation_profile(&profile, &request)
                .unwrap_err()
                .contains("output format 'mp4' is not available")
        );
    }

    #[test]
    fn delivery_qualification_repairs_defaults_and_withholds_mp4_only_recipes() {
        let mut ltx2_input = input("ltx2:bf16", "ltx2");
        ltx2_input.default_frames = Some(97);
        ltx2_input.default_fps = Some(24);
        let mut ltx2 = resolve_generation_profile(ltx2_input);
        let authored_hash = ltx2.profile_hash.clone();
        qualify_generation_profile_delivery(
            &mut ltx2,
            GenerationDeliveryCapabilities::new(false, false),
        );
        assert_ne!(ltx2.profile_hash, authored_hash);
        assert_eq!(
            generation_profile_default_output_format(&ltx2, None).unwrap(),
            OutputFormat::Gif
        );
        let default = ltx2.default_recipe().unwrap();
        assert_eq!(
            default.capabilities.output.formats,
            vec![OutputFormat::Gif, OutputFormat::Apng]
        );
        assert!(!default.capabilities.supports_audio);

        let mut request = request_for(&ltx2, 1216, 704);
        request.output_format = None;
        materialize_generation_profile_output_default(&ltx2, &mut request).unwrap();
        assert_eq!(request.output_format, Some(OutputFormat::Gif));
        request.output_format = Some(OutputFormat::Apng);
        materialize_generation_profile_output_default(&ltx2, &mut request).unwrap();
        assert_eq!(request.output_format, Some(OutputFormat::Apng));

        let mut h3_input = input("minimax-h3-fl2va:official-bf16", "minimax-h3");
        h3_input.default_frames = Some(crate::minimax_h3::MIN_FRAMES);
        h3_input.default_fps = Some(crate::minimax_h3::FIXED_FPS);
        let mut h3 = resolve_generation_profile(h3_input);
        qualify_generation_profile_delivery(
            &mut h3,
            GenerationDeliveryCapabilities::new(false, false),
        );
        assert!(h3.recipes.is_empty());
        assert!(generation_profile_default_output_format(&h3, None)
            .unwrap_err()
            .contains("no default recipe"));
    }

    #[test]
    fn h3_compact_profiles_pin_the_reviewed_envelope() {
        // The reviewed compact stack admits exactly one canvas, one step
        // count, and one frame count. The laundered manifest defaults below
        // are deliberately wrong (a stale user `model_pref` reaches this
        // input through `build_model_catalog`), so the profile must derive
        // the envelope from the identity rather than from what it was told.
        for (model, steps) in [
            (crate::minimax_h3::FL2VA_COMFY, 21),
            (crate::minimax_h3::REF2VA_COMFY, 21),
            (crate::minimax_h3::FL2VA_COMFY_TURBO_8STEP, 9),
            (crate::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P, 5),
        ] {
            let turbo = crate::minimax_h3::turbo_tier_for_model(model).is_some();
            let mut h3_input = input(model, "minimax-h3");
            h3_input.default_width = 768;
            h3_input.default_height = 768;
            h3_input.default_steps = 30;
            h3_input.default_frames = Some(crate::minimax_h3::MAX_FRAMES);
            h3_input.default_fps = Some(crate::minimax_h3::FIXED_FPS);
            let profile = resolve_generation_profile(h3_input);
            let recipe = profile.default_recipe().unwrap();

            // The compact stack is a RANGE, not a bucket set: any 32-aligned
            // canvas inside its area ceiling is admitted and the memory
            // estimate decides what fits.
            assert_eq!(
                recipe.resolution.domain,
                ResolutionDomain::Dynamic,
                "{model}"
            );
            assert_eq!(recipe.resolution.off_bucket, None, "{model}");
            assert_eq!(
                recipe.resolution.alignment,
                crate::minimax_h3::VIDEO_ROW_STRIDE,
                "{model}"
            );
            assert_eq!(
                recipe.resolution.min_width,
                crate::minimax_h3::MIN_COMPACT_AXIS_PIXELS,
                "{model}"
            );
            assert_eq!(
                recipe.resolution.min_height,
                crate::minimax_h3::MIN_COMPACT_AXIS_PIXELS,
                "{model}"
            );
            let presets = recipe
                .resolution
                .aspect_groups
                .iter()
                .flat_map(|group| &group.presets)
                .map(|preset| (preset.width, preset.height))
                .collect::<Vec<_>>();
            for preset in &presets {
                assert!(
                    crate::minimax_h3::is_admitted_compact_canvas(preset.0, preset.1),
                    "{model}: recommended {preset:?} must satisfy the canvas rule"
                );
            }
            assert!(
                presets.contains(&(
                    crate::minimax_h3::DEFAULT_WIDTH,
                    crate::minimax_h3::DEFAULT_HEIGHT
                )),
                "{model}"
            );
            assert_eq!(
                recipe.resolution.max_pixels,
                crate::minimax_h3::reviewed_compact_max_pixels(),
                "{model}"
            );
            assert_eq!(
                recipe.resolution.max_axis_pixels,
                Some(crate::minimax_h3::reviewed_compact_max_axis_pixels()),
                "{model}"
            );

            assert_eq!(recipe.defaults.width, crate::minimax_h3::DEFAULT_WIDTH);
            assert_eq!(recipe.defaults.height, crate::minimax_h3::DEFAULT_HEIGHT);
            assert_eq!(recipe.defaults.steps, steps, "{model}");
            assert_eq!(
                recipe.defaults.frames,
                Some(crate::minimax_h3::DEFAULT_COMPACT_FRAMES),
                "{model}"
            );

            // A Turbo tier's step count is the distilled adapter's own
            // schedule length and stays fixed; the base tag takes a range.
            if turbo {
                assert_eq!(recipe.steps.mode, ControlMode::Fixed, "{model}");
                assert_eq!(recipe.steps.min, steps, "{model}");
                assert_eq!(recipe.steps.max, steps, "{model}");
            } else {
                assert_eq!(recipe.steps.mode, ControlMode::Adjustable, "{model}");
                assert_eq!(
                    recipe.steps.min,
                    crate::minimax_h3::COMPACT_MIN_STEPS,
                    "{model}"
                );
                assert_eq!(
                    recipe.steps.max,
                    crate::minimax_h3::COMPACT_MAX_STEPS,
                    "{model}"
                );
            }
            assert_eq!(recipe.steps.default, steps, "{model}");
            assert_eq!(recipe.steps.recommended, vec![steps], "{model}");

            let temporal = recipe.temporal.as_ref().unwrap();
            assert_eq!(temporal.frames.mode, ControlMode::Adjustable, "{model}");
            assert_eq!(
                temporal.frames.min,
                crate::minimax_h3::MIN_FRAMES,
                "{model}"
            );
            assert_eq!(
                temporal.frames.max,
                crate::minimax_h3::MAX_FRAMES,
                "{model}"
            );
            assert_eq!(
                temporal.frames.step,
                crate::minimax_h3::FRAME_STEP,
                "{model}"
            );
            assert_eq!(
                temporal.frames.default,
                crate::minimax_h3::DEFAULT_COMPACT_FRAMES,
                "{model}"
            );

            let mut request = request_for(
                &profile,
                crate::minimax_h3::DEFAULT_WIDTH,
                crate::minimax_h3::DEFAULT_HEIGHT,
            );
            request.output_format = Some(OutputFormat::Mp4);
            validate_request_against_generation_profile(&profile, &request).unwrap();

            // Every recommended canvas is submittable, and so is a canvas no
            // campaign ran that the rule admits.
            for &(width, height) in crate::minimax_h3::REVIEWED_COMPACT_CANVASES {
                let mut reviewed = request_for(&profile, width, height);
                reviewed.output_format = Some(OutputFormat::Mp4);
                validate_request_against_generation_profile(&profile, &reviewed)
                    .unwrap_or_else(|error| panic!("{model} {width}x{height}: {error}"));
            }
            for (width, height) in [(1024, 768), (1024, 576), (512, 1984)] {
                let mut off_preset = request_for(&profile, width, height);
                off_preset.output_format = Some(OutputFormat::Mp4);
                validate_request_against_generation_profile(&profile, &off_preset)
                    .unwrap_or_else(|error| panic!("{model} {width}x{height}: {error}"));
            }
            // Over the area ceiling is still refused, by the ceiling.
            let mut too_large = request_for(&profile, 1056, 992);
            too_large.output_format = Some(OutputFormat::Mp4);
            assert!(
                validate_request_against_generation_profile(&profile, &too_large).is_err(),
                "{model}"
            );

            // Another clip length on the family grid is submittable now.
            let mut longer = request.clone();
            longer.frames =
                Some(crate::minimax_h3::DEFAULT_COMPACT_FRAMES + crate::minimax_h3::FRAME_STEP);
            validate_request_against_generation_profile(&profile, &longer)
                .unwrap_or_else(|error| panic!("{model}: {error}"));
            let mut off_grid = request.clone();
            off_grid.frames = Some(crate::minimax_h3::DEFAULT_COMPACT_FRAMES + 1);
            assert!(
                validate_request_against_generation_profile(&profile, &off_grid).is_err(),
                "{model}"
            );

            let mut wrong_steps = request.clone();
            wrong_steps.steps = 30;
            let outcome = validate_request_against_generation_profile(&profile, &wrong_steps);
            if turbo {
                assert!(
                    outcome.unwrap_err().contains("steps is fixed at"),
                    "{model}"
                );
            } else {
                outcome.unwrap_or_else(|error| panic!("{model}: {error}"));
            }
            let mut too_many_steps = request.clone();
            too_many_steps.steps = crate::minimax_h3::COMPACT_MAX_STEPS + 1;
            assert!(
                validate_request_against_generation_profile(&profile, &too_many_steps).is_err(),
                "{model}"
            );
        }
    }

    #[test]
    fn h3_official_profiles_keep_the_flexible_ladder() {
        for model in [
            crate::minimax_h3::FL2VA_OFFICIAL,
            crate::minimax_h3::REF2VA_OFFICIAL,
        ] {
            let mut h3_input = input(model, "minimax-h3");
            h3_input.default_width = crate::minimax_h3::DEFAULT_WIDTH;
            h3_input.default_height = crate::minimax_h3::DEFAULT_HEIGHT;
            h3_input.default_steps = crate::minimax_h3::DEFAULT_STEPS;
            h3_input.default_frames = Some(crate::minimax_h3::MIN_FRAMES);
            h3_input.default_fps = Some(crate::minimax_h3::FIXED_FPS);
            let profile = resolve_generation_profile(h3_input);
            let recipe = profile.default_recipe().unwrap();

            assert_eq!(
                recipe.resolution.domain,
                ResolutionDomain::Dynamic,
                "{model}"
            );
            assert_eq!(recipe.resolution.off_bucket, None, "{model}");
            assert!(
                recipe.resolution.aspect_groups.len() >= 2,
                "{model}: the official ladder keeps every reviewed aspect"
            );
            assert_eq!(recipe.steps.mode, ControlMode::Adjustable, "{model}");
            assert_eq!(recipe.steps.min, 2, "{model}");
            assert_eq!(recipe.steps.max, 100, "{model}");

            let temporal = recipe.temporal.as_ref().unwrap();
            assert_eq!(temporal.frames.mode, ControlMode::Adjustable, "{model}");
            assert_eq!(
                temporal.frames.max,
                crate::minimax_h3::MAX_FRAMES,
                "{model}"
            );
        }
    }

    #[test]
    fn profile_admission_enforces_h3_fixed_controls_and_frame_cap() {
        let mut h3_input = input("minimax-h3-fl2va:official-bf16", "minimax-h3");
        h3_input.default_width = 768;
        h3_input.default_height = 768;
        h3_input.default_frames = Some(crate::minimax_h3::MIN_FRAMES);
        h3_input.default_fps = Some(crate::minimax_h3::FIXED_FPS);
        let h3 = resolve_generation_profile(h3_input);
        let mut request = request_for(&h3, 768, 768);
        request.frames = Some(crate::minimax_h3::MAX_FRAMES);
        validate_request_against_generation_profile(&h3, &request).unwrap();

        request.frames = Some(362);
        assert!(validate_request_against_generation_profile(&h3, &request)
            .unwrap_err()
            .contains("frames must be"));
        request.frames = Some(crate::minimax_h3::MAX_FRAMES);
        request.guidance = 1.0;
        assert!(validate_request_against_generation_profile(&h3, &request)
            .unwrap_err()
            .contains("guidance is fixed"));
    }

    /// A fixed control's explanation is authored server-side, at the one
    /// place the fixedness is decided. Clients render it verbatim; the old
    /// hard-coded FLUX/LTX sentence claimed "Distilled recipe fixes CFG at
    /// 1.0" for H3, whose guidance is pinned at 0 by a pipeline that has no
    /// classifier-free branch and no Dev checkpoint to switch to.
    #[test]
    fn h3_guidance_carries_its_own_fixed_control_note() {
        for model in [
            crate::minimax_h3::FL2VA_COMFY,
            crate::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
            crate::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P,
        ] {
            let profile = resolve_generation_profile(input(model, "minimax-h3"));
            let recipe = profile.default_recipe().unwrap();
            assert_eq!(recipe.guidance.mode, ControlMode::Fixed);
            assert_eq!(
                recipe.guidance.note.as_deref(),
                Some("MiniMax H3 does not use classifier-free guidance; guidance is fixed at 0."),
                "{model}"
            );
        }
    }

    /// The base compact tag takes a step RANGE, so there is nothing to
    /// explain and no note is authored — a client renders nothing rather
    /// than inventing copy.
    #[test]
    fn h3_base_steps_are_adjustable_and_carry_no_note() {
        let profile =
            resolve_generation_profile(input(crate::minimax_h3::FL2VA_COMFY, "minimax-h3"));
        let steps = &profile.default_recipe().unwrap().steps;
        assert_eq!(steps.mode, ControlMode::Adjustable);
        assert_eq!(steps.min, crate::minimax_h3::COMPACT_MIN_STEPS);
        assert_eq!(steps.max, crate::minimax_h3::COMPACT_MAX_STEPS);
        assert_eq!(steps.note, None);
    }

    /// A Turbo tier's step count is terminal-inclusive: the published N-step
    /// schedule has N denoise intervals and N+1 sampler grid points, which is
    /// why the field reads 9 for the 8-step tier.
    #[test]
    fn h3_turbo_steps_explain_the_terminal_inclusive_count() {
        let eight = resolve_generation_profile(input(
            crate::minimax_h3::FL2VA_COMFY_TURBO_8STEP,
            "minimax-h3",
        ));
        let steps = &eight.default_recipe().unwrap().steps;
        assert_eq!(steps.mode, ControlMode::Fixed);
        assert_eq!(steps.default, 9);
        assert_eq!(
            steps.note.as_deref(),
            Some(
                "Fixed by the 8-step Turbo tier: 9 terminal-inclusive sampler grid points \
                 (8 denoise intervals)."
            )
        );

        let four = resolve_generation_profile(input(
            crate::minimax_h3::FL2VA_COMFY_TURBO_4STEP_768P,
            "minimax-h3",
        ));
        let steps = &four.default_recipe().unwrap().steps;
        assert_eq!(steps.mode, ControlMode::Fixed);
        assert_eq!(steps.default, 5);
        assert_eq!(
            steps.note.as_deref(),
            Some(
                "Fixed by the 4-step Turbo tier: 5 terminal-inclusive sampler grid points \
                 (4 denoise intervals)."
            )
        );
    }

    /// The distilled FLUX/LTX case keeps the sentence clients used to
    /// hard-code, now generated from the value the recipe actually pinned.
    #[test]
    fn a_distilled_recipe_keeps_the_existing_fixed_cfg_sentence() {
        const DISTILLED: &str = "Distilled recipe fixes CFG at 1.0. Choose a Dev checkpoint with \
                                 Auto or a guided pipeline to adjust it.";
        let ltx = resolve_generation_profile(input("ltx-2.3-22b-distilled:fp8", "ltx2"));
        let recipe = ltx.default_recipe().unwrap();
        assert_eq!(recipe.guidance.mode, ControlMode::Fixed);
        assert_eq!(recipe.guidance.note.as_deref(), Some(DISTILLED));

        let ltx_video = resolve_generation_profile(input("ltx-video-distilled", "ltx-video"));
        assert_eq!(
            ltx_video.default_recipe().unwrap().guidance.note.as_deref(),
            Some(DISTILLED)
        );
    }

    /// An adjustable control has nothing to explain.
    #[test]
    fn an_adjustable_control_carries_no_note() {
        let flux = resolve_generation_profile(input("flux-dev:q8", "flux"));
        let recipe = flux.default_recipe().unwrap();
        assert_eq!(recipe.guidance.mode, ControlMode::Adjustable);
        assert_eq!(recipe.guidance.note, None);
        assert_eq!(recipe.steps.note, None);
        assert_eq!(
            recipe.temporal.as_ref().and_then(|t| t.frames.note.clone()),
            None
        );
    }

    /// The field is additive: an older server never sends it, and that
    /// response must deserialize to `None` rather than failing.
    #[test]
    fn an_absent_note_deserializes_to_none() {
        let integer: IntegerControl = serde_json::from_str(
            r#"{"default":20,"min":1,"max":100,"step":1,"mode":"adjustable"}"#,
        )
        .unwrap();
        assert_eq!(integer.note, None);
        let float: FloatControl = serde_json::from_str(
            r#"{"default":3.5,"min":0.0,"max":100.0,"step":0.1,"mode":"adjustable"}"#,
        )
        .unwrap();
        assert_eq!(float.note, None);

        // An absent note is never serialized, so a recipe carrying none is
        // byte-identical to a pre-`note` build's.
        let encoded = serde_json::to_string(&FloatControl {
            default: 0.0,
            min: 0.0,
            max: 0.0,
            step: 0.1,
            mode: ControlMode::Fixed,
            note: None,
        })
        .unwrap();
        assert!(!encoded.contains("note"), "{encoded}");
    }
}
