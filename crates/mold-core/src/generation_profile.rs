//! Canonical, versioned generation-control profiles.
//!
//! A profile is the one model/recipe authority consumed by admission, Rust
//! clients, and `/api/models`. Browser clients receive the same fully-resolved
//! recipes and never reconstruct family policy. The legacy flattened model
//! defaults remain a derived compatibility view for one release.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{validation, GuidanceCapabilities, Ltx2PipelineMode, Scheduler, SourceImageCapability};

pub const GENERATION_PROFILE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ResolutionDomain {
    Dynamic,
    Buckets,
    SourceDriven,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ControlMode {
    Adjustable,
    Fixed,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "kebab-case")]
pub enum ProvenanceKind {
    Upstream,
    MoldPolicy,
    Derived,
    DeliveryLimit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ProfileProvenance {
    pub kind: ProvenanceKind,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    pub qualified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ResolutionPreset {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub tier: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct AspectGroup {
    pub id: String,
    pub label: String,
    pub presets: Vec<ResolutionPreset>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ResolutionProfile {
    pub domain: ResolutionDomain,
    pub alignment: u32,
    pub min_width: u32,
    pub min_height: u32,
    pub max_pixels: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_axis_pixels: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_aspect_ratio: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_aspect_ratio: Option<f64>,
    pub aspect_groups: Vec<AspectGroup>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct IntegerControl {
    pub default: u32,
    pub min: u32,
    pub max: u32,
    pub step: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recommended: Vec<u32>,
    pub mode: ControlMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct FloatControl {
    pub default: f64,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub mode: ControlMode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(tag = "mode", rename_all = "kebab-case")]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct TemporalProfile {
    pub frames: IntegerControl,
    pub frame_offset: u32,
    pub fps: FpsControl,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_duration_seconds: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct RecipeSelector {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<Ltx2PipelineMode>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct GenerationCapabilitiesProfile {
    pub guidance: GuidanceCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image: Option<SourceImageCapability>,
    pub supports_lora: bool,
    pub supports_controlnet: bool,
    pub supports_sequence: bool,
    pub supports_extend: bool,
    pub supports_audio: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedulers: Vec<Scheduler>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, utoipa::ToSchema)]
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
        if !advertised.is_empty() && !advertised.contains(&scheduler) {
            return Err(format!(
                "scheduler '{scheduler}' is not available for this recipe"
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
        supports_audio: family == "ltx2",
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
/// Official Z-Image-Turbo 1024-tier buckets. Every entry fits Mold's current
/// 1.8 MP resource ceiling and /16 runtime grid.
const Z_IMAGE: &[(u32, u32)] = &[
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
/// Current official Qwen-Image standard buckets.
const QWEN: &[(u32, u32)] = &[
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

pub fn family_presets(family: &str) -> &'static [(u32, u32)] {
    match canonical_family(family) {
        "sd15" => SD15,
        "sdxl" => SDXL,
        "sd3" => SD3,
        "flux" | "flux2" => FLUX,
        "z-image" => Z_IMAGE,
        "qwen-image" | "qwen-image-edit" => QWEN,
        "wuerstchen" => WUERSTCHEN,
        "ltx-video" => LTX_VIDEO,
        "ltx2" => LTX2,
        "wan" => WAN_480_720,
        "minimax-h3" => H3,
        _ => &[],
    }
}

pub fn presets_for_identity<'a>(
    model: &str,
    family: &str,
    sub_family: Option<&str>,
) -> &'a [(u32, u32)] {
    let family = canonical_family(family);
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

fn recipe(
    input: &GenerationProfileInput<'_>,
    id: &str,
    label: &str,
    pipeline: Option<Ltx2PipelineMode>,
) -> GenerationRecipeProfile {
    let family = canonical_family(input.family);
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
            max_axis_pixels: None,
            min_aspect_ratio: None,
            max_aspect_ratio: None,
            aspect_groups: Vec::new(),
        }
    } else {
        ResolutionProfile {
            domain: if source_driven {
                ResolutionDomain::SourceDriven
            } else if family == "wan" {
                ResolutionDomain::Buckets
            } else {
                ResolutionDomain::Dynamic
            },
            alignment,
            min_width: alignment.max(64),
            min_height: alignment.max(64),
            max_pixels: validation::max_pixels_for_family_composed(Some(family), composition),
            max_axis_pixels: validation::max_axis_pixels_for_family_composed(
                Some(family),
                composition,
            ),
            min_aspect_ratio: (family == "minimax-h3")
                .then_some(crate::minimax_h3::MIN_ASPECT_RATIO),
            max_aspect_ratio: (family == "minimax-h3")
                .then_some(crate::minimax_h3::MAX_ASPECT_RATIO),
            aspect_groups: aspect_groups(&dimensions),
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
    let temporal = temporal_profile(input, family);
    let defaults = GenerationDefaultsProfile {
        width: if audio_only { 0 } else { input.default_width },
        height: if audio_only { 0 } else { input.default_height },
        steps: input.default_steps,
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
            default: input.default_steps,
            min: if family == "minimax-h3" { 2 } else { 1 },
            max: 100,
            step: 1,
            recommended: vec![input.default_steps],
            mode: ControlMode::Adjustable,
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
        },
        temporal,
        capabilities: GenerationCapabilitiesProfile {
            guidance: guidance_caps,
            source_image: input.source_image,
            supports_lora: validation::family_supports_lora(family),
            supports_controlnet: matches!(family, "sd15" | "sdxl" | "flux"),
            supports_sequence: input.supports_sequence && !audio_only,
            supports_extend: input.supports_extend && !audio_only,
            supports_audio: input.supports_audio || audio_only,
            schedulers: match family {
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

fn aspect_groups(dimensions: &[(u32, u32)]) -> Vec<AspectGroup> {
    let mut groups: Vec<AspectGroup> = Vec::new();
    for &(width, height) in dimensions {
        let divisor = gcd(width, height);
        let id = format!("{}:{}", width / divisor, height / divisor);
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
    let (source, revision) = match family {
        "z-image" => (
            "https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/app.py",
            None,
        ),
        "qwen-image" | "qwen-image-edit" => ("https://github.com/QwenLM/Qwen-Image", None),
        "ltx-video" => (
            "https://github.com/Lightricks/LTX-Video",
            Some("4b2d053057623ddd4d0a1d3e9cd28890e9ef487f"),
        ),
        "ltx2" => (
            "https://github.com/Lightricks/LTX-2",
            Some("4f8905737aac86a554637cac86c178877a39c744"),
        ),
        "wan" => (
            "https://github.com/Wan-Video/Wan2.2",
            Some("42bf4cfaa384bc21833865abc2f9e6c0e67233dc"),
        ),
        "minimax-h3" => (
            "https://github.com/MiniMax-AI/MiniMax-H3",
            Some("fa6891ff7cdaaa03fa4497e89ac64ff169219acf"),
        ),
        _ => ("mold-qualified compatibility profile", None),
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
        evidence: Some("mold.generation-profile.v1".to_string()),
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

    #[test]
    fn z_image_profile_contains_exact_wide_and_tall_buckets() {
        let profile = resolve_generation_profile(input("z-image-turbo:q4", "z-image"));
        let recipe = profile.default_recipe().unwrap();
        assert!(recipe.resolution.aspect_groups.iter().any(|group| {
            group.id == "16:9"
                && group
                    .presets
                    .iter()
                    .any(|preset| preset.width == 1280 && preset.height == 720)
        }));
        assert!(recipe.resolution.aspect_groups.iter().any(|group| {
            group.id == "9:16"
                && group
                    .presets
                    .iter()
                    .any(|preset| preset.width == 720 && preset.height == 1280)
        }));
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
        let invalid = request_for(&wan, 1024, 768);
        assert!(validate_request_against_generation_profile(&wan, &invalid)
            .unwrap_err()
            .contains("not an available bucket"));
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
}
