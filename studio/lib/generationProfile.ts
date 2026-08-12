/** Versioned generation-control contract emitted by `/api/models`. */

export type ResolutionDomain = "dynamic" | "buckets" | "source-driven" | "none";
export type ControlMode = "adjustable" | "fixed" | "hidden";

export interface ProfileResolutionPreset {
  id: string;
  width: number;
  height: number;
  tier: string;
}

export interface ProfileAspectGroup {
  id: string;
  label: string;
  presets: ProfileResolutionPreset[];
}

export interface ResolutionProfile {
  domain: ResolutionDomain;
  alignment: number;
  min_width: number;
  min_height: number;
  max_pixels: number;
  max_axis_pixels?: number | null;
  min_aspect_ratio?: number | null;
  max_aspect_ratio?: number | null;
  aspect_groups: ProfileAspectGroup[];
}

export interface IntegerControl {
  default: number;
  min: number;
  max: number;
  step: number;
  recommended?: number[];
  mode: ControlMode;
}

export interface FloatControl {
  default: number;
  min: number;
  max: number;
  step: number;
  mode: ControlMode;
}

export type ProfileFpsControl =
  | { mode: "fixed"; value: number }
  | {
      mode: "adjustable";
      default: number;
      min: number;
      max: number;
      step: number;
    };

export interface TemporalProfile {
  frames: IntegerControl;
  frame_offset: number;
  fps: ProfileFpsControl;
  max_duration_seconds?: number | null;
}

export interface GenerationRecipeProfile {
  id: string;
  label: string;
  request_selector: { pipeline?: string | null };
  defaults: {
    width: number;
    height: number;
    steps: number;
    guidance: number;
    frames?: number | null;
    fps?: number | null;
    negative_prompt?: string | null;
  };
  resolution: ResolutionProfile;
  steps: IntegerControl;
  guidance: FloatControl;
  temporal?: TemporalProfile | null;
  capabilities: {
    guidance: {
      adjustable: boolean;
      supports_negative_prompt: boolean;
      fixed_scale?: number | null;
    };
    source_image?: "unsupported" | "optional" | "required" | null;
    supports_lora: boolean;
    supports_controlnet: boolean;
    supports_sequence: boolean;
    supports_extend: boolean;
    supports_audio: boolean;
    schedulers: string[];
  };
}

export interface GenerationProfileSet {
  schema_version: 1;
  profile_id: string;
  profile_hash: string;
  default_recipe_id: string;
  recipes: GenerationRecipeProfile[];
}

export interface GenerationProfileModel {
  name?: string;
  family: string;
  default_width?: number;
  default_height?: number;
  default_steps?: number;
  default_guidance?: number;
  default_frames?: number | null;
  default_fps?: number | null;
  min_frames?: number | null;
  max_frames?: number | null;
  max_runtime_seconds?: number | null;
  frame_step?: number | null;
  frame_offset?: number | null;
  max_pixels?: number | null;
  max_axis_pixels?: number | null;
  dimension_alignment?: number | null;
  recommended_dimensions?: { width: number; height: number }[] | null;
  generation_profile?: GenerationProfileSet | null;
}

/** Strictly accept the schema this client understands. */
export function advertisedGenerationProfile(
  model: GenerationProfileModel | null | undefined,
): GenerationProfileSet | null {
  const profile = model?.generation_profile;
  if (
    !profile ||
    profile.schema_version !== 1 ||
    !profile.profile_id ||
    !profile.profile_hash ||
    !profile.default_recipe_id ||
    !Array.isArray(profile.recipes) ||
    profile.recipes.length === 0
  ) {
    return null;
  }
  return profile;
}

/** Resolve one complete recipe; clients never merge recipe overrides. */
export function effectiveGenerationRecipe(
  model: GenerationProfileModel | null | undefined,
  pipeline?: string | null,
): GenerationRecipeProfile | null {
  const profile = advertisedGenerationProfile(model);
  if (!profile) {
    const hasLegacyContract =
      model &&
      (Number.isFinite(model.default_width) ||
        Number.isFinite(model.default_height) ||
        Boolean(model.recommended_dimensions?.length));
    return hasLegacyContract ? legacyRecipe(model, pipeline) : null;
  }
  if (pipeline) {
    const exact = profile.recipes.find(
      (recipe) => recipe.request_selector.pipeline === pipeline,
    );
    if (exact) return exact;
  }
  return (
    profile.recipes.find((recipe) => recipe.id === profile.default_recipe_id) ??
    profile.recipes[0] ??
    null
  );
}

/** One-release bridge for servers that predate `generation_profile_v1`. */
function legacyRecipe(
  model: GenerationProfileModel,
  pipeline?: string | null,
): GenerationRecipeProfile {
  const family = model.family.trim().toLowerCase();
  const alignment = positiveInteger(model.dimension_alignment, 16);
  const refining =
    family === "ltx2" &&
    (!pipeline ||
      [
        "distilled",
        "two-stage",
        "two-stage-hq",
        "ic-lora",
        "keyframe",
        "a2-vid",
      ].includes(pipeline));
  const maxPixels = refining
    ? positiveInteger(model.max_pixels, 4_096 * 2_176)
    : family === "ltx2"
      ? Math.min(
          positiveInteger(model.max_pixels, 1_920 * 1_088),
          1_920 * 1_088,
        )
      : positiveInteger(model.max_pixels, 1_800_000);
  const advertisedAxis =
    model.max_axis_pixels ?? (family === "ltx2" ? 2_048 : null);
  const maxAxisPixels =
    advertisedAxis === null
      ? null
      : refining
        ? advertisedAxis
        : Math.min(advertisedAxis, 2_048);
  const dimensions = (
    model.recommended_dimensions?.filter(
      ({ width, height }) => width > 0 && height > 0,
    ) ?? []
  ).filter(
    ({ width, height }) =>
      width % alignment === 0 &&
      height % alignment === 0 &&
      width * height <= maxPixels &&
      (maxAxisPixels === null || Math.max(width, height) <= maxAxisPixels),
  );
  if (dimensions.length === 0) {
    dimensions.push({
      width: positiveInteger(model.default_width, Math.max(64, alignment)),
      height: positiveInteger(model.default_height, Math.max(64, alignment)),
    });
  }
  const aspectGroups = groupDimensions(dimensions);
  const fps = positiveInteger(model.default_fps, 24);
  const temporal = model.default_frames
    ? {
        frames: {
          default: model.default_frames,
          min: positiveInteger(model.min_frames, 1),
          max: positiveInteger(model.max_frames, model.default_frames),
          step: positiveInteger(model.frame_step, 1),
          recommended: [model.default_frames],
          mode: "adjustable" as const,
        },
        frame_offset: positiveInteger(model.frame_offset, 1),
        fps: {
          mode: "adjustable" as const,
          default: fps,
          min: 1,
          max: 60,
          step: 1,
        },
        max_duration_seconds: model.max_runtime_seconds ?? null,
      }
    : null;
  return {
    id: "legacy",
    label: "Default",
    request_selector: {},
    defaults: {
      width: positiveInteger(model.default_width, Math.max(64, alignment)),
      height: positiveInteger(model.default_height, Math.max(64, alignment)),
      steps: positiveInteger(model.default_steps, 20),
      guidance: Number.isFinite(model.default_guidance)
        ? Number(model.default_guidance)
        : 3.5,
      ...(model.default_frames !== undefined
        ? { frames: model.default_frames }
        : {}),
      ...(model.default_fps !== undefined ? { fps: model.default_fps } : {}),
    },
    resolution: {
      domain: "dynamic",
      alignment,
      min_width: Math.max(64, alignment),
      min_height: Math.max(64, alignment),
      max_pixels: maxPixels,
      max_axis_pixels: maxAxisPixels,
      aspect_groups: aspectGroups,
    },
    steps: {
      default: positiveInteger(model.default_steps, 20),
      min: 1,
      max: 100,
      step: 1,
      recommended: [positiveInteger(model.default_steps, 20)],
      mode: "adjustable",
    },
    guidance: {
      default: Number.isFinite(model.default_guidance)
        ? Number(model.default_guidance)
        : 3.5,
      min: 0,
      max: 100,
      step: 0.1,
      mode: "adjustable",
    },
    temporal,
    capabilities: {
      guidance: {
        adjustable: true,
        supports_negative_prompt: true,
      },
      supports_lora: false,
      supports_controlnet: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      schedulers: [],
    },
  };
}

export function groupDimensions(
  dimensions: readonly { width: number; height: number }[],
): ProfileAspectGroup[] {
  const groups: ProfileAspectGroup[] = [];
  for (const { width, height } of dimensions) {
    const divisor = gcd(width, height);
    const id = `${Math.round(width) / divisor}:${Math.round(height) / divisor}`;
    let group = groups.find((candidate) => candidate.id === id);
    if (!group) {
      group = { id, label: id, presets: [] };
      groups.push(group);
    }
    group.presets.push({
      id: `${width}x${height}`,
      width,
      height,
      tier: "recommended",
    });
  }
  for (const group of groups) {
    group.presets.sort(
      (left, right) => left.width * left.height - right.width * right.height,
    );
  }
  return groups;
}

export function profilePresets(
  model: GenerationProfileModel | null | undefined,
  pipeline?: string | null,
): ProfileResolutionPreset[] {
  return (
    effectiveGenerationRecipe(
      model,
      pipeline,
    )?.resolution.aspect_groups.flatMap((group) => group.presets) ?? []
  );
}

export interface ProfileAspectOption {
  id: string;
  label: string;
  ratio: number;
}

export function profileAspectOptions(
  model: GenerationProfileModel | null | undefined,
  pipeline?: string | null,
): ProfileAspectOption[] {
  return (
    effectiveGenerationRecipe(model, pipeline)
      ?.resolution.aspect_groups.map((group) => {
        const first = group.presets[0];
        return first
          ? {
              id: group.id,
              label: group.label,
              ratio: first.width / first.height,
            }
          : null;
      })
      .filter((option): option is ProfileAspectOption => option !== null) ?? []
  );
}

export function profileAspectIdForResolution(
  model: GenerationProfileModel | null | undefined,
  pipeline: string | null | undefined,
  width: number,
  height: number,
  tolerance = 0.02,
): string | null {
  const recipe = effectiveGenerationRecipe(model, pipeline);
  if (!recipe || !(width > 0) || !(height > 0)) return null;
  for (const group of recipe.resolution.aspect_groups) {
    if (
      group.presets.some(
        (preset) => preset.width === width && preset.height === height,
      )
    ) {
      return group.id;
    }
  }
  if (recipe.resolution.domain !== "dynamic") return null;
  const ratio = width / height;
  const nearest = profileAspectOptions(model, pipeline)
    .map((option) => ({
      id: option.id,
      diff: Math.abs(option.ratio - ratio) / option.ratio,
    }))
    .sort((left, right) => left.diff - right.diff)[0];
  return nearest && nearest.diff <= tolerance ? nearest.id : null;
}

export function integerControlError(
  label: string,
  value: number,
  control: IntegerControl | null | undefined,
): string | null {
  if (!control) return null;
  if (control.mode === "fixed" && value !== control.default) {
    return `${label} is fixed at ${control.default} for this recipe.`;
  }
  if (
    !Number.isInteger(value) ||
    value < control.min ||
    value > control.max ||
    (value - control.min) % control.step !== 0
  ) {
    return `${label} must be a whole number from ${control.min} to ${control.max} in increments of ${control.step}.`;
  }
  return null;
}

export function floatControlError(
  label: string,
  value: number,
  control: FloatControl | null | undefined,
): string | null {
  if (!control) return null;
  if (control.mode === "fixed" && value !== control.default) {
    return `${label} is fixed at ${control.default} for this recipe.`;
  }
  return Number.isFinite(value) && value >= control.min && value <= control.max
    ? null
    : `${label} must be from ${control.min} to ${control.max}.`;
}

export function resolutionProfileError(
  width: number,
  height: number,
  resolution: ResolutionProfile | null | undefined,
): string | null {
  if (!resolution || resolution.domain === "none") return null;
  if (!Number.isInteger(width) || !Number.isInteger(height)) {
    return "Width and height must be whole numbers.";
  }
  if (width < resolution.min_width || height < resolution.min_height) {
    return `Width and height must be at least ${resolution.min_width} × ${resolution.min_height} pixels.`;
  }
  if (
    width % resolution.alignment !== 0 ||
    height % resolution.alignment !== 0
  ) {
    return `Width and height must be multiples of ${resolution.alignment}.`;
  }
  if (
    resolution.max_axis_pixels &&
    Math.max(width, height) > resolution.max_axis_pixels
  ) {
    return `${width} × ${height} exceeds the ${resolution.max_axis_pixels}px span this recipe can hold.`;
  }
  if (width * height > resolution.max_pixels) {
    return `${width} × ${height} exceeds this recipe's ${(resolution.max_pixels / 1_000_000).toFixed(1)} MP limit.`;
  }
  const ratio = width / height;
  if (resolution.min_aspect_ratio && ratio < resolution.min_aspect_ratio) {
    return `The ${ratio.toFixed(2)} aspect ratio is narrower than this recipe supports.`;
  }
  if (resolution.max_aspect_ratio && ratio > resolution.max_aspect_ratio) {
    return `The ${ratio.toFixed(2)} aspect ratio is wider than this recipe supports.`;
  }
  if (resolution.domain === "buckets") {
    const exact = resolution.aspect_groups.some((group) =>
      group.presets.some(
        (preset) => preset.width === width && preset.height === height,
      ),
    );
    if (!exact)
      return "Choose one of this recipe's supported resolution buckets.";
  }
  return null;
}

function positiveInteger(
  value: number | null | undefined,
  fallback: number,
): number {
  return Number.isFinite(value) && Number(value) > 0
    ? Math.max(1, Math.round(Number(value)))
    : fallback;
}

function gcd(a: number, b: number): number {
  let left = Math.max(1, Math.round(Math.abs(a)));
  let right = Math.max(1, Math.round(Math.abs(b)));
  while (right !== 0) [left, right] = [right, left % right];
  return left;
}
