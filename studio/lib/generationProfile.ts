/** Versioned generation-control contract emitted by `/api/models`. */
import type {
  AdapterControlProfile,
  FeatureControlProfile,
  GenerationRecipeProfile as WireGenerationRecipeProfile,
  GenerationProfileSet,
  IntegerControl,
  OutputCapabilitiesProfile,
  FloatControl,
  ProfileAspectGroup,
  ProfileFpsControl,
  ResolutionProfile,
  TemporalProfile,
  WanRecipeCapabilitiesProfile,
} from "./generated/generationProfileV1";

export type {
  AdapterControlProfile,
  ControlMode,
  FeatureControlProfile,
  FloatControl,
  GenerationProfileSet,
  IntegerControl,
  OutputCapabilitiesProfile,
  ProfileAspectGroup,
  ProfileFpsControl,
  ProfileResolutionPreset,
  ResolutionDomain,
  ResolutionProfile,
  TemporalProfile,
  WanRecipeCapabilitiesProfile,
} from "./generated/generationProfileV1";

/** Wire recipe plus the contained Release-N legacy-adapter marker. */
export type GenerationRecipeProfile = WireGenerationRecipeProfile & {
  legacy_adapter?: true;
};

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
  return isGenerationProfileSetV1(profile) ? profile : null;
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
  return advertisedGenerationRecipe(model, pipeline);
}

/** Resolve only a server-authored v1 recipe, never the Release-N legacy adapter. */
export function advertisedGenerationRecipe(
  model: GenerationProfileModel | null | undefined,
  pipeline?: string | null,
): GenerationRecipeProfile | null {
  const profile = advertisedGenerationProfile(model);
  if (!profile) return null;
  if (pipeline) {
    const exact = profile.recipes.find(
      (recipe) => recipe.request_selector.pipeline === pipeline,
    );
    return exact ?? null;
  }
  return (
    profile.recipes.find((recipe) => recipe.id === profile.default_recipe_id) ??
    null
  );
}

/** Name an explicit recipe selector the advertised v1 contract cannot resolve. */
export function generationRecipeSelectionError(
  model: GenerationProfileModel | null | undefined,
  pipeline?: string | null,
): string | null {
  const profile = advertisedGenerationProfile(model);
  if (!profile || !pipeline) return null;
  return profile.recipes.some(
    (recipe) => recipe.request_selector.pipeline === pipeline,
  )
    ? null
    : `Pipeline “${pipeline}” is not supported by this model.`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function positiveIntegerValue(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) > 0;
}

function nullableFinite(value: unknown): boolean {
  return value === undefined || value === null || finiteNumber(value);
}

function isIntegerControl(value: unknown): value is IntegerControl {
  if (!isRecord(value)) return false;
  const { default: initial, min, max, step, mode, recommended } = value;
  return (
    Number.isInteger(initial) &&
    Number.isInteger(min) &&
    Number.isInteger(max) &&
    positiveIntegerValue(step) &&
    Number(min) <= Number(initial) &&
    Number(initial) <= Number(max) &&
    (mode === "adjustable" || mode === "fixed" || mode === "hidden") &&
    (recommended === undefined ||
      (Array.isArray(recommended) &&
        recommended.every((item) => Number.isInteger(item))))
  );
}

function isFloatControl(value: unknown): value is FloatControl {
  if (!isRecord(value)) return false;
  const { default: initial, min, max, step, mode } = value;
  return (
    finiteNumber(initial) &&
    finiteNumber(min) &&
    finiteNumber(max) &&
    finiteNumber(step) &&
    step > 0 &&
    min <= initial &&
    initial <= max &&
    (mode === "adjustable" || mode === "fixed" || mode === "hidden")
  );
}

function isResolutionProfile(value: unknown): value is ResolutionProfile {
  if (!isRecord(value) || !Array.isArray(value.aspect_groups)) return false;
  const noResolution = value.domain === "none";
  if (
    !["dynamic", "buckets", "source-driven", "none"].includes(
      String(value.domain),
    ) ||
    !positiveIntegerValue(value.alignment) ||
    (noResolution
      ? value.min_width !== 0 ||
        value.min_height !== 0 ||
        value.max_pixels !== 0 ||
        value.aspect_groups.length !== 0
      : !positiveIntegerValue(value.min_width) ||
        !positiveIntegerValue(value.min_height) ||
        !positiveIntegerValue(value.max_pixels)) ||
    !nullableFinite(value.max_axis_pixels) ||
    !nullableFinite(value.source_max_pixels) ||
    !nullableFinite(value.min_aspect_ratio) ||
    !nullableFinite(value.max_aspect_ratio)
  ) {
    return false;
  }
  return value.aspect_groups.every((group) => {
    if (
      !isRecord(group) ||
      typeof group.id !== "string" ||
      !group.id ||
      typeof group.label !== "string"
    ) {
      return false;
    }
    if (!Array.isArray(group.presets)) return false;
    return group.presets.every(
      (preset) =>
        isRecord(preset) &&
        typeof preset.id === "string" &&
        Boolean(preset.id) &&
        positiveIntegerValue(preset.width) &&
        positiveIntegerValue(preset.height) &&
        typeof preset.tier === "string",
    );
  });
}

function isTemporalProfile(value: unknown): value is TemporalProfile {
  if (
    !isRecord(value) ||
    !isIntegerControl(value.frames) ||
    !Number.isInteger(value.frame_offset)
  ) {
    return false;
  }
  const fps = value.fps;
  const validFps =
    isRecord(fps) &&
    ((fps.mode === "fixed" && finiteNumber(fps.value)) ||
      (fps.mode === "adjustable" &&
        finiteNumber(fps.default) &&
        finiteNumber(fps.min) &&
        finiteNumber(fps.max) &&
        finiteNumber(fps.step) &&
        fps.step > 0 &&
        fps.min <= fps.default &&
        fps.default <= fps.max));
  return validFps && nullableFinite(value.max_duration_seconds);
}

function isRecipe(value: unknown): value is GenerationRecipeProfile {
  if (
    !isRecord(value) ||
    !isRecord(value.request_selector) ||
    !isRecord(value.defaults)
  ) {
    return false;
  }
  const selector = value.request_selector.pipeline;
  const defaults = value.defaults;
  const caps = value.capabilities;
  const structurallyValid =
    typeof value.id === "string" &&
    Boolean(value.id) &&
    typeof value.label === "string" &&
    (selector === undefined ||
      selector === null ||
      (typeof selector === "string" && selector.length > 0)) &&
    (isRecord(value.resolution) && value.resolution.domain === "none"
      ? defaults.width === 0 && defaults.height === 0
      : positiveIntegerValue(defaults.width) &&
        positiveIntegerValue(defaults.height)) &&
    positiveIntegerValue(defaults.steps) &&
    finiteNumber(defaults.guidance) &&
    nullableFinite(defaults.frames) &&
    nullableFinite(defaults.fps) &&
    (defaults.negative_prompt === undefined ||
      defaults.negative_prompt === null ||
      typeof defaults.negative_prompt === "string") &&
    isResolutionProfile(value.resolution) &&
    isIntegerControl(value.steps) &&
    isFloatControl(value.guidance) &&
    (value.temporal === undefined ||
      value.temporal === null ||
      isTemporalProfile(value.temporal)) &&
    isRecord(caps) &&
    isRecord(caps.guidance) &&
    typeof caps.guidance.adjustable === "boolean" &&
    typeof caps.guidance.supports_negative_prompt === "boolean" &&
    isFeatureControl(caps.negative_prompt) &&
    nullableFinite(caps.guidance.fixed_scale) &&
    (caps.source_image === undefined ||
      caps.source_image === null ||
      ["unsupported", "optional", "required"].includes(
        String(caps.source_image),
      )) &&
    typeof caps.supports_lora === "boolean" &&
    typeof caps.supports_controlnet === "boolean" &&
    typeof caps.supports_sequence === "boolean" &&
    typeof caps.supports_extend === "boolean" &&
    typeof caps.supports_audio === "boolean" &&
    isFeatureControl(caps.source_video) &&
    isFeatureControl(caps.mask) &&
    isFeatureControl(caps.keyframes) &&
    isFeatureControl(caps.audio) &&
    isAdapterControl(caps.lora) &&
    isAdapterControl(caps.controlnet) &&
    isOutputCapabilities(caps.output) &&
    isWanRecipeCapabilities(caps.wan_recipe) &&
    (caps.schedulers === undefined ||
      (Array.isArray(caps.schedulers) &&
        caps.schedulers.every((scheduler) => typeof scheduler === "string")));
  if (!structurallyValid) return false;

  const recipe = value as unknown as GenerationRecipeProfile;
  return (
    resolutionProfileError(
      recipe.defaults.width,
      recipe.defaults.height,
      recipe.resolution,
    ) === null &&
    integerControlError("Steps", recipe.defaults.steps, recipe.steps) ===
      null &&
    floatControlError("Guidance", recipe.defaults.guidance, recipe.guidance) ===
      null &&
    recipe.resolution.aspect_groups.every((group) =>
      group.presets.every(
        (preset) =>
          resolutionProfileError(
            preset.width,
            preset.height,
            recipe.resolution,
          ) === null,
      ),
    ) &&
    (recipe.temporal === null ||
      recipe.temporal === undefined ||
      ((recipe.defaults.frames === null ||
        recipe.defaults.frames === undefined ||
        integerControlError(
          "Frames",
          recipe.defaults.frames,
          recipe.temporal.frames,
        ) === null) &&
        (recipe.defaults.fps === null ||
          recipe.defaults.fps === undefined ||
          fpsControlAccepts(recipe.temporal.fps, recipe.defaults.fps))))
  );
}

function isFeatureControl(value: unknown): value is FeatureControlProfile {
  return (
    isRecord(value) &&
    ["adjustable", "fixed", "hidden"].includes(String(value.mode)) &&
    typeof value.required === "boolean" &&
    (value.reason === undefined ||
      value.reason === null ||
      typeof value.reason === "string")
  );
}

function isAdapterControl(value: unknown): value is AdapterControlProfile {
  return (
    isRecord(value) &&
    ["adjustable", "fixed", "hidden"].includes(String(value.mode)) &&
    Number.isInteger(value.max_count) &&
    Number(value.max_count) >= 0 &&
    (value.reason === undefined ||
      value.reason === null ||
      typeof value.reason === "string")
  );
}

const OUTPUT_FORMATS = ["png", "jpeg", "gif", "apng", "webp", "mp4", "wav"];

function isOutputCapabilities(
  value: unknown,
): value is OutputCapabilitiesProfile {
  return (
    isRecord(value) &&
    OUTPUT_FORMATS.includes(String(value.default_format)) &&
    Array.isArray(value.formats) &&
    value.formats.length > 0 &&
    value.formats.every((format) => OUTPUT_FORMATS.includes(String(format))) &&
    value.formats.includes(value.default_format) &&
    typeof value.audio_requires_mp4 === "boolean" &&
    (value.delivery_reason === undefined ||
      value.delivery_reason === null ||
      typeof value.delivery_reason === "string")
  );
}

function isWanRecipeCapabilities(
  value: unknown,
): value is WanRecipeCapabilitiesProfile {
  return (
    isRecord(value) &&
    ["adjustable", "fixed", "hidden"].includes(String(value.mode)) &&
    typeof value.supports_distill_strength === "boolean" &&
    typeof value.supports_first_last_frame === "boolean" &&
    (value.first_last_frame_min_frames === undefined ||
      value.first_last_frame_min_frames === null ||
      positiveIntegerValue(value.first_last_frame_min_frames)) &&
    (value.reason === undefined ||
      value.reason === null ||
      typeof value.reason === "string")
  );
}

function fpsControlAccepts(control: ProfileFpsControl, value: number): boolean {
  if (control.mode === "fixed") return value === control.value;
  return (
    value >= control.min &&
    value <= control.max &&
    Number.isInteger((value - control.min) / control.step)
  );
}

function isGenerationProfileSetV1(
  value: unknown,
): value is GenerationProfileSet {
  if (
    !isRecord(value) ||
    !Array.isArray(value.recipes) ||
    value.recipes.length === 0
  )
    return false;
  if (
    value.schema_version !== 1 ||
    typeof value.profile_id !== "string" ||
    !value.profile_id ||
    typeof value.profile_hash !== "string" ||
    !value.profile_hash ||
    typeof value.default_recipe_id !== "string" ||
    !value.default_recipe_id ||
    !value.recipes.every(isRecipe)
  ) {
    return false;
  }
  const ids = value.recipes.map((recipe) => recipe.id);
  const pipelines = value.recipes
    .map((recipe) => recipe.request_selector.pipeline)
    .filter((pipeline) => typeof pipeline === "string");
  return (
    new Set(ids).size === ids.length &&
    new Set(pipelines).size === pipelines.length &&
    ids.includes(value.default_recipe_id)
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
    legacy_adapter: true,
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
      negative_prompt: {
        mode: "adjustable",
        required: false,
      },
      supports_lora: false,
      supports_controlnet: false,
      supports_sequence: false,
      supports_extend: false,
      supports_audio: false,
      source_video: { mode: "hidden", required: false },
      mask: { mode: "adjustable", required: false },
      keyframes: { mode: "hidden", required: false },
      audio: { mode: "hidden", required: false },
      lora: { mode: "hidden", max_count: 0 },
      controlnet: { mode: "hidden", max_count: 0 },
      output: {
        default_format:
          family === "ltx2" || family === "ltx-video" || family === "wan"
            ? "mp4"
            : "png",
        formats:
          family === "ltx2" || family === "ltx-video" || family === "wan"
            ? ["mp4", "gif", "apng", "webp"]
            : ["png", "jpeg", "webp"],
        audio_requires_mp4: family === "ltx2",
      },
      wan_recipe: {
        mode: "hidden",
        supports_distill_strength: false,
        supports_first_last_frame: false,
      },
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

/**
 * The values a recipe's FIXED controls demand. A form value that disagrees
 * with a fixed control is never user authority — the control renders
 * disabled — so callers snap the hidden value instead of stranding Generate
 * behind an error the user cannot correct. Desktop's
 * `reconcileModelCapabilities` and web's model-defaults/submit paths share
 * this so the two surfaces cannot drift.
 *
 * Resolution is fixed authority under exactly one shape: a bucket domain
 * that refuses off-bucket sizes and advertises a single preset, which is
 * H3's reviewed compact envelope. An absent `off_bucket` reads as Reject —
 * the same fail-closed reading admission takes. Wan advertises buckets that
 * only warn, and Wuerstchen advertises one preset on a dynamic canvas; a
 * preset is a recommendation in both cases and neither snaps.
 */
export function fixedRecipeControlOverrides(
  recipe: GenerationRecipeProfile | null | undefined,
): {
  steps?: number;
  guidance?: number;
  width?: number;
  height?: number;
  frames?: number;
} {
  const overrides: ReturnType<typeof fixedRecipeControlOverrides> = {};
  if (recipe?.steps.mode === "fixed") overrides.steps = recipe.steps.default;
  if (recipe?.guidance.mode === "fixed") {
    overrides.guidance = recipe.guidance.default;
  }
  const resolution = recipe?.resolution;
  if (
    resolution?.domain === "buckets" &&
    (resolution.off_bucket ?? "reject") !== "warn"
  ) {
    const [only, ...rest] = resolution.aspect_groups.flatMap(
      (group) => group.presets,
    );
    if (only && rest.length === 0) {
      overrides.width = only.width;
      overrides.height = only.height;
    }
  }
  if (recipe?.temporal?.frames.mode === "fixed") {
    overrides.frames = recipe.temporal.frames.default;
  }
  return overrides;
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
  if (resolution.domain === "buckets" && resolution.off_bucket !== "warn") {
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

export interface ResolutionFinding {
  level: "block" | "warn";
  message: string;
}

/**
 * The client-side resolution verdict: the server is the authority on whether
 * a size renders, so every profile constraint (minimums, alignment, span,
 * pixel budget, aspect range, bucket membership) is an ADVISORY here — the
 * request still submits and a real refusal comes back as the server's error.
 * Only input that cannot form a coherent request at all (non-integer / NaN /
 * < 1) blocks. `resolutionProfileError` keeps its strict semantics for the
 * Rust-parity contract and the recipe-shape guard; render THIS on surfaces.
 */
export function resolutionProfileFinding(
  width: number,
  height: number,
  resolution: ResolutionProfile | null | undefined,
): ResolutionFinding | null {
  if (!resolution || resolution.domain === "none") return null;
  if (
    !Number.isInteger(width) ||
    !Number.isInteger(height) ||
    width < 1 ||
    height < 1
  ) {
    return {
      level: "block",
      message: "Width and height must be whole numbers.",
    };
  }
  if (width < resolution.min_width || height < resolution.min_height) {
    return {
      level: "warn",
      message: `This model expects at least ${resolution.min_width} × ${resolution.min_height} — the server may reject or adjust this size.`,
    };
  }
  if (
    width % resolution.alignment !== 0 ||
    height % resolution.alignment !== 0
  ) {
    return {
      level: "warn",
      message: `This model expects multiples of ${resolution.alignment} — the server may reject this size.`,
    };
  }
  if (
    resolution.max_axis_pixels &&
    Math.max(width, height) > resolution.max_axis_pixels
  ) {
    return {
      level: "warn",
      message: `${width} × ${height} exceeds the ${resolution.max_axis_pixels}px span this recipe was trained for — the server may reject it.`,
    };
  }
  if (width * height > resolution.max_pixels) {
    return {
      level: "warn",
      message: `${width} × ${height} exceeds this recipe's ${(resolution.max_pixels / 1_000_000).toFixed(1)} MP budget — the server may reject it.`,
    };
  }
  const ratio = width / height;
  if (resolution.min_aspect_ratio && ratio < resolution.min_aspect_ratio) {
    return {
      level: "warn",
      message: `The ${ratio.toFixed(2)} aspect ratio is narrower than this recipe was trained for — the server may reject it.`,
    };
  }
  if (resolution.max_aspect_ratio && ratio > resolution.max_aspect_ratio) {
    return {
      level: "warn",
      message: `The ${ratio.toFixed(2)} aspect ratio is wider than this recipe was trained for — the server may reject it.`,
    };
  }
  const exact = resolution.aspect_groups.some((group) =>
    group.presets.some(
      (preset) => preset.width === width && preset.height === height,
    ),
  );
  if (resolution.domain === "buckets" && !exact) {
    return resolution.off_bucket === "warn"
      ? {
          level: "warn",
          message: `This model isn't optimized for ${width}×${height} — results may vary.`,
        }
      : {
          level: "warn",
          message: `${width} × ${height} isn't one of this model's trained sizes — the server may reject it.`,
        };
  }
  return null;
}

export interface ClosestProfileAspect {
  id: string;
  label: string;
  ratio: number;
  /** False when the size matches no preset — the chip is an approximation. */
  exact: boolean;
}

/**
 * The aspect group a custom size should highlight. Unlike
 * {@link profileAspectIdForResolution} (exact presets, nearest-ratio only in
 * a dynamic domain), this always answers for any domain with no tolerance
 * cutoff, flagging approximations so surfaces can mark the chip "≈" — the
 * Advanced exact size drives the shape, never the other way around.
 */
export function closestProfileAspect(
  model: GenerationProfileModel | null | undefined,
  pipeline: string | null | undefined,
  width: number,
  height: number,
): ClosestProfileAspect | null {
  const recipe = effectiveGenerationRecipe(model, pipeline);
  if (!recipe || !(width > 0) || !(height > 0)) return null;
  const options = profileAspectOptions(model, pipeline);
  if (options.length === 0) return null;
  const exactGroup = recipe.resolution.aspect_groups.find((group) =>
    group.presets.some(
      (preset) => preset.width === width && preset.height === height,
    ),
  );
  if (exactGroup) {
    const option = options.find((candidate) => candidate.id === exactGroup.id);
    if (option) return { ...option, exact: true };
  }
  const ratio = width / height;
  const nearest = [...options].sort(
    (left, right) =>
      Math.abs(left.ratio - ratio) / left.ratio -
      Math.abs(right.ratio - ratio) / right.ratio,
  )[0];
  return nearest ? { ...nearest, exact: false } : null;
}

/**
 * Advisory counterpart to {@link resolutionProfileError} for `warn`-policy
 * bucket recipes: the size is admitted (the buckets are the trained sizes,
 * not the only runnable ones), but the user should know quality may vary.
 * Never rendered as a blocker.
 */
export function resolutionProfileWarning(
  width: number,
  height: number,
  resolution: ResolutionProfile | null | undefined,
): string | null {
  if (
    !resolution ||
    resolution.domain !== "buckets" ||
    resolution.off_bucket !== "warn"
  ) {
    return null;
  }
  if (resolutionProfileError(width, height, resolution)) return null;
  const exact = resolution.aspect_groups.some((group) =>
    group.presets.some(
      (preset) => preset.width === width && preset.height === height,
    ),
  );
  if (exact) return null;
  return `This model isn't optimized for ${width}×${height} — results may vary.`;
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
