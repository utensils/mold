import {
  isMinimaxH3Identity,
  minimaxH3TaskForModel,
} from "./minimaxH3Authoring";
import {
  resolveSourceImageCapability,
  supportsFirstLastFrames,
  type SourceImageCapability,
} from "./sourceImageCapability";
import type { GenerationRecipeProfile } from "./generationProfile";

export { isMinimaxH3Family } from "./minimaxH3Authoring";

/**
 * Solver selection, spelled exactly as `mold-core`'s kebab-case `Scheduler`
 * serializes plus a `"default"` sentinel meaning "omit the field".
 *
 * Two disjoint sets share the slot: `ddim` / `euler-ancestral` / `uni-pc` are
 * the UNet schedulers, and `uni-pc` / `euler` / `dpm-pp` are wan's flow sample
 * solvers. `schedulerOptions` narrows it per family — validation rejects a
 * wan solver off-family and a UNet scheduler on wan.
 */
export type GenerationScheduler =
  | "default"
  | "ddim"
  | "euler-ancestral"
  | "uni-pc"
  | "edm-dpm-pp-2m"
  | "euler"
  | "dpm-pp";

export type SourceImageMode =
  | "single"
  | "qwen-edit"
  | "references"
  | "h3-boundaries"
  | "ordered-references";

/** Whether the resolved model takes the wan sampler-recipe controls. */
export interface WanRecipeCapabilities {
  /** Flow shift and the sample-solver picker apply to this model. */
  supported: boolean;
  /** The tier ships a Lightning distill, so the per-expert strengths apply. */
  supportsDistillStrength: boolean;
}

export interface BaseGenerationCapabilities {
  supportsNegativePrompt: boolean;
  guidanceAdjustable: boolean;
  fixedGuidance: number | null;
  supportsScheduler: boolean;
  schedulerOptions: GenerationScheduler[];
  /** Wan flow shift / solver / distill strengths (`wanRecipe.ts`). */
  wanRecipe: WanRecipeCapabilities;
  supportsCfgPlus: boolean;
  supportsVideo: boolean;
  supportsAudio: boolean;
  supportsLora: boolean;
  maxLoraStack: number;
  supportsControlNet: boolean;
  outputFormats: string[];
  defaultOutputFormat: string;
  outputDeliveryReason: string | null;
  supportsSourceVideo: boolean;
  requiresSourceVideo: boolean;
  supportsKeyframes: boolean;
  requiresKeyframes: boolean;
  supportsAudioInput: boolean;
  requiresAudioInput: boolean;
  sourceImageMode: SourceImageMode;
  /**
   * The effective per-model source-image contract (#772): the model's own
   * advertised mode when it has one, this family's heuristic otherwise.
   */
  sourceImageCapability: SourceImageCapability;
  /** Whether the source-image well renders at all. */
  supportsSourceImage: boolean;
  /** Whether Generate stays gated until a source image is attached. */
  requiresSourceImage: boolean;
  /** Whether the optional wan End frame well renders (#779). */
  supportsEndFrame: boolean;
  supportsMask: boolean;
  /**
   * Whether the engine reads `strength` for its source image. Wan pins the
   * first frame exactly — its conditioning has no denoise-strength knob — so
   * showing the slider (or serializing the field) would advertise a control
   * the render ignores.
   */
  supportsStrength: boolean;
  forcesBatchSizeOne: boolean;
}

export interface AdvertisedGuidanceCapabilities {
  adjustable: boolean;
  supports_negative_prompt: boolean;
  fixed_scale?: number | null;
}

/**
 * Resolve the guidance value that may cross a request boundary.
 *
 * Forms deliberately retain the user's adjustable value so switching back to
 * a guided recipe restores it. A fixed distilled recipe still owns the wire
 * value; sending the hidden form value makes the server reject a control the
 * user cannot edit.
 */
export function effectiveGenerationGuidance(
  capabilities: Pick<BaseGenerationCapabilities, "fixedGuidance">,
  formGuidance: number,
): number {
  return capabilities.fixedGuidance ?? formGuidance;
}

const SCHEDULER_OPTIONS: GenerationScheduler[] = [
  "default",
  "ddim",
  "euler-ancestral",
  "uni-pc",
];

/** Wan's `--sample_solver` choices. `uni-pc` is also the server's own default,
 * which `"default"` selects by omitting the field. */
const WAN_SOLVER_OPTIONS: GenerationScheduler[] = [
  "default",
  "uni-pc",
  "euler",
  "dpm-pp",
];

/** Display names for every solver, so the three surfaces agree on wording. */
const SCHEDULER_LABELS: Record<GenerationScheduler, string> = {
  default: "Default",
  ddim: "DDIM",
  "euler-ancestral": "Euler ancestral",
  "uni-pc": "UniPC",
  "edm-dpm-pp-2m": "EDM DPM++ 2M",
  euler: "Euler",
  "dpm-pp": "DPM++",
};

export function schedulerLabel(scheduler: string): string {
  return SCHEDULER_LABELS[scheduler as GenerationScheduler] ?? scheduler;
}

const NO_NEGATIVE_PROMPT_FAMILIES = new Set([
  "flux",
  "flux2",
  "flux.2",
  "flux-2",
  "z-image",
  "qwen-image",
  "qwen_image",
  "qwen-image-edit",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
]);

const SCHEDULER_FAMILIES = new Set([
  "sd15",
  "sd1.5",
  "stable-diffusion-1.5",
  "sdxl",
]);

const CFG_PLUS_FAMILIES = new Set(["sd3", "sd3.5"]);
const VIDEO_FAMILIES = new Set([
  "ltx-video",
  "ltx2",
  "ltx-2",
  "wan",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
]);
/**
 * Generated audio is LTX-2's alone. Wan is a video family with no audio
 * branch at all — its checkpoints ship no audio VAE and no vocoder — so it
 * belongs in `VIDEO_FAMILIES` and emphatically not here. Listing it would
 * surface an audio toggle whose request the server rejects before denoising.
 */
const AUDIO_FAMILIES = new Set([
  "ltx2",
  "ltx-2",
  "minimax-h3",
  "minimax_h3",
  "minimaxh3",
]);
/**
 * The families with a bespoke advanced-controls panel (LTX-2's guidance
 * overrides, STG, spatial/temporal rungs). Wan needs only the generic video
 * controls — steps, guidance, negative — so it stays out.
 */
const ADVANCED_VIDEO_FAMILIES = new Set(["ltx2", "ltx-2"]);
/**
 * Video families whose engine accepts a still image as conditioning.
 *
 * This is deliberately its own set rather than a synonym for
 * `ADVANCED_VIDEO_FAMILIES`, even though the two held the same members until
 * now. Desktop derived the source-image well from the advanced-video flag
 * (`!supportsVideo || supportsAdvancedVideo`), which was indistinguishable
 * from correct while LTX-2 was the only image-conditioned video family. Wan is
 * the first that is image-conditioned *without* the advanced panel, and under
 * that derivation its well vanished — including for `wan22-i2v-a14b`, which
 * cannot generate at all without a source image.
 *
 * Plain `ltx-video` stays out: its engine has no image-to-video path and would
 * silently ignore an attached image.
 */
const IMAGE_CONDITIONED_VIDEO_FAMILIES = new Set(["ltx2", "ltx-2", "wan"]);
const CONTROLNET_FAMILIES = new Set(["sd15", "sd1.5", "stable-diffusion-1.5"]);

/**
 * Wan A14B tiers that ship a Lightning distill adapter, and therefore accept
 * the per-expert `distill_strength_*` controls.
 *
 * `/api/models` advertises no component list, so the tier is read off the
 * resolved model identity the same way `isFlux2DevModel` reads a checkpoint's.
 * This mirrors `A14bTier::has_distill` in `mold-core`'s manifest — only the
 * A14B `:q5` (fast) and `:q4` (compact) tiers pull the distill files; the `:q8`
 * quality tier and the 1.3B/5B checkpoints have no distill slot at all.
 *
 * It fails closed: an opaque `cv:` / `hf:` wan checkpoint gets no strength
 * controls, which is what the engine's own "ships no distill adapter"
 * rejection would say anyway.
 */
const WAN_DISTILL_TIER = /a14b[:-]q[45]$/;

/**
 * Wan A14B tiers whose fp8-scaled loader refuses every LoRA stack (#777):
 * an fp8 merge would dequantize each targeted weight, add the delta, and
 * re-round it to three mantissa bits. The bf16 and GGUF tiers keep adapters.
 */
const WAN_FP8_TIER = /a14b[:-]fp8$/;

export const MAX_LORA_STACK = 4;

/**
 * Families whose engines can merge an adapter. Mirrors the `workflows.lora`
 * flag each family declares in `crates/mold-inference/src/batch.rs`; a family
 * missing here has its LoRA control hidden on every surface even though the
 * server would accept the request.
 *
 * Wan reaches its adapters two different ways depending on the checkpoint's
 * container — merged into the weights for bf16/fp8 safetensors, applied as a
 * parallel low-rank branch for GGUF — but that is invisible from here: the
 * request shape is identical either way.
 */
export const LORA_CAPABLE_FAMILIES = [
  "flux",
  "flux2",
  "ltx2",
  "sd15",
  "sd3",
  "sdxl",
  "qwen-image",
  "qwen-image-edit",
  "wan",
  "z-image",
] as const;

/**
 * `advertisedSourceImage` is the selected model row's additive
 * `source_image` field (#772). Pass it wherever the row is in scope; omitting
 * it falls back to the family heuristic below, which is what an older server
 * — one that enforces nothing at admission — expects.
 */
export function baseGenerationCapabilities(
  family: string,
  model = "",
  pipeline?: string | null,
  advertisedGuidance?: AdvertisedGuidanceCapabilities | null,
  advertisedSourceImage?: string | null,
  advertisedRecipe?: GenerationRecipeProfile | null,
): BaseGenerationCapabilities {
  const normalized = family.trim().toLowerCase();
  const qwenEdit = isQwenImageEditFamily(normalized);
  const flux2Dev = isFlux2DevModel(model);
  const h3 = isMinimaxH3Identity(normalized, model);
  const h3Ref2va = h3 && isMinimaxH3Ref2vaModel(model);
  const wan = isWanFamily(normalized);
  const profileCaps = advertisedRecipe?.legacy_adapter
    ? undefined
    : advertisedRecipe?.capabilities;
  const advertisedSchedulers = profileCaps?.schedulers;
  const schedulerOptions = advertisedSchedulers
    ? advertisedSchedulers.length > 0
      ? (["default", ...advertisedSchedulers] as GenerationScheduler[])
      : []
    : wan
      ? WAN_SOLVER_OPTIONS.slice()
      : SCHEDULER_FAMILIES.has(normalized)
        ? SCHEDULER_OPTIONS.slice()
        : [];
  const normalizedModel = model.trim().toLowerCase();
  const ltx =
    normalized === "ltx-video" ||
    normalized === "ltx2" ||
    normalized === "ltx-2";
  const fixedLtxPipeline = [
    "distilled",
    "ic-lora",
    "retake",
    "lip-dub",
  ].includes(pipeline ?? "");
  const inferredFixedGuidance =
    ltx &&
    (fixedLtxPipeline || (!pipeline && normalizedModel.includes("distilled")));
  const supportsVideo = h3 || VIDEO_FAMILIES.has(normalized);
  // The family heuristic behind the advertised contract: every image family
  // reads a source image, and among the video families only the
  // image-conditioned ones do. It can say a checkpoint *reads* a source
  // image, never that it cannot render without one — requiredness comes from
  // the server alone.
  const familySourceImage: SourceImageCapability =
    !supportsVideo || isImageConditionedVideoFamily(normalized) || h3
      ? "optional"
      : "unsupported";
  const sourceImageCapability = resolveSourceImageCapability(
    advertisedSourceImage,
    familySourceImage,
  );
  const advertisedDefault = !pipeline ? advertisedGuidance : null;
  const fixedGuidance = h3
    ? true
    : advertisedDefault
      ? !advertisedDefault.adjustable
      : inferredFixedGuidance;
  const profileControlVisible = (mode: string | undefined) =>
    mode === "adjustable" || mode === "fixed";
  const legacyOutputFormats = supportsVideo
    ? ["mp4", "gif", "apng", "webp"]
    : ["png", "jpeg", "webp"];
  return {
    supportsNegativePrompt:
      (profileCaps
        ? profileControlVisible(profileCaps.negative_prompt.mode)
        : advertisedDefault?.supports_negative_prompt) ??
      (!NO_NEGATIVE_PROMPT_FAMILIES.has(normalized) && !fixedGuidance),
    guidanceAdjustable: !fixedGuidance,
    fixedGuidance: fixedGuidance
      ? h3
        ? 0
        : (advertisedDefault?.fixed_scale ?? 1)
      : null,
    supportsScheduler: schedulerOptions.length > 0,
    schedulerOptions,
    wanRecipe: {
      supported: profileCaps
        ? profileControlVisible(profileCaps.wan_recipe.mode)
        : wan,
      supportsDistillStrength:
        profileCaps?.wan_recipe.supports_distill_strength ??
        (wan && WAN_DISTILL_TIER.test(normalizedModel)),
    },
    supportsCfgPlus: CFG_PLUS_FAMILIES.has(normalized),
    supportsVideo,
    supportsAudio:
      profileCaps?.supports_audio ?? (h3 || AUDIO_FAMILIES.has(normalized)),
    supportsLora: profileCaps?.lora
      ? profileControlVisible(profileCaps.lora.mode)
      : !flux2Dev &&
        // The wan fp8-scaled tier deliberately refuses every adapter stack —
        // merging into e4m3 would re-round the delta to three mantissa bits —
        // so offering the control would advertise a load that always fails.
        // Mirrors `WanTransformer::from_safetensors_with_loras`.
        !(wan && WAN_FP8_TIER.test(normalizedModel)) &&
        (LORA_CAPABLE_FAMILIES as readonly string[]).includes(normalized),
    maxLoraStack: profileCaps?.lora.max_count ?? MAX_LORA_STACK,
    supportsControlNet: profileCaps
      ? profileControlVisible(profileCaps.controlnet.mode)
      : CONTROLNET_FAMILIES.has(normalized),
    outputFormats: profileCaps?.output.formats.slice() ?? legacyOutputFormats,
    defaultOutputFormat:
      profileCaps?.output.default_format ?? legacyOutputFormats[0]!,
    outputDeliveryReason: profileCaps?.output.delivery_reason ?? null,
    supportsSourceVideo: profileCaps
      ? profileControlVisible(profileCaps.source_video.mode)
      : ltx,
    requiresSourceVideo: profileCaps?.source_video.required ?? false,
    supportsKeyframes: profileCaps
      ? profileControlVisible(profileCaps.keyframes.mode)
      : ltx || wan,
    requiresKeyframes: profileCaps?.keyframes.required ?? false,
    supportsAudioInput: profileCaps
      ? profileControlVisible(profileCaps.audio.mode)
      : ltx,
    requiresAudioInput: profileCaps?.audio.required ?? false,
    sourceImageMode: h3Ref2va
      ? "ordered-references"
      : h3
        ? "h3-boundaries"
        : flux2Dev
          ? "references"
          : qwenEdit
            ? "qwen-edit"
            : "single",
    sourceImageCapability,
    supportsSourceImage: sourceImageCapability !== "unsupported",
    requiresSourceImage: sourceImageCapability === "required",
    supportsEndFrame:
      profileCaps?.wan_recipe.supports_first_last_frame ??
      supportsFirstLastFrames(normalized, advertisedSourceImage),
    // Wan pins conditioning frames exactly: no repaint mask, no denoise
    // strength — the engine rejects the former and never reads the latter.
    supportsMask: profileCaps
      ? profileControlVisible(profileCaps.mask.mode)
      : !h3 && !qwenEdit && !flux2Dev && !wan,
    supportsStrength: !h3 && !qwenEdit && !flux2Dev && !wan,
    forcesBatchSizeOne: h3 || qwenEdit,
  };
}

export function isAdvancedVideoFamily(family: string): boolean {
  return ADVANCED_VIDEO_FAMILIES.has(family.trim().toLowerCase());
}

/**
 * Whether a video family's engine reads a still source image.
 *
 * Surfaces gate the source-image control on this, never on
 * `isAdvancedVideoFamily` — the two answer different questions and only
 * coincided while LTX-2 was the sole image-conditioned video family.
 */
export function isImageConditionedVideoFamily(family: string): boolean {
  return IMAGE_CONDITIONED_VIDEO_FAMILIES.has(family.trim().toLowerCase());
}

export function isWanFamily(family: string): boolean {
  return family.trim().toLowerCase() === "wan";
}

export function isQwenImageEditFamily(family: string): boolean {
  return family === "qwen-image-edit";
}

export function isFlux2DevModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  return normalized.includes("flux2-dev") || normalized.includes("flux.2-dev");
}

export function isMinimaxH3Ref2vaModel(model: string): boolean {
  return minimaxH3TaskForModel(model) === "ref2va";
}
