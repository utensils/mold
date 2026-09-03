/**
 * Reactive form state for the Generate workspace and the pure builder that
 * turns it into a wire `GenerateRequest`. The capability matrix
 * (`capabilities.ts`) decides which fields survive; this module only holds the
 * editable shape and the model-default / request-assembly plumbing.
 */
import type {
  GenerateRequest,
  KeyframeConditionWire,
  LoraWeight,
  Ltx2PipelineMode,
  Ltx2SpatialUpscale,
  Ltx2TemporalUpscale,
  ModelEntry,
  OutputFormat,
  OutputMetadata,
  Scheduler,
  TimeRange,
} from "./api/types";
import {
  MAX_LORA_STACK,
  coerceFormOutputFormat,
  defaultOutputFormat,
  generationCapabilitiesForFamily,
  outputFormatsForFamily,
  pruneRequestForFamily,
  recipeCapabilitiesSnapshot,
  type RecipeCapabilitiesSnapshot,
} from "./capabilities";
import {
  emptyMeshForm,
  meshFormFromMetadata,
  meshRequestFromForm,
  type MeshFormState,
} from "@studio/lib/meshControls";
import {
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  parseSourceFitPolicy,
  type SourceFitPolicy,
} from "@studio/lib/sourceFit";
import {
  addTag,
  buildFileUnderRequestFields,
  deriveGhostTag,
  emptyFileUnderState,
  pickCollection,
  requestTagKey,
  type FileUnderCollectionLike,
  type FileUnderState,
} from "@studio/lib/fileUnder";
import { defaultVideoFps } from "@studio/lib/sequence";
import { videoFramesForModelSelection } from "@studio/lib/videoDuration";
import { pipelineForSettingsReuse } from "@studio/lib/outputReuse";
import { familySupportsExtend, resolveExtendOverlapFrames } from "@studio/lib/extend";
import { findInstalledModel } from "./generateModels";
import {
  cameraMotionFromLoraPath,
  cameraMotionLoraLabel,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import {
  emptyGuidanceOverrides,
  guidanceOverridesFromWire,
  guidanceOverridesToWire,
  type Ltx2GuidanceOverridesState,
} from "@studio/lib/guidanceOverrides";
import {
  emptyWanRecipe,
  wanRecipeFromWire,
  wanRecipeToWire,
  type WanRecipeState,
} from "@studio/lib/wanRecipe";
import {
  effectiveNegativeDefault,
  negativePromptOnDefaultChange,
  negativePromptWireValue,
  restoredNegativeExplicitClear,
  restoredNegativePrompt,
} from "@studio/lib/negativePrompt";
import { pipelineForControlId } from "@studio/lib/ltx2Control";
import {
  identityRequestFields,
  identityReuse,
  supportsIdentity,
} from "@studio/lib/identityConditioning";
import { validatePrintTitle } from "@studio/lib/libraryOrganization";
import { firstLastFrameKeyframes } from "@studio/lib/sourceImageCapability";
import { effectiveGenerationGuidance, isWanFamily } from "@studio/lib/generationCapabilities";
import { conditioningForRequest, type ExclusiveWell } from "@studio/lib/sourceMediaPlan";
import { isMeshFamily } from "@studio/lib/legacyRecipeRules";
import { isAudioOnlyPipeline, stripAudioOnlyIncompatibleFields } from "@studio/lib/ltx2Pipeline";
import { requestVideoOnly } from "@studio/lib/videoOnly";
import {
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
} from "@studio/lib/generationProfile";
import {
  MINIMAX_H3_REVIEWED_COMPACT_FRAMES,
  cloneMinimaxH3AuthoringState,
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Family,
  minimaxH3BoundaryFromSourceMetadata,
  minimaxH3BoundaryFromStagedImage,
  minimaxH3ClosingBoundaryFromMetadata,
  minimaxH3ReferenceDraftsFromMetadata,
  minimaxH3TaskForModel,
  serializeMinimaxH3Authoring,
  stagedImageFromMinimaxH3Boundary,
  type MinimaxH3AuthoringState,
} from "@studio/lib/minimaxH3Authoring";

/** A LoRA row in the stack: wire fields plus display metadata (name, triggers). */
export interface FormLora {
  path: string;
  name: string;
  scale: number;
  trainedWords: string[];
  /** Machine whose `/api/loras` returned this host-local path. UI-only: the
   * request builder strips it before the LoRA reaches the wire. */
  hostId?: string;
  /** Endpoint that returned the path, used when instance telemetry was not
   * available yet. */
  hostBaseUrl?: string;
  /** Server installation that owned the path. Null means identity telemetry
   * was unavailable when picked; a later identity may be accepted only at the
   * same host id and endpoint. */
  hostInstanceId?: string | null;
}

export type LoraHostBinding =
  | { kind: "unbound" }
  | { kind: "bound"; hostId: string; baseUrl: string | null; instanceId: string | null }
  | { kind: "conflict"; hostIds: string[] };

/** A LoRA path is host-local, so a picked stack freezes generation to the
 * machine that supplied it. Legacy/restored rows have no binding and retain
 * the selected generation policy. */
export function loraHostBinding(loras: readonly FormLora[]): LoraHostBinding {
  const hostIds = new Set<string>();
  const baseUrls = new Set<string>();
  const instanceIds = new Set<string>();
  let unboundCount = 0;
  for (const lora of loras) {
    if (!lora.hostId) {
      unboundCount += 1;
      continue;
    }
    hostIds.add(lora.hostId);
    if (lora.hostBaseUrl) baseUrls.add(lora.hostBaseUrl);
    if (lora.hostInstanceId) instanceIds.add(lora.hostInstanceId);
  }
  if (hostIds.size === 0) return { kind: "unbound" };
  if (unboundCount > 0 || hostIds.size > 1 || baseUrls.size > 1 || instanceIds.size > 1)
    return { kind: "conflict", hostIds: [...hostIds] };
  return {
    kind: "bound",
    hostId: hostIds.values().next().value!,
    baseUrl: baseUrls.values().next().value ?? null,
    instanceId: instanceIds.values().next().value ?? null,
  };
}

/** Whether a concrete route can safely consume a host-local LoRA path. A
 * missing picked instance may promote to later telemetry only when the host
 * id and exact endpoint still match. */
export function loraBindingMatchesRoute(
  binding: Extract<LoraHostBinding, { kind: "bound" }>,
  route: { hostId: string; instanceId?: string | null; target: { baseUrl: string } },
): boolean {
  return (
    route.hostId === binding.hostId &&
    (binding.baseUrl === null || route.target.baseUrl === binding.baseUrl) &&
    (binding.instanceId === null || (route.instanceId ?? null) === binding.instanceId)
  );
}

/** A picked file (base64, no data-URI prefix); `filename` is display metadata. */
export interface PickedFile {
  filename: string;
  base64: string;
}

/** An image picked via {@link ImagePickerModal} (upload or gallery). */
export type PickedImage = PickedFile;

/** One LTX-2 keyframe: a conditioning image pinned to a frame index. */
export interface FormKeyframe {
  frame: number;
  image: PickedImage;
}

/** Whether the current seed field means "roll fresh" or "locked". */
export function seedMode(seed: string): "random" | "fixed" {
  return seed.trim() === "" ? "random" : "fixed";
}

export interface GenerateForm {
  prompt: string;
  /** Original prompt before an expand; sent as `original_prompt` and used for undo. */
  originalPrompt: string | null;
  /** User-authored print title (Create header). Ships as `title` on every
   * request built from this form — one-shot, Batch N siblings, prepared
   * variations — after `validatePrintTitle`; `""` = untitled. Generating
   * never clears it (a named session wants its siblings and re-rolls to share
   * the name); only the explicit ⌘N "new print" (`clearComposer`) does. */
  title: string;
  /** "File under" — the Create-time Library filing draft (ghost-tag opt-out,
   * typed tags, collection pick). Reducers live in `@studio/lib/fileUnder`;
   * `buildRequest` materializes it into the additive `tags` / `collection`
   * wire fields, so every one-shot, Batch N sibling, and prepared variation
   * built from this form files identically — exactly like `title`. */
  fileUnder: FileUnderState;
  /** Settings ▸ Library "Tag new prints with their title", mirrored onto the
   * form so `buildRequest` stays a pure function of it. Not a form value: a
   * wholesale reset preserves it and the owning surface re-syncs it from its
   * own preference store.
   *
   * Defaults to FALSE, which is not the product default (that is on) — it is
   * the safe default for a surface that has not wired the group up. A ghost
   * tag files the print on the user's behalf and must be visible before
   * Generate, so a shell with no File under UI has to opt in rather than
   * inherit the behaviour invisibly. Desktop opts in at boot
   * (`libraryPrefs.init()`). */
  fileUnderAutoTag: boolean;
  /** The fleet collection whose slug equals the current title's slug, as last
   * resolved from the merged Library listings. Held here (rather than the
   * whole listing) so the request builder can offer the auto-match without
   * knowing about stores; the inspector keeps it in sync. */
  fileUnderMatch: FileUnderCollectionLike | null;
  model: string;
  family: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  guidanceCapabilities: ModelEntry["guidance_capabilities"];
  /** The selected model's advertised source-image contract (#772), snapshotted
   * on model change like `guidanceCapabilities`. Read it through
   * `generationCapabilitiesForFamily`, never raw; `null` means the host
   * advertised nothing and the family heuristic answers. */
  sourceImageCapability: ModelEntry["source_image"];
  /** Empty string = random seed. */
  seed: string;
  negativePrompt: string;
  /** The selected model's advertised default negative
   * (`default_negative_prompt`, wan today; "" when none). Prefill and wire
   * semantics derive from `@studio/lib/negativePrompt`: text equal to this
   * stays absent on the wire, a cleared field ships the explicit `""`
   * opt-out. */
  negativePromptDefault: string;
  /** Restore-time explicit-clear authority (#787 round 3): true when a reuse
   * carried the explicit `""` opt-out while the advertised default was still
   * unknown (rows not loaded). Keeps the clear from decaying to "untouched"
   * once the row resolves, and keeps the wire shipping `""` meanwhile.
   * Resets on model selection and on any resolved default change. */
  negativeExplicitClear: boolean;
  scheduler: Scheduler;
  cfgPlus: boolean;
  batchSize: number;
  outputFormat: OutputFormat;
  /** Post-generate upscaler model name; empty = off. */
  upscaleModel: string;
  strength: number;
  /** base64, no data-URI prefix. */
  sourceImage: string | null;
  /** Provenance label for `sourceImage` — the gallery filename or upload
   * name it came from. Ships as `source_image_name` so Reuse settings can
   * restore the input image later; always cleared with the image. */
  sourceImageName: string | null;
  /** Decoded dimensions of the effective primary conditioning image. For
   * single-source families this describes `sourceImage`; for Qwen edit it
   * describes attachment 0 (the Target). These are UI sizing metadata only
   * and never travel on the generation wire. */
  sourceImageWidth: number | null;
  sourceImageHeight: number | null;
  /** Wan first/last-frame conditioning (#779): the closing still. Optional,
   * offered only on a wan checkpoint whose advertised contract accepts a
   * source image, and meaningless without `sourceImage` — the pair ships as
   * the two-entry `keyframes` layout, never a lone keyframe. */
  endFrame: PickedImage | null;
  /** Face-identity (PuLID) reference photo. Primary-form media beside the
   * source wells, and deliberately NOT source conditioning: it is never
   * fitted against the canvas and ships verbatim. A bytes-less entry is the
   * reattach descriptor Reuse settings leaves behind when the local stash no
   * longer holds the photo the print recorded. */
  identityImage: PickedImage | null;
  /** Identity strength; null = untouched, so the server default applies. */
  identityWeight: number | null;
  /** First identity-conditioned step; null = untouched. */
  identityStartStep: number | null;
  /** Whether the selected checkpoint accepts an identity photo, snapshotted
   * from its resolved recipe / `/api/models` row exactly like
   * `sourceImageCapability` — `buildRequest` takes only the form, and the
   * capability is what decides whether the partition may ride the wire. Null
   * means nothing has been read yet, which reads as "no". */
  identitySupported: boolean | null;
  /** Ordered edit/reference strip, base64 each (no data-URI prefix). For Qwen,
   * index 0 is the edit Target and the rest are References. FLUX.2 Dev treats
   * every entry as an ordered Reference. Empty in single-source mode. */
  imageAttachments: string[];
  /**
   * Which exclusive well was written last, on a `single-or-references` recipe
   * (FLUX.2 [klein]) whose source image and references are mutually exclusive.
   * `resolveExclusiveWells` reads it to decide which well is active and which
   * parks; `null` — an untouched form, or a snapshot from before the field —
   * reads as the source well.
   */
  exclusiveWell: ExclusiveWell | null;
  /** How a source image that doesn't match width×height maps onto the canvas.
   * Applied client-side on submit (`sourceFitPreprocess.ts`), never wired. */
  sourceFit: SourceFitPolicy;
  maskImage: string | null;
  controlImage: string | null;
  controlModel: string;
  controlScale: number;
  loras: FormLora[];
  // Video families (ltx-video / ltx2).
  frames: number;
  /** Omit frames so a qualified LTX-2.5 duration head chooses the clip length. */
  predictDuration: boolean;
  /** Selected host/model positively advertised duration-head readiness. */
  durationPredictionSupported: boolean;
  fps: number;
  enableAudio: boolean;
  /** LTX-2 video-only opt-in (#1037); output-changing, never a default. */
  videoOnly: boolean;
  // LTX-2 advanced video (ltx2 only). All optional-safe: null / [] defaults so
  // a partial stored form (template snapshot) still hydrates cleanly.
  sourceVideo: PickedImage | null;
  /** Existing video to continue; set makes the request a continuation. */
  extendVideo: PickedImage | null;
  /** Pixel-frame overlap; null takes the host's advertised default. */
  extendOverlapFrames: number | null;
  /** The selected model's advertised `extend_default_overlap_frames`, snapshot
   * from its `/api/models` row the same way `sourceImageCapability` is. The
   * request builder takes only the form, and a continuation must submit the
   * overlap the inspector is showing rather than leave the field absent — see
   * `resolveExtendOverlapFrames`. Null = the host advertised none. */
  extendDefaultOverlapFrames: number | null;
  keyframes: FormKeyframe[];
  pipeline: Ltx2PipelineMode | null;
  /** Official host-provided IC-LoRA control adapter ID. */
  icLoraControl?: string | null;
  retakeRange: TimeRange | null;
  spatialUpscale: Ltx2SpatialUpscale | null;
  temporalUpscale: Ltx2TemporalUpscale | null;
  /** Optional LTX-2 guider overrides. Empty values preserve pipeline defaults. */
  guidanceOverrides: Ltx2GuidanceOverridesState;
  /** Wan flow shift and per-expert distill strengths. Null until touched for
   * the same reason as `guidanceOverrides`: the field must stay off the wire
   * so the resolved tier keeps its own value. */
  wanRecipe: WanRecipeState;
  /** 3-D mesh controls; `null` entries take the recipe's advertised default.
   * Rendered only while `recipeCapabilities.mesh` is present, and never on
   * the wire otherwise. */
  mesh: MeshFormState;
  /** The resolved recipe's request-shaping facts (formats, prompt mode,
   * strength, canvasless, mesh controls), snapshotted when a model or
   * pipeline is applied so the request builder needs only the form. `null`
   * on a host that advertises no recipe. */
  recipeCapabilities: RecipeCapabilitiesSnapshot | null;
  /** Conditioning audio for the a2-vid pipeline; base64 on the wire. */
  audioFile: PickedFile | null;
  /** LTX-2 camera-motion LoRA: a preset id (dolly-in, …, static) or an
   * explicit `.safetensors` path; null = off. Ships as a `loras[]` entry
   * (`camera-control:<preset>` or the raw path) at scale 1.0 — exactly what
   * the CLI's `--camera-control` sends; there is no dedicated wire field. */
  cameraControl: string | null;
  /** Composer style preset id (see `stylePresets.ts`); `""` = none. A
   * look-and-feel modifier composed into the outgoing prompt at submit — never
   * mutates the textarea and carries no dedicated wire field. */
  stylePreset: string;
  /** Dedicated H3 contract; never projected through edit_images. */
  h3Authoring?: MinimaxH3AuthoringState;
}

export function newGenerateForm(): GenerateForm {
  return {
    prompt: "",
    originalPrompt: null,
    title: "",
    fileUnder: emptyFileUnderState(),
    fileUnderAutoTag: false,
    fileUnderMatch: null,
    model: "",
    family: "",
    width: 1024,
    height: 1024,
    steps: 4,
    guidance: 3.5,
    guidanceCapabilities: null,
    sourceImageCapability: null,
    seed: "",
    negativePrompt: "",
    negativePromptDefault: "",
    negativeExplicitClear: false,
    scheduler: "default",
    cfgPlus: false,
    batchSize: 1,
    outputFormat: "png",
    upscaleModel: "",
    strength: 0.75,
    sourceImage: null,
    sourceImageName: null,
    sourceImageWidth: null,
    sourceImageHeight: null,
    endFrame: null,
    identityImage: null,
    identityWeight: null,
    identityStartStep: null,
    identitySupported: null,
    imageAttachments: [],
    exclusiveWell: null,
    sourceFit: defaultSourceFitPolicy(),
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1.0,
    loras: [],
    frames: 97,
    predictDuration: false,
    durationPredictionSupported: false,
    fps: 24,
    enableAudio: false,
    videoOnly: false,
    sourceVideo: null,
    extendVideo: null,
    extendOverlapFrames: null,
    extendDefaultOverlapFrames: null,
    keyframes: [],
    pipeline: null,
    icLoraControl: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    guidanceOverrides: emptyGuidanceOverrides(),
    wanRecipe: emptyWanRecipe(),
    mesh: emptyMeshForm(),
    recipeCapabilities: null,
    audioFile: null,
    cameraControl: null,
    stylePreset: "",
    h3Authoring: emptyMinimaxH3AuthoringState(),
  };
}

/**
 * Take a submission-safe snapshot of the mutable composer state. Source-fit
 * preprocessing is asynchronous and may run for minutes; callers must never
 * read the live reactive form again after a user taps Generate.
 */
export function cloneGenerateForm(form: GenerateForm): GenerateForm {
  const sourceFit: SourceFitPolicy =
    form.sourceFit.mode === "upscale-then-fit"
      ? { ...form.sourceFit, fit: { ...form.sourceFit.fit } }
      : { ...form.sourceFit };
  return {
    ...form,
    imageAttachments: [...form.imageAttachments],
    fileUnder: {
      ...form.fileUnder,
      manualTags: [...form.fileUnder.manualTags],
      picked: form.fileUnder.picked ? { ...form.fileUnder.picked } : null,
    },
    fileUnderMatch: form.fileUnderMatch ? { ...form.fileUnderMatch } : null,
    sourceFit,
    loras: form.loras.map((lora) => ({
      ...lora,
      trainedWords: [...lora.trainedWords],
    })),
    endFrame: form.endFrame ? { ...form.endFrame } : null,
    identityImage: form.identityImage ? { ...form.identityImage } : null,
    sourceVideo: form.sourceVideo ? { ...form.sourceVideo } : null,
    extendVideo: form.extendVideo ? { ...form.extendVideo } : null,
    keyframes: form.keyframes.map((keyframe) => ({
      ...keyframe,
      image: { ...keyframe.image },
    })),
    retakeRange: form.retakeRange ? { ...form.retakeRange } : null,
    guidanceOverrides: { ...form.guidanceOverrides },
    wanRecipe: { ...form.wanRecipe },
    mesh: { ...(form.mesh ?? emptyMeshForm()) },
    recipeCapabilities: form.recipeCapabilities
      ? {
          ...form.recipeCapabilities,
          outputFormats: [...form.recipeCapabilities.outputFormats],
        }
      : null,
    audioFile: form.audioFile ? { ...form.audioFile } : null,
    h3Authoring: cloneMinimaxH3AuthoringState(form.h3Authoring),
  };
}

/**
 * Apply a model's defaults and prune anything the new family can't use. LoRAs
 * clear on every model change — even FLUX→FLUX — because an adapter may not
 * target the new variant's tensor layout.
 */
export function applyModelDefaults(form: GenerateForm, m: ModelEntry): void {
  const cameraRows = form.loras.filter(
    (lora) => lora.path.startsWith("camera-control:") || lora.path === form.cameraControl?.trim(),
  );
  form.width = m.default_width;
  form.height = m.default_height;
  form.steps = m.default_steps;
  form.guidance = m.default_guidance;
  form.guidanceCapabilities = m.guidance_capabilities ?? null;
  form.predictDuration = false;
  if (isMinimaxH3Family(m.family)) {
    form.frames = m.default_frames ?? MINIMAX_H3_REVIEWED_COMPACT_FRAMES;
  } else if (m.default_frames != null) {
    form.frames = videoFramesForModelSelection(form.frames, m);
  }
  // The model's advertised rate is applied like steps/guidance — it is only
  // absent-server/absent-field that leaves the current value in place.
  form.fps = defaultVideoFps(m, form.fps);
  form.loras = [];
  form.icLoraControl = null;
  // Selecting a model is fresh authority: a deferred restore-time clear
  // marker (#787 round 3) belongs to the restored model, not this pick, so
  // the advertised default prefills like any other model switch. The
  // row-refresh path (`reconcileModelCapabilities` alone) keeps the marker.
  form.negativeExplicitClear = false;
  reconcileModelCapabilities(form, m);
  if (form.cameraControl) {
    form.loras = syncCameraMotionLora(
      cameraRows,
      form.cameraControl,
      form.cameraControl,
      (path, scale) => ({
        path,
        name: cameraMotionLoraLabel(path),
        scale,
        trainedWords: [],
      }),
    );
  }
}

/** Apply one fully resolved recipe as a fresh model-owned settings boundary.
 * Prompt, seed, batch and compatible conditioning remain user-owned. An
 * unknown explicit selector returns false and leaves the prior controls in
 * place so client validation can fail closed without destroying authored work.
 */
export function applyRecipeDefaults(
  form: GenerateForm,
  m: ModelEntry | null | undefined,
  pipeline: string | null,
): boolean {
  form.pipeline = pipeline as Ltx2PipelineMode | null;
  const recipe = effectiveGenerationRecipe(m, pipeline);
  if (!recipe) return false;

  // The pipeline chooses the recipe, and identity support is a recipe
  // capability — re-resolve it here or a pipeline switch would keep the
  // previous recipe's answer.
  form.identitySupported = supportsIdentity(recipe, m);
  form.recipeCapabilities = recipeCapabilitiesSnapshot(
    recipe,
    m?.family ?? form.family,
    m?.name ?? form.model,
    pipeline,
    m?.source_image,
  );
  if (!form.recipeCapabilities?.mesh) form.mesh = emptyMeshForm();
  form.width = recipe.defaults.width;
  form.height = recipe.defaults.height;
  form.steps = recipe.defaults.steps;
  form.guidance = recipe.defaults.guidance;
  if (recipe.defaults.frames != null) form.frames = recipe.defaults.frames;
  if (recipe.defaults.fps != null) form.fps = recipe.defaults.fps;
  form.outputFormat =
    coerceFormOutputFormat(form.outputFormat, form.family, form.recipeCapabilities) ??
    form.outputFormat;

  const negativeDefault = recipe.defaults.negative_prompt ?? "";
  form.negativePrompt = negativeDefault;
  form.negativePromptDefault = negativeDefault;
  form.negativeExplicitClear = false;
  form.guidanceCapabilities = {
    adjustable: recipe.capabilities.guidance.adjustable,
    supports_negative_prompt: recipe.capabilities.guidance.supports_negative_prompt,
    fixed_scale: recipe.capabilities.guidance.fixed_scale ?? null,
  };

  form.scheduler = "default";
  form.cfgPlus = false;
  form.spatialUpscale = null;
  form.temporalUpscale = null;
  form.guidanceOverrides = emptyGuidanceOverrides();
  form.retakeRange = null;
  form.icLoraControl = null;
  if (recipe.capabilities.source_image === "unsupported") {
    form.sourceImage = null;
    form.sourceImageName = null;
    form.sourceImageWidth = null;
    form.sourceImageHeight = null;
    form.maskImage = null;
    form.imageAttachments = [];
  }
  if (!recipe.capabilities.supports_extend) {
    form.extendVideo = null;
    form.extendOverlapFrames = null;
  }
  if (!recipe.capabilities.supports_audio) {
    form.enableAudio = false;
    form.audioFile = null;
    form.videoOnly = false;
  }
  return true;
}

/**
 * Refresh family/capability metadata for the same named model on a different
 * host without discarding portable user parameters. Host manifests are the
 * authority; two remotes may advertise corrected or aliased family metadata.
 */
export function reconcileModelCapabilities(form: GenerateForm, m: ModelEntry): void {
  // The outgoing model's source layout, read before the row overwrites
  // model/family. A same-model re-reconcile (host refresh, template load)
  // sees no transition and the bridge below stays a no-op.
  const prevMode = generationCapabilitiesForFamily(
    form.family,
    form.model,
    null,
    null,
    form.sourceImageCapability,
  ).sourceImageMode;
  form.model = m.name;
  form.family = m.family;
  form.sourceImageCapability = m.source_image ?? null;
  form.durationPredictionSupported =
    m.supports_duration_prediction === true && m.runtime_ready !== false;
  if (!form.durationPredictionSupported) form.predictDuration = false;
  form.extendDefaultOverlapFrames = m.extend_default_overlap_frames ?? null;
  // #787: a Negative field still showing the previous model's advertised
  // default follows the new model (that is also how the default first
  // appears); typed text and an explicit clear are user authority. The
  // family constant backs an older server that omits the additive field so
  // a known default (and the "" opt-out against it) never decays to absence.
  const nextNegativeDefault = effectiveNegativeDefault(m, m.family);
  form.negativePrompt = negativePromptOnDefaultChange(
    form.negativePrompt,
    form.negativePromptDefault,
    nextNegativeDefault,
    // A reuse restored before this row landed may carry the explicit ""
    // opt-out — indistinguishable from untouched without the marker, which
    // keeps the clear instead of prefilling (#787 round 3).
    form.negativeExplicitClear,
  );
  form.negativePromptDefault = nextNegativeDefault;
  if (nextNegativeDefault !== "") {
    // The default is known now; the ordinary tri-state rules carry the
    // clear from here, so the deferred marker has served its purpose.
    form.negativeExplicitClear = false;
  }
  const recipe = effectiveGenerationRecipe(m, form.pipeline);
  // Identity is a property of the checkpoint, snapshotted here for the same
  // reason as `sourceImageCapability`: the request builder takes only the
  // form. The staged photo deliberately survives a switch that loses the
  // capability — `buildRequest` keeps it off the wire, and the inline reason
  // beside the well (plus the blocked submit) is what tells the user.
  form.identitySupported = supportsIdentity(recipe, m);
  // A row refresh can resolve a persisted/template form against a newer
  // authoritative recipe. Fixed controls are not user choices: normalize the
  // hidden form value to the same value the disabled control displays, or the
  // validator can strand Generate behind an error the user cannot correct.
  // Shared with web so the surfaces cannot drift.
  Object.assign(form, fixedRecipeControlOverrides(recipe));
  form.recipeCapabilities = recipeCapabilitiesSnapshot(
    recipe,
    m.family,
    m.name,
    form.pipeline,
    m.source_image,
  );
  // Pre-mesh snapshots restored via Object.assign may lack the slot.
  form.mesh ??= emptyMeshForm();
  if (!form.recipeCapabilities?.mesh) form.mesh = emptyMeshForm();
  if (form.recipeCapabilities?.canvasless) {
    // A canvasless recipe advertises a zero canvas; the model row's
    // `default_width`/`default_height` describe nothing the request reads.
    form.width = recipe?.defaults.width ?? 0;
    form.height = recipe?.defaults.height ?? 0;
  }
  const caps = generationCapabilitiesForFamily(
    m.family,
    m.name,
    null,
    null,
    m.source_image,
    recipe,
  );
  if (!outputFormatsForFamily(m.family, recipe).includes(form.outputFormat)) {
    form.outputFormat = defaultOutputFormat(m.family, recipe);
  }
  if (!caps.supportsScheduler || !caps.schedulerOptions.includes(form.scheduler)) {
    // The UNet schedulers and wan's solvers share one field but are disjoint
    // on the server, so a value carried across that boundary would be
    // rejected rather than ignored.
    form.scheduler = "default";
  }
  if (!caps.supportsCfgPlus) form.cfgPlus = false;
  if (!caps.wanRecipe.supported) {
    form.wanRecipe = emptyWanRecipe();
  } else if (!caps.wanRecipe.supportsDistillStrength) {
    form.wanRecipe = {
      ...form.wanRecipe,
      distillStrengthHigh: null,
      distillStrengthLow: null,
    };
  }
  if (caps.forcesBatchSizeOne) form.batchSize = 1;
  // ── Source-media bridge + retention ─────────────────────────────────────
  // Staged media survives capability-losing switches: `buildRequest`'s
  // capability gates plus `pruneRequestForFamily` keep it off the wire, and
  // switching back restores the picture instead of losing authored work.
  // Layout *moves* between the three source authorities (single source ↔
  // qwen-edit strip ↔ H3 boundaries) so the visible well keeps the image.
  const enteringH3 = caps.sourceImageMode === "h3-boundaries" && prevMode !== "h3-boundaries";
  const leavingH3 = prevMode === "h3-boundaries" && caps.sourceImageMode !== "h3-boundaries";
  // Pre-#-era snapshots restored via Object.assign may lack the slot.
  form.h3Authoring ??= emptyMinimaxH3AuthoringState();
  if (enteringH3) {
    if (!form.h3Authoring.firstFrame) {
      const boundary = minimaxH3BoundaryFromStagedImage(
        form.sourceImage
          ? {
              base64: form.sourceImage,
              filename: form.sourceImageName,
              width: form.sourceImageWidth,
              height: form.sourceImageHeight,
            }
          : form.imageAttachments[0]
            ? { base64: form.imageAttachments[0] }
            : null,
      );
      if (boundary) {
        form.h3Authoring.firstFrame = boundary;
        form.sourceImage = null;
        form.sourceImageName = null;
        form.sourceImageWidth = null;
        form.sourceImageHeight = null;
        form.imageAttachments = [];
      }
    }
    // A first-frame-only checkpoint (`required`) rejects a lastFrame outright
    // (minimaxH3AuthoringError), so the closing frame stays parked instead.
    if (!form.h3Authoring.lastFrame && form.endFrame && !caps.requiresSourceImage) {
      const closing = minimaxH3BoundaryFromStagedImage({
        base64: form.endFrame.base64,
        filename: form.endFrame.filename,
      });
      if (closing) {
        form.h3Authoring.lastFrame = closing;
        form.endFrame = null;
      }
    }
  } else if (leavingH3) {
    // Promote authored boundaries back into the target layout. Bytes-less
    // reattach descriptors stay parked in h3Authoring for a later H3 return.
    const promoted = stagedImageFromMinimaxH3Boundary(form.h3Authoring.firstFrame);
    if (promoted) {
      if (caps.sourceImageMode === "single") {
        if (!form.sourceImage) {
          form.sourceImage = promoted.base64;
          form.sourceImageName = promoted.filename;
          form.sourceImageWidth = promoted.width ?? null;
          form.sourceImageHeight = promoted.height ?? null;
          form.h3Authoring.firstFrame = null;
        }
      } else if (form.imageAttachments.length === 0) {
        form.imageAttachments = [promoted.base64];
        form.h3Authoring.firstFrame = null;
      }
    }
    const closing = stagedImageFromMinimaxH3Boundary(form.h3Authoring.lastFrame);
    if (closing && caps.supportsEndFrame && !form.endFrame) {
      form.endFrame = { filename: closing.filename, base64: closing.base64 };
      form.h3Authoring.lastFrame = null;
    }
  }
  if (caps.sourceImageMode === "h3-boundaries") {
    // Boundaries are the only source authority here; nothing else moves.
  } else if (caps.sourceImageMode === "single-or-references") {
    // Klein takes BOTH wells, so neither layout moves: a source image stays a
    // source image and a strip stays a strip. Whichever holds media is the
    // active one and the other parks — the request builder picks exactly one.
  } else if (caps.sourceImageMode !== "single") {
    // Entering qwen-edit/references: a single-mode source seeds the strip as
    // the Target (web parity — the attachment survives the model switch).
    if (form.imageAttachments.length === 0 && form.sourceImage) {
      form.imageAttachments = [form.sourceImage];
    }
    form.sourceImage = null;
    // The picture strip carries no per-image labels.
    form.sourceImageName = null;
  } else if (form.imageAttachments.length > 0) {
    // Leaving qwen-edit: the Target becomes the single img2img source (web
    // parity — attachments truncate to one, which single mode reads).
    if (!form.sourceImage) {
      form.sourceImage = form.imageAttachments[0] ?? null;
      form.sourceImageName = null;
    }
    form.imageAttachments = [];
  }
  if (!caps.supportsMask) {
    // Maskless img2img (LTX-2 image-to-video) can't repaint pad bands, so a
    // mask-dependent fit policy flips to crop-fill on entry. The mask itself
    // is staged media and survives; `pruneRequestForFamily` gates the wire.
    if (caps.supportsImg2img && caps.sourceImageMode === "single") {
      form.sourceFit = coerceSourceFitForMaskless(form.sourceFit);
    }
  }
  if (!caps.supportsAudio || m.supports_audio === false) {
    form.enableAudio = false;
    form.videoOnly = false;
  }
  // Continuation is per family, not part of the LTX-2 advanced-video suite
  // (#783) — wan continues and is deliberately not an advanced-video family,
  // so clearing it with that suite would drop a staged clip on every row
  // refresh for the SAME model and quietly send a plain text-to-video job.
  if (!familySupportsExtend(m.family)) {
    form.extendVideo = null;
    form.extendOverlapFrames = null;
  }
  if (!caps.supportsAdvancedVideo) {
    // Settings knobs clear with the suite; staged media (sourceVideo,
    // keyframes, audioFile) is retained — the wire prune gates it and a
    // return to LTX-2 finds it where the user left it.
    form.pipeline = null;
    form.icLoraControl = null;
    form.retakeRange = null;
    form.spatialUpscale = null;
    form.temporalUpscale = null;
    form.guidanceOverrides = emptyGuidanceOverrides();
    form.loras = syncCameraMotionLora(form.loras, form.cameraControl, null, (path, scale) => ({
      path,
      name: path,
      scale,
      trainedWords: [],
    }));
    form.cameraControl = null;
  }
}

/**
 * Normalize a form snapshot saved before `negativePromptDefault` existed —
 * legacy templates and drafts (#787 round 2) — before it is `Object.assign`ed
 * into a live form. Such snapshots carry no tri-state authority: their empty
 * `negativePrompt` means "untouched", never the explicit `""` opt-out.
 * Assigning one raw keeps the live form's previous-model default next to the
 * template's `""` and manufactures an opt-out the user never made. The
 * snapshot's own model/family resolve the default (live inventory row first,
 * family constant fallback); a snapshot that already carries the key is
 * post-#787 authority and passes through untouched.
 */
export function normalizeLegacyNegativeSnapshot(
  snapshot: GenerateForm,
  models: ModelEntry[] = [],
): GenerateForm {
  if (typeof (snapshot as Partial<GenerateForm>).negativePromptDefault === "string") {
    // Post-round-2, pre-round-3 snapshots lack the marker; leaving it
    // undefined would let Object.assign keep the live form's stale value.
    snapshot.negativeExplicitClear ??= false;
    return snapshot;
  }
  const model = findInstalledModel(models, snapshot.model);
  const nextDefault = effectiveNegativeDefault(model, snapshot.family);
  snapshot.negativePromptDefault = nextDefault;
  snapshot.negativeExplicitClear = false;
  if ((snapshot.negativePrompt ?? "").trim() === "") {
    // Legacy omission is the untouched state: show the default so the wire
    // stays absent, exactly what the pre-#787 template produced.
    snapshot.negativePrompt = nextDefault;
  }
  return snapshot;
}

/**
 * Run a wholesale form rewrite while the print keeps its own identity.
 *
 * `title`, `fileUnder`, and `fileUnderMatch` name and file THIS print; none of
 * them is a model-owned generation control, and the desktop contract is that
 * only ⌘N (`useGenerateFormStore.clearComposer`) clears them. Every rewrite
 * short of that — both inspector Resets and a loaded template — goes through
 * here so it restores parameters without renaming or re-filing the print in
 * progress. `fileUnderAutoTag` rides along for a related reason: it mirrors
 * Settings ▸ Library, and a form rewrite is not a preference change.
 *
 * Deliberately NOT applied to `applyRequestToForm` / `applyMetadataToForm`,
 * which restore a recorded print and therefore restore its recorded identity
 * too.
 */
export function keepingPrintIdentity(form: GenerateForm, rewrite: () => void): void {
  const { title, fileUnder, fileUnderAutoTag, fileUnderMatch } = form;
  rewrite();
  form.title = title;
  form.fileUnder = fileUnder;
  form.fileUnderAutoTag = fileUnderAutoTag;
  form.fileUnderMatch = fileUnderMatch;
}

/**
 * Restore every generation knob to the selected model's defaults. The prompt
 * (with its expand provenance) and the model/family survive. Batch is a
 * general generation setting, so the primary Reset restores it to one. Any
 * prepared siblings remain retained and become explicitly stale rather than
 * being silently discarded. The print's name and filing survive too — see
 * {@link keepingPrintIdentity}.
 *
 * With no `ModelEntry` — an uninstalled or not-yet-resolved model — the named
 * model and family are kept and the form falls back to `newGenerateForm()`
 * scalars.
 */
export function resetFormToModelDefaults(
  form: GenerateForm,
  m: ModelEntry | null | undefined,
): void {
  const { prompt, originalPrompt, model, family } = form;
  keepingPrintIdentity(form, () => Object.assign(form, newGenerateForm()));
  if (m) {
    applyModelDefaults(form, m);
  } else {
    form.model = model;
    form.family = family;
  }
  form.prompt = prompt;
  form.originalPrompt = originalPrompt;
  form.batchSize = 1;
}

/**
 * The Advanced ("Fine controls") Reset. Source media lives in the primary
 * form now, so unlike {@link resetFormToModelDefaults} — the inspector's
 * deliberate wholesale reset — this one restores model defaults while every
 * staged media field (and the conditioning knobs rendered beside the wells:
 * strength, source fit) survives untouched.
 */
export function resetAdvancedToModelDefaults(
  form: GenerateForm,
  m: ModelEntry | null | undefined,
): void {
  const batchSize = form.batchSize;
  const media = {
    strength: form.strength,
    sourceImage: form.sourceImage,
    sourceImageName: form.sourceImageName,
    sourceImageWidth: form.sourceImageWidth,
    sourceImageHeight: form.sourceImageHeight,
    endFrame: form.endFrame,
    // The identity photo is media, not a knob: Reset clears the strength and
    // start step beside it (they rebuild from `newGenerateForm`) and leaves
    // the attached face where the user put it. `identitySupported` is
    // deliberately NOT preserved — the reset restores the model's default
    // pipeline, and the capability belongs to the recipe that resolves.
    identityImage: form.identityImage,
    imageAttachments: form.imageAttachments,
    sourceFit: form.sourceFit,
    maskImage: form.maskImage,
    controlImage: form.controlImage,
    controlModel: form.controlModel,
    controlScale: form.controlScale,
    sourceVideo: form.sourceVideo,
    extendVideo: form.extendVideo,
    extendOverlapFrames: form.extendOverlapFrames,
    extendDefaultOverlapFrames: form.extendDefaultOverlapFrames,
    keyframes: form.keyframes,
    enableAudio: form.enableAudio,
    audioFile: form.audioFile,
    h3Authoring: form.h3Authoring,
  };
  resetFormToModelDefaults(form, m);
  form.batchSize = generationCapabilitiesForFamily(form.family, form.model).forcesBatchSizeOne
    ? 1
    : batchSize;
  Object.assign(form, media);
}

/**
 * Assemble the wire request from the form, honoring the family's capabilities.
 * `pruneRequestForFamily` is the final guard so no unsupported field ever
 * ships even if the form retained a stale value.
 */
/**
 * The overlap a continuation built from this form will submit.
 *
 * The desktop and iPhone inspectors render this so the number on screen is
 * literally the number `buildRequest` puts on the wire — wan's select offers a
 * single option, so it never fires `@change` and the form field stays null.
 * The advertised default rides the form (`extendDefaultOverlapFrames`) because
 * `buildRequest` takes only the form.
 */
export function formExtendOverlapFrames(form: GenerateForm): number {
  return resolveExtendOverlapFrames(form.extendOverlapFrames, {
    family: form.family,
    extend_default_overlap_frames: form.extendDefaultOverlapFrames,
  });
}

export function buildRequest(form: GenerateForm): GenerateRequest {
  const caps = generationCapabilitiesForFamily(
    form.family,
    form.model,
    form.pipeline,
    form.guidanceCapabilities,
    form.sourceImageCapability,
  );
  const parsedSeed = form.seed.trim() === "" ? undefined : Number(form.seed);
  let loras: LoraWeight[] = form.loras.map((l) => ({ path: l.path, scale: l.scale }));

  // Camera motion rides the ordinary loras[] stack (mirrors the CLI's
  // --camera-control, run.rs): presets ship as the `camera-control:<preset>`
  // virtual alias the server resolves; explicit `.safetensors` paths pass
  // through raw. The host-provided capability list is the compatibility
  // authority; the serializer never guesses from a public model id.
  const cameraControl = form.cameraControl?.trim();
  if (caps.supportsAdvancedVideo && cameraControl) {
    loras = syncCameraMotionLora(
      loras,
      cameraControl,
      cameraControl,
      (path, scale) => ({
        path,
        scale,
      }),
      MAX_LORA_STACK,
    );
  }

  // WHICH conditioning this request carries — one shared decision, so an
  // exclusive (Klein) recipe ships `source_image` + `strength` OR
  // `edit_images`, never both, whatever the form is holding.
  const conditioning = conditioningForRequest(caps.sourceImageMode, {
    hasSource: Boolean(form.sourceImage),
    referenceCount: form.imageAttachments.length,
    lastWrite: form.exclusiveWell ?? null,
  });

  const req: GenerateRequest = {
    prompt: form.prompt.trim(),
    model: form.model,
    width: form.width,
    height: form.height,
    steps: form.steps,
    guidance: effectiveGenerationGuidance(caps, form.guidance),
    batch_size: caps.forcesBatchSizeOne || conditioning === "references" ? 1 : form.batchSize,
    output_format: form.outputFormat,
  };

  if (parsedSeed !== undefined && Number.isFinite(parsedSeed)) req.seed = parsedSeed;
  if (req.prompt && form.originalPrompt && form.originalPrompt !== req.prompt) {
    req.original_prompt = form.originalPrompt;
  }
  // Additive and validated client-side exactly as the server validates it;
  // an invalid title is dropped rather than failing the whole submit (the
  // header refuses to commit one, so this only guards stale snapshots).
  const title = validatePrintTitle(form.title ?? "");
  if (title.ok && title.value) req.title = title.value;
  // "File under" rides every request built from this form, so a Batch N
  // sibling and a prepared variation file exactly like the one-shot does.
  // Both fields stay ABSENT when nothing is filed.
  Object.assign(
    req,
    buildFileUnderRequestFields(
      form.fileUnder,
      form.title,
      form.fileUnderAutoTag,
      form.fileUnderMatch ? [form.fileUnderMatch] : [],
    ),
  );
  if (caps.supportsNegativePrompt) {
    // Tri-state (#787): text equal to the advertised default stays absent
    // (older servers behave identically), a cleared defaulted field ships
    // the explicit "" opt-out, typed text travels verbatim.
    const negative = negativePromptWireValue(
      form.negativePrompt,
      form.negativePromptDefault,
      form.negativeExplicitClear,
    );
    if (negative !== undefined) req.negative_prompt = negative;
  }
  if (caps.supportsScheduler && form.scheduler !== "default") req.scheduler = form.scheduler;
  if (caps.supportsCfgPlus && form.cfgPlus) req.cfg_plus = true;

  // Assigned key by key: an untouched control contributes nothing, which is
  // what keeps the resolved wan tier's own shift and distill strengths.
  Object.assign(req, wanRecipeToWire(form.wanRecipe, caps.wanRecipe));

  // qwen-edit ships the ordered picture strip (first = Target, rest =
  // References) and never source_image/strength; batch is already locked to 1
  // by forcesBatchSizeOne + pruneRequestForFamily.
  if (caps.supportsImg2img && conditioning === "references") {
    req.edit_images = [...form.imageAttachments];
  }

  if (caps.supportsImg2img && conditioning === "source" && form.sourceImage) {
    // Wan's first/last-frame render rides the keyframes contract: BOTH ends
    // travel as `keyframes` and `source_image` stays home — the engine
    // refuses a request carrying both ("first frame from either
    // source_image or keyframes[0], not both"), and admission counts
    // keyframes as source presence for an I2V-required checkpoint. The
    // closing index is computed here, from the frame count this request is
    // actually carrying, so changing clip length after attaching the end
    // frame can never ship a stale index. A lone source image is an
    // ordinary image-to-video request and sends no keyframes.
    const firstLast =
      caps.supportsEndFrame && form.endFrame
        ? firstLastFrameKeyframes(
            { base64: form.sourceImage, filename: form.sourceImageName },
            { base64: form.endFrame.base64, filename: form.endFrame.filename },
            form.frames,
          )
        : null;
    if (firstLast) {
      req.keyframes = firstLast;
    } else {
      req.source_image = form.sourceImage;
      if (form.sourceImageName) req.source_image_name = form.sourceImageName;
      // Wan pins the first frame exactly; it never reads strength, and the
      // advertised recipe is the authority when the host sends one.
      if (caps.supportsStrength && form.recipeCapabilities?.supportsStrength !== false) {
        req.strength = form.strength;
      }
      if (caps.supportsMask && form.maskImage) req.mask_image = form.maskImage;
    }
  }

  // The 3-D controls travel only for a recipe that advertises them, and only
  // the values that differ from the advertised defaults (the server applies
  // the same defaults and the print records what actually rendered).
  if (form.recipeCapabilities?.mesh) {
    const mesh = meshRequestFromForm(form.mesh ?? emptyMeshForm(), form.recipeCapabilities.mesh);
    if (mesh) req.mesh = mesh;
  }

  // Face identity is its own conditioning partition (#1224), gated on the
  // checkpoint's own advertised support. The photo is NEVER routed through
  // source-fit preprocessing — it is a face reference, not a composition
  // input — and an untouched knob stays absent so the server's own default
  // remains authoritative. The shared policy owns every one of those rules.
  Object.assign(
    req,
    identityRequestFields({
      supported: form.identitySupported === true,
      image: form.identityImage,
      weight: form.identityWeight,
      startStep: form.identityStartStep,
    }),
  );

  // ControlNet is independent conditioning, not an img2img derivative. An
  // SD1.5 request may carry a control image without a source image; nesting it
  // under source_image silently discarded that valid text-to-image workflow.
  if (caps.supportsControlNet && form.controlImage) {
    req.control_image = form.controlImage;
    if (form.controlModel.trim()) {
      req.control_model = form.controlModel.trim();
      req.control_scale = form.controlScale;
    }
  }

  if (caps.supportsLora && loras.length) req.loras = loras;

  // Video requests use this same model selection to queue a durable
  // Framewise upscale after the source clip is published.
  if ((!caps.supportsAudio || caps.supportsVideo) && form.upscaleModel) {
    req.upscale_model = form.upscaleModel;
  }

  if (caps.supportsVideo) {
    if (!(form.predictDuration && form.durationPredictionSupported)) {
      req.frames = form.frames;
    }
    req.fps = form.fps;
    if (caps.supportsAudio) {
      req.enable_audio = form.enableAudio;
      // `video_only` is an LTX-2 request field (server validation refuses it
      // elsewhere), so a flag parked from an earlier LTX-2 selection must not
      // ride a MiniMax H3 request just because H3 also supports audio —
      // matching the web builder's family gate.
      if (caps.supportsAdvancedVideo) {
        const videoOnly = requestVideoOnly(form.videoOnly, {
          audioEnabled: form.enableAudio,
          audioOnlyPipeline: isAudioOnlyPipeline(form.pipeline),
          hasConditioningAudio: form.audioFile !== null,
          isExtend: form.extendVideo !== null,
        });
        if (videoOnly) req.video_only = videoOnly;
      }
    }
  }

  // Continuation is per family, not part of the LTX-2 suite (#783): wan
  // continues by seeding the render with the source clip's final frame, so
  // its fields have to survive outside `supportsAdvancedVideo` or an offered
  // wan continuation goes out as a plain text-to-video job.
  if (familySupportsExtend(form.family) && form.extendVideo) {
    req.extend_video = form.extendVideo.base64;
    // Only travels with a clip to continue — the server rejects a bare
    // overlap — but when it does travel it carries the value the inspector is
    // showing, resolved by the one shared authority. Leaving it absent handed
    // the host its own default, which for wan is the family-wide 17 that
    // `wan/pipeline.rs`'s `extend_inner` refuses.
    req.extend_overlap_frames = formExtendOverlapFrames(form);
  }

  if (caps.supportsAdvancedVideo) {
    if (form.sourceVideo) req.source_video = form.sourceVideo.base64;
    if (form.keyframes.length) {
      req.keyframes = form.keyframes.map<KeyframeConditionWire>((k) => ({
        frame: k.frame,
        image: k.image.base64,
        name: k.image.filename,
      }));
    }
    if (form.icLoraControl) {
      req.ic_lora_control = form.icLoraControl;
      // Lip dub is a pipeline of its own; every other adapter drives `ic-lora`.
      req.pipeline = pipelineForControlId(form.icLoraControl);
    } else if (form.pipeline) req.pipeline = form.pipeline;
    if (form.retakeRange) req.retake_range = form.retakeRange;
    if (form.spatialUpscale) req.spatial_upscale = form.spatialUpscale;
    if (form.temporalUpscale) req.temporal_upscale = form.temporalUpscale;
    const guidanceOverrides = guidanceOverridesToWire(form.guidanceOverrides);
    if (guidanceOverrides) req.guidance_overrides = guidanceOverrides;
    // a2-vid (audio-to-video) requires conditioning audio; other pipelines ignore it.
    if (form.pipeline === "a2-vid" && form.audioFile) req.audio_file = form.audioFile.base64;
  }

  // Last, and after family pruning: an audio-only pipeline renders no frames,
  // so every conditioning input, upscaler, and a `false` audio flag is
  // something the server refuses. Stripping here rather than on the pipeline
  // transition keeps the user's source media intact if they switch back.
  const finalized = stripAudioOnlyIncompatibleFields(
    serializeMinimaxH3Authoring(
      pruneRequestForFamily(
        req,
        form.family,
        form.model,
        form.sourceImageCapability,
        form.recipeCapabilities,
      ),
      form.family,
      form.model,
      form.h3Authoring ?? emptyMinimaxH3AuthoringState(),
    ),
  );
  // Crop provenance rides only when the wire actually carries fitted source
  // media (the server echoes it verbatim into OutputMetadata so Reuse
  // settings and running-job selection can restore the crop controls). A
  // canvasless recipe never fits the source to a canvas, so it records none.
  const canvasless = form.recipeCapabilities?.canvasless ?? isMeshFamily(form.family);
  if (
    !canvasless &&
    (finalized.source_image || finalized.edit_images?.length || finalized.keyframes?.length)
  ) {
    finalized.source_fit = form.sourceFit;
  }
  return finalized;
}

/**
 * Title and creation-time filing for a SEQUENCE, which carries them on the
 * `POST /api/chain-jobs` body rather than a `GenerateRequest`.
 *
 * They apply to the stitched print only — an intermediate clip is a working
 * artifact inside the job dir and never reaches the gallery — so this is one
 * timeline's filing, not one per clip. Same validation and same absent-when-
 * empty shape as `buildRequest`, because it is the same wire contract.
 */
export function chainFilingFields(form: GenerateForm): {
  title?: string;
  tags?: string[];
  collection?: { name: string };
} {
  const fields: { title?: string; tags?: string[]; collection?: { name: string } } = {};
  const title = validatePrintTitle(form.title ?? "");
  if (title.ok && title.value) fields.title = title.value;
  return Object.assign(
    fields,
    buildFileUnderRequestFields(
      form.fileUnder,
      form.title,
      form.fileUnderAutoTag,
      form.fileUnderMatch ? [form.fileUnderMatch] : [],
    ),
  );
}

/**
 * Rebuild the "File under" draft from a print's recorded filing (Reuse
 * settings, or restoring an exact queued request).
 *
 * The ghost chip is never restored as a chip: it stays derived from the live
 * title, so a recorded copy of the title slug is dropped from the manual list
 * rather than coming back as a duplicate. Its ABSENCE is restored though — a
 * print whose recorded tags don't include its own title slug was filed with
 * the ghost removed, and re-offering it would quietly re-file the reuse under
 * a tag the original never carried. Absent tags are legacy silence, not an
 * opt-out, so they restore the untouched default.
 */
export function restoredFileUnderState(
  title: string,
  autoTagEnabled: boolean,
  tags: readonly string[] | null | undefined,
  collectionName: string | null | undefined,
): FileUnderState {
  let state = emptyFileUnderState();
  const ghost = deriveGhostTag(title, autoTagEnabled);
  if (Array.isArray(tags)) {
    const ghostKey = ghost === null ? null : requestTagKey(ghost);
    let sawGhost = false;
    for (const tag of tags) {
      if (ghostKey !== null && requestTagKey(tag) === ghostKey) {
        sawGhost = true;
        continue;
      }
      state = addTag(state, tag);
    }
    if (ghostKey !== null && !sawGhost) state = { ...state, ghostRemoved: true };
  }
  const name = collectionName?.trim();
  if (name) state = pickCollection(state, { name });
  return state;
}

const KNOWN_SCHEDULERS: readonly Scheduler[] = [
  "default",
  "ddim",
  "euler-ancestral",
  "uni-pc",
  "euler",
  "dpm-pp",
];

/** Match separator-insensitively: legacy rows carry `"unipc"` / `"uni_pc"` /
 * `"euler_ancestral"` where the wire now writes `"uni-pc"` /
 * `"euler-ancestral"`. Squash `-`/`_` to compare. */
const squash = (name: string): string => name.toLowerCase().replace(/[-_]/g, "");
const SCHEDULER_BY_SQUASHED = new Map<string, Scheduler>(
  KNOWN_SCHEDULERS.map((s) => [squash(s), s]),
);

/** Collapse a metadata scheduler value (`"ddim"` or serde-tagged
 * `{ ddim: … }`) onto the form's string union; anything unknown → default. */
function normalizeMetadataScheduler(s: OutputMetadata["scheduler"]): Scheduler {
  if (!s) return "default";
  const name = typeof s === "string" ? s : (Object.keys(s)[0] ?? "default");
  return SCHEDULER_BY_SQUASHED.get(squash(name)) ?? "default";
}

/** Display name for a LoRA restored from metadata — the path's basename. */
function loraNameFromPath(path: string): string {
  if (cameraMotionFromLoraPath(path)) return cameraMotionLoraLabel(path);
  const base = path.split("/").pop() ?? path;
  return base.replace(/\.safetensors$/i, "");
}

/**
 * Full-fidelity "Reuse settings": restore every serialized generation knob a
 * gallery item's embedded metadata carries (port of the web SPA's
 * `applyMetadataToForm`). Static-seed semantics recreate the print exact-ish;
 * binary media (source/mask/control/video/audio bytes) is cleared because
 * output metadata never contains it. When the model isn't installed anywhere,
 * the name is still set (family blank) and the existing missing-model UI takes
 * over — reuse never forces a host, so model-aware Auto routing still applies.
 */
export function applyMetadataToForm(
  form: GenerateForm,
  metadata: OutputMetadata,
  models: ModelEntry[] = [],
): void {
  form.cameraControl = null;
  const model = findInstalledModel(models, metadata.model);
  if (model) {
    applyModelDefaults(form, model);
  } else {
    form.model = metadata.model;
    form.family = "";
    form.negativePromptDefault = "";
    form.recipeCapabilities = null;
  }

  form.prompt = metadata.prompt ?? "";
  form.originalPrompt = metadata.original_prompt ?? null;
  form.title = metadata.title ?? "";
  form.fileUnder = restoredFileUnderState(
    form.title,
    form.fileUnderAutoTag,
    metadata.tags,
    metadata.collection,
  );
  form.fileUnderMatch = null;
  // Absence predates truthful recording: on a defaulted model it means the
  // default conditioned the render, so restore shows it rather than
  // silently flipping the reuse into an explicit empty-uncond opt-out.
  form.negativePrompt = restoredNegativePrompt(
    metadata.negative_prompt,
    form.negativePromptDefault,
  );
  // A recorded "" is the explicit opt-out; when this restore ran before the
  // model rows resolved the empty control is otherwise identical to
  // "untouched" — the marker carries the print's authority until the default
  // is known (#787 round 3).
  form.negativeExplicitClear = restoredNegativeExplicitClear(metadata.negative_prompt);
  // Prefer the pre-upscale generation canvas over the saved raster size.
  form.width = metadata.generation_width || metadata.width || form.width;
  form.height = metadata.generation_height || metadata.height || form.height;
  form.steps = metadata.steps || form.steps;
  form.guidance = metadata.guidance ?? form.guidance;
  form.seed = metadata.seed == null ? "" : String(metadata.seed);
  form.scheduler = normalizeMetadataScheduler(metadata.scheduler);
  form.cfgPlus = metadata.cfg_plus ?? false;
  if (metadata.strength != null) form.strength = metadata.strength;
  // Additive crop provenance: restore only a policy that parses exactly —
  // corrupt or future-shaped values must never poison the live form.
  const recordedFit = parseSourceFitPolicy(metadata.source_fit);
  if (recordedFit) form.sourceFit = recordedFit;

  const loras =
    metadata.loras ??
    (metadata.lora ? [{ path: metadata.lora, scale: metadata.lora_scale ?? 1.0 }] : []);
  form.loras = loras.slice(0, MAX_LORA_STACK).map<FormLora>((l) => ({
    path: l.path,
    name: loraNameFromPath(l.path),
    scale: l.scale,
    trainedWords: [],
  }));
  form.cameraControl =
    form.loras
      .map((lora) => cameraMotionFromLoraPath(lora.path))
      .find((value): value is string => value !== null) ?? null;

  form.controlModel = metadata.control_model ?? "";
  if (metadata.control_scale != null) form.controlScale = metadata.control_scale;
  form.upscaleModel = metadata.upscale_model ?? "";
  if (metadata.output_format) form.outputFormat = metadata.output_format;
  form.outputFormat =
    coerceFormOutputFormat(form.outputFormat, form.family, form.recipeCapabilities) ??
    form.outputFormat;
  // A mesh print recorded the resolved controls; a raster print has none.
  form.mesh = meshFormFromMetadata(metadata.mesh);
  if (form.recipeCapabilities?.canvasless) {
    // `width`/`height` on a mesh print describe its poster, not a canvas.
    form.width = 0;
    form.height = 0;
  }

  // Video params (`video_frames`/`video_fps` are legacy desktop aliases).
  const frames = metadata.frames ?? metadata.video_frames;
  if (frames != null) form.frames = frames;
  // Preserve authored provenance even when Library reuse happens before the
  // inventory row arrives. Request building remains fail-closed against
  // `durationPredictionSupported`, and the later capability reconcile either
  // validates this opt-in or clears it.
  form.predictDuration = metadata.duration_prediction_requested === true;
  const fps = metadata.fps ?? metadata.video_fps;
  if (fps != null) form.fps = fps;
  if (metadata.enable_audio != null) form.enableAudio = metadata.enable_audio;
  form.videoOnly = metadata.video_only === true;
  form.pipeline = pipelineForSettingsReuse(metadata);
  form.icLoraControl = metadata.ic_lora_control ?? null;
  form.retakeRange = metadata.retake_range ?? null;
  form.spatialUpscale = metadata.spatial_upscale ?? null;
  form.temporalUpscale = metadata.temporal_upscale ?? null;
  form.guidanceOverrides = guidanceOverridesFromWire(metadata.guidance_overrides);
  form.wanRecipe = wanRecipeFromWire(metadata);

  // Output metadata never carries source/mask/control/video/audio bytes —
  // clear any stale attachment instead of silently pairing it with the print.
  // (The async source restore may repopulate the pair afterwards.)
  form.sourceImage = null;
  form.sourceImageName = null;
  form.sourceImageWidth = null;
  form.sourceImageHeight = null;
  form.maskImage = null;
  form.controlImage = null;
  form.imageAttachments = [];
  form.sourceVideo = null;
  form.keyframes = [];
  // The closing frame is binary media too: a staged end frame from the
  // previous draft must never silently pair with this print's restored or
  // newly attached opening image (the restore notice already says it cannot
  // be rebuilt from metadata).
  form.endFrame = null;
  form.audioFile = null;
  // Identity: the knobs restore exactly, and the photo becomes a bytes-less
  // reattach descriptor the async stash lookup fills in. Metadata records the
  // digest, never the face, so an unresolved lookup must show the reattach
  // state rather than silently render a different person. `identityRequestFields`
  // refuses an empty payload, so this descriptor can never reach the wire.
  const identity = identityReuse(metadata);
  form.identityImage = identity
    ? { filename: identity.name ?? "identity photo", base64: "" }
    : null;
  form.identityWeight = identity ? identity.weight : null;
  form.identityStartStep = identity ? identity.startStep : null;
  form.h3Authoring = {
    ...emptyMinimaxH3AuthoringState(),
    firstFrame:
      minimaxH3TaskForModel(metadata.model) === "fl2va"
        ? minimaxH3BoundaryFromSourceMetadata(
            metadata.source_image_name,
            metadata.source_image_sha256,
          )
        : null,
    lastFrame:
      minimaxH3TaskForModel(metadata.model) === "fl2va"
        ? minimaxH3ClosingBoundaryFromMetadata(
            metadata.frames ?? metadata.video_frames,
            metadata.keyframes,
          )
        : null,
    references: minimaxH3ReferenceDraftsFromMetadata(metadata.references),
  };
}

/** Lossy scalar prefill used by non-gallery callers (palette, history, jobs). */
export interface ScalarPrefill {
  prompt: string;
  model: string;
  seed: number | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  upscaleModel?: string;
}

/** Full-fidelity prefill: the gallery item's embedded metadata, verbatim. */
export interface MetadataPrefill {
  metadata: OutputMetadata;
  /** Present only when Queue selection supplied the metadata. Create uses it
   * to show that exact server-owned render while the canonical Library reuse
   * mapper restores the settings. */
  queueSelection?: import("@studio/api/generationSelection").SelectedQueuePreviewSource;
}

/** Exact queued request, including advanced and media inputs. */
export interface RequestPrefill {
  request: GenerateRequest;
}

export type GeneratePrefill = ScalarPrefill | MetadataPrefill | RequestPrefill;

export function applyRequestToForm(
  form: GenerateForm,
  request: GenerateRequest,
  models: ModelEntry[],
): void {
  // The auto-tag mirror is a Settings preference, not part of the request.
  const fileUnderAutoTag = form.fileUnderAutoTag;
  Object.assign(form, newGenerateForm());
  form.fileUnderAutoTag = fileUnderAutoTag;
  const model = findInstalledModel(models, request.model);
  if (model) applyModelDefaults(form, model);
  form.prompt = request.prompt;
  form.originalPrompt = request.original_prompt ?? null;
  form.title = request.title ?? "";
  form.fileUnder = restoredFileUnderState(
    form.title,
    form.fileUnderAutoTag,
    request.tags,
    request.collection?.name,
  );
  form.fileUnderMatch = null;
  form.negativePrompt = restoredNegativePrompt(request.negative_prompt, form.negativePromptDefault);
  form.negativeExplicitClear = restoredNegativeExplicitClear(request.negative_prompt);
  form.model = request.model;
  form.width = request.width;
  form.height = request.height;
  form.steps = request.steps;
  form.guidance = request.guidance ?? form.guidance;
  form.seed = request.seed == null ? "" : String(request.seed);
  form.scheduler = request.scheduler ?? "default";
  form.cfgPlus = request.cfg_plus ?? false;
  form.batchSize = request.batch_size ?? 1;
  form.outputFormat =
    coerceFormOutputFormat(
      request.output_format ?? form.outputFormat,
      form.family,
      form.recipeCapabilities,
    ) ?? form.outputFormat;
  form.mesh = meshFormFromMetadata(request.mesh);
  form.upscaleModel = request.upscale_model ?? "";
  form.strength = request.strength ?? form.strength;
  form.sourceImage = request.source_image ?? null;
  form.sourceImageName = request.source_image_name ?? null;
  form.sourceImageWidth = null;
  form.sourceImageHeight = null;
  form.sourceFit = parseSourceFitPolicy(request.source_fit) ?? form.sourceFit;
  // A running job's exact request still carries the identity bytes, so this
  // restore is lossless where the metadata one is not.
  form.identityImage = request.id_image
    ? { filename: request.id_image_name ?? "identity photo", base64: request.id_image }
    : null;
  form.identityWeight = request.id_weight ?? null;
  form.identityStartStep = request.id_start_step ?? null;
  form.imageAttachments = [...(request.edit_images ?? [])];
  form.maskImage = request.mask_image ?? null;
  form.controlImage = request.control_image ?? null;
  form.controlModel = request.control_model ?? "";
  form.controlScale = request.control_scale ?? 1;
  const loras = request.loras ?? (request.lora ? [request.lora] : []);
  form.loras = loras.slice(0, MAX_LORA_STACK).map((lora) => ({
    path: lora.path,
    name: loraNameFromPath(lora.path),
    scale: lora.scale,
    trainedWords: [],
  }));
  form.cameraControl =
    form.loras
      .map((lora) => cameraMotionFromLoraPath(lora.path))
      .find((value): value is string => value !== null) ?? null;
  form.frames = request.frames ?? form.frames;
  form.predictDuration =
    request.frames == null &&
    request.model.startsWith("ltx-2.5") &&
    form.durationPredictionSupported;
  form.fps = request.fps ?? form.fps;
  form.enableAudio = request.enable_audio ?? false;
  form.videoOnly = request.video_only === true;
  form.audioFile = request.audio_file
    ? { filename: "Audio input", base64: request.audio_file }
    : null;
  form.sourceVideo = request.source_video
    ? { filename: "Video input", base64: request.source_video }
    : null;
  form.keyframes = (request.keyframes ?? []).map((keyframe) => ({
    frame: keyframe.frame,
    image: {
      filename: keyframe.name ?? `Keyframe ${keyframe.frame}`,
      base64: keyframe.image,
    },
  }));
  // A wan first/last-frame request carries both stills as keyframes and no
  // `source_image` (the engine refuses both together), so restoring it means
  // mapping the pair back into the wells the serializer read them from. Wan
  // has no mid-clip keyframe UI, so the raw list must not linger where only
  // the LTX-2 advanced panel would show it.
  if (
    isWanFamily(form.family) &&
    !request.source_image &&
    form.keyframes.length === 2 &&
    form.keyframes[0]!.frame === 0
  ) {
    const [first, last] = form.keyframes;
    form.sourceImage = first!.image.base64;
    form.sourceImageName = first!.image.filename;
    form.endFrame = { filename: last!.image.filename, base64: last!.image.base64 };
    form.keyframes = [];
  }
  form.pipeline = request.pipeline ?? null;
  form.icLoraControl = request.ic_lora_control ?? null;
  form.retakeRange = request.retake_range ?? null;
  form.spatialUpscale = request.spatial_upscale ?? null;
  form.temporalUpscale = request.temporal_upscale ?? null;
  form.guidanceOverrides = guidanceOverridesFromWire(request.guidance_overrides);
  form.wanRecipe = wanRecipeFromWire(request);
  if (isMinimaxH3Family(form.family) || minimaxH3TaskForModel(request.model)) {
    form.h3Authoring ??= emptyMinimaxH3AuthoringState();
    form.h3Authoring.references = (request.references ?? []).map((reference) => ({
      reference: JSON.parse(JSON.stringify(reference)),
    }));
    if (request.source_image) {
      form.h3Authoring.firstFrame = {
        filename: request.source_image_name ?? "First frame",
        mimeType: "image/*",
        width: 0,
        height: 0,
        data: request.source_image,
      };
    }
    const finalFrame = (request.frames ?? form.frames) - 1;
    const last = request.keyframes?.find((keyframe) => keyframe.frame === finalFrame);
    if (last) {
      form.h3Authoring.lastFrame = {
        filename: last.name ?? "Last frame",
        mimeType: "image/*",
        width: 0,
        height: 0,
        data: last.image,
      };
    }
  }
}

/**
 * Route a composer prefill into the form: gallery reuse ships full metadata
 * through {@link applyMetadataToForm}; everything else keeps the legacy
 * scalar copy exactly as before.
 */
export function applyPrefillToForm(
  form: GenerateForm,
  prefill: GeneratePrefill,
  models: ModelEntry[] = [],
): void {
  if ("metadata" in prefill) {
    applyMetadataToForm(form, prefill.metadata, models);
    return;
  }
  if ("request" in prefill) {
    applyRequestToForm(form, prefill.request, models);
    return;
  }
  form.prompt = prefill.prompt;
  form.model = prefill.model;
  form.seed = prefill.seed !== null ? String(prefill.seed) : "";
  form.width = prefill.width;
  form.height = prefill.height;
  form.steps = prefill.steps;
  form.guidance = prefill.guidance;
  form.upscaleModel = prefill.upscaleModel ?? "";
  const m = findInstalledModel(models, prefill.model);
  if (m) {
    // The named model's ADVERTISED recipe answers for the format, the canvas
    // and the mesh controls. Copying only `family` left the previous model's
    // snapshot behind, so a ⌘K "Generate with sdxl" after Hunyuan3D still
    // pinned `glb` onto the raster request.
    reconcileModelCapabilities(form, m);
  } else {
    // Nothing installed can answer for this model: a stale snapshot must not
    // speak for it (the same reading `applyMetadataToForm` takes).
    form.family = "";
    form.recipeCapabilities = null;
    form.mesh = emptyMeshForm();
  }
}
