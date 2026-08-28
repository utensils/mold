import { ref, watch, type Ref, type WatchStopHandle } from "vue";
import type {
  GenerateFormState,
  GenerateRequestWire,
  LoraSelection,
  ModelInfoExtended,
  OutputMetadata,
  OutputFormat,
  Scheduler,
} from "../types";
import { MAX_LORA_STACK } from "../types";
import {
  clearMemoryDraftsForTest,
  getDraftMedia,
  newDraftId,
} from "../lib/draftMediaStore";
import {
  putDurableMediaBatch,
  type DraftMediaRecord,
} from "@studio/lib/draftMediaStore";
import {
  generationCapabilitiesForFamily,
  isQwenImageEditFamily,
  isVideoFamily,
  supportsLora,
  supportsNegativePrompt,
  supportsScheduler,
} from "../lib/generateCapabilities";
import { composeStyle } from "../lib/stylePresets";
import {
  defaultSourceFitPolicy,
  parseSourceFitPolicy,
} from "@studio/lib/sourceFit";
import {
  effectiveNegativeDefault,
  negativePromptOnDefaultChange,
  negativePromptWireValue,
  restoredNegativeExplicitClear,
  restoredNegativePrompt,
} from "@studio/lib/negativePrompt";
import { defaultVideoFps } from "@studio/lib/sequence";
import { videoFramesForModelSelection } from "@studio/lib/videoDuration";
import { pipelineForSettingsReuse } from "@studio/lib/outputReuse";
import {
  familySupportsExtend,
  resolveExtendOverlapFrames,
} from "@studio/lib/extend";
import {
  effectiveGenerationRecipe,
  fixedRecipeControlOverrides,
} from "@studio/lib/generationProfile";
import { effectiveGenerationGuidance } from "@studio/lib/generationCapabilities";
import {
  cameraMotionLoraPath,
  normalizeCameraMotionLoraState,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import { pipelineForControlId } from "@studio/lib/ltx2Control";
import {
  identityRequestFields,
  identityReuse,
  supportsIdentity,
} from "@studio/lib/identityConditioning";
import { validatePrintTitle } from "@studio/lib/libraryOrganization";
import { firstLastFrameKeyframes } from "@studio/lib/sourceImageCapability";
import { stripAudioOnlyIncompatibleFields } from "@studio/lib/ltx2Pipeline";
import {
  emptyGuidanceOverrides,
  guidanceOverridesFromWire,
  guidanceOverridesToWire,
} from "@studio/lib/guidanceOverrides";
import {
  emptyWanRecipe,
  wanRecipeFromWire,
  wanRecipeToWire,
} from "@studio/lib/wanRecipe";
import {
  MINIMAX_H3_REVIEWED_COMPACT_FRAMES,
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
  type MinimaxH3BoundaryImage,
  type MinimaxH3ReferenceDraft,
} from "@studio/lib/minimaxH3Authoring";

/** The prompt actually sent to the server: the textarea content composed with
 * the active style preset (the shared kit substitutes a "{prompt}" template,
 * otherwise it comma-appends). Never mutates `state.prompt` — the style row is
 * a request-time modifier, not a rewrite of what the user typed. Shared by
 * `toRequest` and the estimate/summary display so both agree. */
export function promptWithStyle(state: GenerateFormState): string {
  return composeStyle(state.prompt, state.stylePreset ?? "", {
    supportsNegativePrompt: false,
  }).prompt;
}

const STORAGE_KEY = "mold.generate.form";
const FORM_VERSION = 3 as const;
const QWEN_IMAGE_EDIT_FAMILY = "qwen-image-edit";

function selectedFamily(s: GenerateFormState): string {
  if (s.modelFamily) return s.modelFamily;
  if (s.model.startsWith(`${QWEN_IMAGE_EDIT_FAMILY}:`)) {
    return QWEN_IMAGE_EDIT_FAMILY;
  }
  if (s.model.startsWith("sdxl")) return "sdxl";
  if (s.model.startsWith("sd15") || s.model.startsWith("sd1.5")) return "sd15";
  if (s.model.startsWith("sd3.5")) return "sd3.5";
  if (s.model.startsWith("sd3")) return "sd3";
  if (s.model.startsWith("flux2") || s.model.startsWith("flux.2"))
    return "flux2";
  if (s.model.startsWith("flux")) return "flux";
  if (s.model.startsWith("z-image")) return "z-image";
  if (s.model.startsWith("qwen-image-edit")) return "qwen-image-edit";
  if (s.model.startsWith("qwen-image")) return "qwen-image";
  if (s.model.startsWith("ltx-video")) return "ltx-video";
  if (s.model.startsWith("ltx2") || s.model.startsWith("ltx-2")) return "ltx2";
  // Every Wan manifest name is versioned — `wan21-…`, `wan22-…` — so the
  // prefix is unambiguous here in a way a bare `wan` substring would not be.
  // This is only the fallback for a stored form whose `modelFamily` predates
  // the field; the server-supplied family above wins whenever it is present.
  if (s.model.startsWith("wan2") || s.model.startsWith("wan-")) return "wan";
  if (s.model.startsWith("minimax-h3") || s.model.startsWith("minimax_h3")) {
    return "minimax-h3";
  }
  return "";
}

function defaultForm(): GenerateFormState {
  return {
    version: FORM_VERSION,
    prompt: "",
    title: null,
    originalPrompt: null,
    stylePreset: null,
    negativePrompt: "",
    negativePromptDefault: "",
    negativeExplicitClear: false,
    model: "",
    modelFamily: "",
    width: 1024,
    height: 1024,
    steps: 20,
    guidance: 3.5,
    guidanceCapabilities: null,
    sourceImageCapability: null,
    endFrame: null,
    identityImage: null,
    identityWeight: null,
    identityStartStep: null,
    identitySupported: null,
    seedMode: "random",
    seed: null,
    batchSize: 1,
    strength: 0.75,
    frames: null,
    predictDuration: false,
    fps: null,
    scheduler: null,
    cfgPlus: false,
    outputFormat: "png",
    expand: { enabled: false, variations: 1, familyOverride: null },
    sourceFitPolicy: defaultSourceFitPolicy(),
    imageAttachments: [],
    maskImage: null,
    controlImage: null,
    controlModel: "",
    controlScale: 1.0,
    upscaleModel: "",
    gifPreview: false,
    audioFile: null,
    audioFilePath: "",
    sourceVideo: null,
    sourceVideoPath: "",
    extendVideo: null,
    extendVideoPath: "",
    extendOverlapFrames: null,
    keyframes: [],
    pipeline: null,
    icLoraControl: null,
    retakeRange: null,
    spatialUpscale: null,
    temporalUpscale: null,
    guidanceOverrides: emptyGuidanceOverrides(),
    wanRecipe: emptyWanRecipe(),
    cameraControl: null,
    placement: null,
    loras: [],
    enableAudio: null,
    h3Authoring: emptyMinimaxH3AuthoringState(),
  };
}

function cloneFormState<T>(state: T): T {
  return JSON.parse(JSON.stringify(state)) as T;
}

/** Remove browser-only binary media from a form snapshot before writing it to
 * localStorage or a local generation template. Server-local path fields
 * (`audioFilePath`, `sourceVideoPath`) are preserved because they are stable
 * references; upload/gallery base64 payloads remain outside serialized form
 * state and durable Create drafts hydrate them from IndexedDB instead. */
export function sanitizePersistedForm(
  state: GenerateFormState,
): GenerateFormState {
  // Pull every byte-bearing root out before cloning the ordinary form fields.
  // JSON-cloning the full state first briefly doubled multi-gigabyte H3 media
  // in memory even though those bytes were immediately stripped below.
  const {
    imageAttachments,
    endFrame,
    identityImage,
    maskImage,
    controlImage,
    audioFile,
    sourceVideo,
    extendVideo,
    keyframes,
    h3Authoring,
    ...metadata
  } = state;
  return {
    ...cloneFormState(metadata),
    version: FORM_VERSION,
    imageAttachments: imageAttachments.map(stripMediaBytes),
    endFrame: endFrame ? stripMediaBytes(endFrame) : null,
    // A face photo is the last payload that should sit in localStorage: the
    // descriptor is kept so the well can offer a reattach, the bytes live in
    // IndexedDB under the draft id like every other staged image.
    identityImage: identityImage ? stripMediaBytes(identityImage) : null,
    maskImage: maskImage ? stripMediaBytes(maskImage) : null,
    controlImage: controlImage ? stripMediaBytes(controlImage) : null,
    audioFile: audioFile ? stripMediaBytes(audioFile) : null,
    sourceVideo: sourceVideo ? stripMediaBytes(sourceVideo) : null,
    extendVideo: extendVideo ? stripMediaBytes(extendVideo) : null,
    keyframes: keyframes.map((k) => ({
      ...k,
      image: stripMediaBytes(k.image),
    })),
    h3Authoring: stripH3AuthoringMedia(h3Authoring),
  };
}

/** Clone the generation configuration that can be safely represented in a
 * web-local template. Binary source/mask/control/media bytes are stripped by
 * `sanitizePersistedForm`; callers can store separate filename references for
 * display, but loading the template will not silently pretend those bytes are
 * still available. */
export function cloneTemplateForm(state: GenerateFormState): GenerateFormState {
  return sanitizePersistedForm(state);
}

function modelDefaultsPatch(
  current: GenerateFormState,
  model: ModelInfoExtended,
): GenerateFormState {
  const next: GenerateFormState = {
    ...cloneFormState(current),
    model: model.name,
    modelFamily: model.family,
    width: model.default_width,
    height: model.default_height,
    steps: model.default_steps,
    guidance: model.default_guidance,
    guidanceCapabilities: model.guidance_capabilities ?? null,
    // Snapshotted like `guidanceCapabilities`: request assembly runs long
    // after the catalog row went out of scope, and a stale snapshot from the
    // previously selected checkpoint would be worse than none.
    sourceImageCapability: model.source_image ?? null,
    // Same reason: `toRequest` re-asks the resolved row when it has one, and
    // this snapshot is what answers when it does not.
    identitySupported: supportsIdentity(
      effectiveGenerationRecipe(model, null),
      model,
    ),
    loras: [],
    icLoraControl: null,
  };
  if (
    model.supports_duration_prediction !== true ||
    model.runtime_ready === false
  ) {
    next.predictDuration = false;
  }
  // Staged media survives a capability-losing switch (see the source-media
  // bridge below); the identity SETTINGS do not. A weight left set on a
  // checkpoint whose Identity group no longer renders is an inline error the
  // user cannot see, let alone clear — the same reason the LTX-2 knobs clear
  // while its audio/video stays parked.
  if (!next.identitySupported) {
    next.identityWeight = null;
    next.identityStartStep = null;
  }
  // #787: a Negative field still showing the previous model's advertised
  // default follows the new model (that is also how the default first
  // appears); typed text and an explicit clear are user authority. The
  // family constant backs an older server that omits the additive field so
  // a known default (and the "" opt-out against it) never decays to absence.
  const nextNegativeDefault = effectiveNegativeDefault(model, model.family);
  next.negativePrompt = negativePromptOnDefaultChange(
    current.negativePrompt,
    current.negativePromptDefault ?? "",
    nextNegativeDefault,
  );
  next.negativePromptDefault = nextNegativeDefault;
  // Selecting a model is fresh authority: any deferred restore-time clear
  // marker (#787 round 3) has been resolved by the rules above.
  next.negativeExplicitClear = false;
  const capabilities = generationCapabilitiesForFamily(
    model.family,
    model.name,
    null,
    null,
    model.source_image,
    effectiveGenerationRecipe(model, null),
  );
  if (capabilities.supportsVideo) {
    next.frames = isMinimaxH3Family(model.family)
      ? (model.default_frames ?? MINIMAX_H3_REVIEWED_COMPACT_FRAMES)
      : videoFramesForModelSelection(next.frames, model);
    // The model's advertised rate is applied like steps/guidance — it is only
    // absent-server/absent-field that leaves the current value in place.
    next.fps = defaultVideoFps(model, next.fps);
  } else {
    next.frames = null;
    next.fps = null;
    next.gifPreview = false;
  }
  const formats = capabilities.outputFormats as OutputFormat[];
  if (!formats.includes(next.outputFormat)) {
    next.outputFormat = formats[0];
  }
  next.enableAudio = capabilities.supportsAudio
    ? model.supports_audio !== false
    : null;
  if (!capabilities.supportsScheduler) {
    next.scheduler = null;
  } else if (
    next.scheduler !== null &&
    !capabilities.schedulerOptions.includes(next.scheduler)
  ) {
    // The UNet schedulers and wan's flow solvers share the wire slot but are
    // rejected on each other's families — membership in the new model's
    // options is the test, not mere scheduler support (codex review).
    next.scheduler = null;
  }
  if (!capabilities.supportsCfgPlus) {
    next.cfgPlus = false;
  }
  // The recipe knobs are wan's alone, and the strengths additionally need a
  // distill slot — a value carried over from another model would be rejected,
  // not ignored, so it is cleared on the way in.
  if (!capabilities.wanRecipe.supported) {
    next.wanRecipe = emptyWanRecipe();
  } else if (!capabilities.wanRecipe.supportsDistillStrength) {
    next.wanRecipe = {
      ...(next.wanRecipe ?? emptyWanRecipe()),
      distillStrengthHigh: null,
      distillStrengthLow: null,
    };
  }
  // Continuation is per family, not part of the LTX-2 suite (#783) — wan
  // continues too, so the staged clip is cleared on leaving a family that can
  // continue at all rather than on leaving LTX-2. Desktop and iPhone apply the
  // same rule in `reconcileModelCapabilities`.
  if (!familySupportsExtend(model.family)) {
    next.extendVideo = null;
    next.extendVideoPath = "";
    next.extendOverlapFrames = null;
  }
  if (model.family !== "ltx2" && model.family !== "ltx-2") {
    // Settings knobs clear with the LTX-2 suite; staged media (audio, source
    // video, keyframes) is retained — `toRequest`'s family spread keeps it
    // off the wire, and a return to LTX-2 finds it where the user left it.
    next.pipeline = null;
    next.icLoraControl = null;
    next.retakeRange = null;
    next.spatialUpscale = null;
    next.temporalUpscale = null;
    next.guidanceOverrides = emptyGuidanceOverrides();
    next.cameraControl = null;
  }
  // ── Source-media bridge + retention ─────────────────────────────────────
  // Staged media survives capability-losing switches (`toRequest`'s
  // capability gates keep it off the wire); the layout *moves* between the
  // source authorities (attachment well ↔ H3 boundaries) so the visible
  // well keeps the image across e.g. H3 ↔ LTX-2 switches.
  const prevMode = generationCapabilitiesForFamily(
    selectedFamily(current),
    current.model,
    null,
    null,
    current.sourceImageCapability,
  ).sourceImageMode;
  const enteringH3 =
    capabilities.sourceImageMode === "h3-boundaries" &&
    prevMode !== "h3-boundaries";
  const leavingH3 =
    prevMode === "h3-boundaries" &&
    capabilities.sourceImageMode !== "h3-boundaries";
  if (enteringH3) {
    const authoring = next.h3Authoring ?? emptyMinimaxH3AuthoringState();
    if (!authoring.firstFrame) {
      const staged = current.imageAttachments[0];
      const boundary = minimaxH3BoundaryFromStagedImage(
        staged
          ? {
              base64: staged.base64,
              filename: staged.filename,
              width: staged.width,
              height: staged.height,
              mime: staged.mime,
              draftId: staged.draftId,
            }
          : null,
      );
      if (boundary) {
        authoring.firstFrame = boundary;
        next.imageAttachments = [];
      }
    }
    // A first-frame-only checkpoint (`required`) rejects a lastFrame
    // outright (minimaxH3AuthoringError); the closing frame stays parked.
    if (
      !authoring.lastFrame &&
      current.endFrame?.base64 &&
      !capabilities.requiresSourceImage
    ) {
      const closing = minimaxH3BoundaryFromStagedImage({
        base64: current.endFrame.base64,
        filename: current.endFrame.filename,
        width: current.endFrame.width,
        height: current.endFrame.height,
        mime: current.endFrame.mime,
        draftId: current.endFrame.draftId,
      });
      if (closing) {
        authoring.lastFrame = closing;
        next.endFrame = null;
      }
    }
    next.h3Authoring = authoring;
  } else if (leavingH3 && next.h3Authoring) {
    // Promote authored boundaries back into the attachment layout. Bytes-less
    // reattach descriptors stay parked in h3Authoring for a later H3 return.
    const promoted = stagedImageFromMinimaxH3Boundary(
      next.h3Authoring.firstFrame,
    );
    if (promoted && next.imageAttachments.length === 0) {
      next.imageAttachments = [
        {
          kind: "upload",
          filename: promoted.filename,
          base64: promoted.base64,
          width: promoted.width ?? null,
          height: promoted.height ?? null,
          mime: promoted.mime ?? null,
          ...(promoted.draftId ? { draftId: promoted.draftId } : {}),
        },
      ];
      next.h3Authoring.firstFrame = null;
    }
    const closing = stagedImageFromMinimaxH3Boundary(
      next.h3Authoring.lastFrame,
    );
    if (closing && capabilities.supportsEndFrame && !next.endFrame) {
      next.endFrame = {
        kind: "upload",
        filename: closing.filename,
        base64: closing.base64,
        width: closing.width ?? null,
        height: closing.height ?? null,
        mime: closing.mime ?? null,
        ...(closing.draftId ? { draftId: closing.draftId } : {}),
      };
      next.h3Authoring.lastFrame = null;
    }
  }
  if (capabilities.sourceImageMode !== "single") {
    if (
      capabilities.sourceImageMode !== "references" &&
      capabilities.sourceImageMode !== "h3-boundaries"
    ) {
      next.batchSize = 1;
    }
    if (capabilities.forcesBatchSizeOne) next.batchSize = 1;
  } else if (next.imageAttachments.length > 1) {
    next.imageAttachments = next.imageAttachments.slice(0, 1);
  }
  if (next.cameraControl) {
    const cameraPath = cameraMotionLoraPath(next.cameraControl);
    const cameraRows = current.loras.filter(
      (lora) =>
        lora.path === cameraPath || lora.path.startsWith("camera-control:"),
    );
    next.loras = syncCameraMotionLora(
      cameraRows,
      current.cameraControl,
      next.cameraControl,
      (path, scale) => ({ path, scale, trainedWords: [] }),
    );
  }
  // Fixed recipe controls are not user choices: normalize the hidden form
  // value to what the disabled control displays, or the submit validator can
  // strand Generate behind an error the user cannot correct (shared with
  // desktop's `reconcileModelCapabilities`).
  Object.assign(
    next,
    fixedRecipeControlOverrides(
      effectiveGenerationRecipe(model, next.pipeline ?? null),
    ),
  );
  return next;
}

/**
 * "Reset settings to defaults" (Create rail): every generation knob returns to
 * the factory default, then to the selected model's own defaults on top.
 *
 * What survives is what the reset is *for* — the user is re-tuning a print they
 * are still composing: the prompt, the style preset, and the selected
 * model/family. Batch is a general setting and returns to one; prepared work is
 * retained and becomes explicitly stale. Everything else — shape, resolution,
 * detail, prompt strength, seed, and every advanced field including source
 * media, LoRAs and the video suite — goes back to defaults.
 */
export function settingsResetPatch(
  current: GenerateFormState,
  model: ModelInfoExtended | null,
): GenerateFormState {
  const base: GenerateFormState = {
    ...defaultForm(),
    prompt: current.prompt,
    stylePreset: current.stylePreset,
    model: current.model,
    modelFamily: current.modelFamily,
    batchSize: 1,
  };
  // With a resolved catalog row the model's own defaults (and its capability
  // gates) win over the generic ones; without it the generic defaults stand and
  // the model name is left alone.
  return model ? modelDefaultsPatch(base, model) : base;
}

/**
 * Normalize a form snapshot saved before `negativePromptDefault` existed —
 * legacy generation templates (#787 round 3) — before it replaces the live
 * form wholesale. Such snapshots carry no tri-state authority: their empty
 * `negativePrompt` means "untouched", never the explicit `""` opt-out, and a
 * template loaded AFTER the installed-models watcher settled would otherwise
 * sit with an empty default forever (the watcher's deps never change again).
 * The snapshot's own model resolves the default (live inventory row first,
 * then the family constant via the snapshot's stored family); a snapshot that
 * already carries the key is post-#787 authority and passes through
 * untouched. Mirrors desktop's `normalizeLegacyNegativeSnapshot`.
 */
export function normalizeLegacyNegativeFormState(
  state: GenerateFormState,
  models: ModelInfoExtended[] = [],
): GenerateFormState {
  if (typeof state.negativePromptDefault === "string") {
    state.negativeExplicitClear ??= false;
    return state;
  }
  const row = models.find((m) => m.name === state.model) ?? null;
  const nextDefault = effectiveNegativeDefault(
    row,
    row?.family ?? selectedFamily(state),
  );
  state.negativePromptDefault = nextDefault;
  state.negativeExplicitClear = false;
  if (state.negativePrompt.trim() === "") {
    // Legacy omission is the untouched state: show the default so the wire
    // stays absent, exactly what the pre-#787 template produced.
    state.negativePrompt = nextDefault;
  }
  return state;
}

export interface ApplyMetadataOptions {
  models?: ModelInfoExtended[];
  format?: OutputFormat | null;
}

/** Recreate-safe metadata application shared by Gallery Recreate and future
 * template/import paths. It restores serialized generation knobs, uses static
 * seed semantics for exact-ish recreation, and clears stale binary media
 * because output metadata does not contain source/mask/control bytes. */
export function applyMetadataToForm(
  current: GenerateFormState,
  metadata: OutputMetadata,
  options: ApplyMetadataOptions = {},
): GenerateFormState {
  const model = options.models?.find((m) => m.name === metadata.model);
  const next = model
    ? modelDefaultsPatch(current, model)
    : {
        ...cloneFormState(current),
        model: metadata.model,
        modelFamily: "",
        negativePromptDefault: "",
      };
  const loras =
    metadata.loras?.map<LoraSelection>((l) => ({
      path: l.path,
      scale: l.scale,
    })) ??
    (metadata.lora
      ? [
          {
            path: metadata.lora,
            scale: metadata.lora_scale ?? 1.0,
          },
        ]
      : []);
  const outputFormat = metadata.output_format ?? options.format ?? null;
  // Saved provenance carries the identity photo's NAME and digest, never its
  // bytes, so reuse restores a bytes-less reattach descriptor and the page
  // looks the payload back up in the local stash by `id_image_sha256`. An
  // empty payload is refused by `identityRequestFields`, so the descriptor
  // can never smuggle a blank `id_image` onto the wire.
  const identity = identityReuse(metadata);
  const camera = normalizeCameraMotionLoraState(
    loras.slice(0, MAX_LORA_STACK),
    null,
    (path, scale) => ({ path, scale }),
    MAX_LORA_STACK,
  );
  return {
    ...next,
    prompt: metadata.prompt ?? "",
    // Creation-time title; the Library overrides it with the row's editable
    // title before handing off, so the composer shows what the print is
    // called now.
    title: metadata.title ?? null,
    originalPrompt: metadata.original_prompt ?? null,
    // Saved metadata already carries the fully-composed prompt (style extras
    // included at generation time); re-applying a preset would double-append.
    stylePreset: null,
    // Absence predates truthful recording: on a defaulted model it means the
    // default conditioned the render, so restore shows it rather than
    // silently flipping the reuse into an explicit empty-uncond opt-out.
    negativePrompt: restoredNegativePrompt(
      metadata.negative_prompt,
      next.negativePromptDefault ?? "",
    ),
    // A recorded `""` is the explicit opt-out. When this restore ran before
    // the model rows resolved, the empty control is otherwise identical to
    // "untouched" — the marker carries the print's authority until the
    // default is known (#787 round 3).
    negativeExplicitClear: restoredNegativeExplicitClear(
      metadata.negative_prompt,
    ),
    width: metadata.generation_width || metadata.width || next.width,
    height: metadata.generation_height || metadata.height || next.height,
    steps: metadata.steps || next.steps,
    guidance: metadata.guidance ?? next.guidance,
    seedMode: metadata.seed == null ? "random" : "static",
    seed: metadata.seed ?? null,
    scheduler: metadata.scheduler ?? null,
    cfgPlus: metadata.cfg_plus ?? false,
    strength:
      metadata.strength !== undefined && metadata.strength !== null
        ? metadata.strength
        : next.strength,
    // Additive crop provenance: restore only a policy that parses exactly —
    // corrupt or future-shaped values must never poison the live form.
    sourceFitPolicy:
      parseSourceFitPolicy(metadata.source_fit) ??
      next.sourceFitPolicy ??
      current.sourceFitPolicy,
    loras: camera.loras,
    cameraControl: camera.cameraControl,
    controlModel: metadata.control_model ?? "",
    controlScale: metadata.control_scale ?? next.controlScale,
    upscaleModel: metadata.upscale_model ?? "",
    gifPreview: metadata.gif_preview ?? false,
    enableAudio: metadata.enable_audio ?? next.enableAudio,
    audioFilePath: metadata.audio_file_path ?? "",
    sourceVideoPath: metadata.source_video_path ?? "",
    extendVideoPath: metadata.extend_video_path ?? "",
    extendOverlapFrames: metadata.extend_overlap_frames ?? null,
    pipeline: pipelineForSettingsReuse(metadata),
    icLoraControl: metadata.ic_lora_control ?? null,
    retakeRange: metadata.retake_range ?? null,
    spatialUpscale: metadata.spatial_upscale ?? null,
    temporalUpscale: metadata.temporal_upscale ?? null,
    guidanceOverrides: guidanceOverridesFromWire(metadata.guidance_overrides),
    wanRecipe: wanRecipeFromWire(metadata),
    frames: metadata.frames ?? null,
    predictDuration: metadata.duration_prediction_requested === true,
    fps: metadata.fps ?? null,
    outputFormat: outputFormat ?? next.outputFormat,
    imageAttachments: [],
    // Saved keyframe metadata is name + sha256 with no payload, so a
    // first/last-frame render's closing still cannot be rebuilt from it. The
    // page says so out loud rather than leaving Generate looking ready.
    endFrame: null,
    identityImage: identity
      ? {
          kind: "upload",
          filename: identity.name || "identity photo",
          base64: "",
        }
      : null,
    identityWeight: identity ? identity.weight : null,
    identityStartStep: identity ? identity.startStep : null,
    maskImage: null,
    controlImage: null,
    audioFile: null,
    sourceVideo: null,
    extendVideo: null,
    keyframes: [],
    h3Authoring: {
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
              metadata.frames,
              metadata.keyframes,
            )
          : null,
      references: minimaxH3ReferenceDraftsFromMetadata(metadata.references),
    },
  };
}

function stripMediaBytes<T extends { base64: string }>(media: T): T {
  const { base64: _base64, ...rest } = media;
  void _base64;
  return cloneFormState(rest) as T;
}

function stripH3AuthoringMedia(
  state: MinimaxH3AuthoringState | null | undefined,
): MinimaxH3AuthoringState {
  const boundary = (
    value: MinimaxH3BoundaryImage | null,
  ): MinimaxH3BoundaryImage | null => {
    if (!value) return null;
    const { data: _data, ...descriptor } = value;
    void _data;
    return cloneFormState({ ...descriptor, data: "" });
  };
  return {
    firstFrame: boundary(state?.firstFrame ?? null),
    lastFrame: boundary(state?.lastFrame ?? null),
    references: (state?.references ?? []).map((draft) => {
      const { reference, ...draftMetadata } = draft;
      return {
        ...cloneFormState(draftMetadata),
        reference: cloneFormState({
          ...reference,
          media: { authority: "descriptor" },
        }) as typeof reference,
      };
    }),
  };
}

function ensureDraftIds(state: GenerateFormState) {
  const ensure = (media: { base64: string; draftId?: string } | null): void => {
    if (media?.base64 && !media.draftId) media.draftId = newDraftId();
  };
  state.imageAttachments.forEach(ensure);
  ensure(state.endFrame);
  ensure(state.identityImage ?? null);
  ensure(state.maskImage);
  ensure(state.controlImage);
  ensure(state.audioFile);
  ensure(state.sourceVideo);
  ensure(state.extendVideo);
  state.keyframes.forEach((keyframe) => ensure(keyframe.image));
  const h3 = state.h3Authoring;
  if (!h3) return;
  for (const boundary of [h3.firstFrame, h3.lastFrame]) {
    if (boundary?.data && !boundary.draftId) boundary.draftId = newDraftId();
  }
  for (const draft of h3.references) {
    const media = draft.reference.media;
    if (media.authority === "inline" && media.data && !draft.draftId) {
      draft.draftId = newDraftId();
    }
  }
}

function h3MediaFromState(state: GenerateFormState): DraftMediaRecord[] {
  const h3 = state.h3Authoring;
  if (!h3) return [];
  const records: DraftMediaRecord[] = [];
  for (const boundary of [h3.firstFrame, h3.lastFrame]) {
    if (boundary?.data) {
      records.push({ draftId: boundary.draftId, base64: boundary.data });
    }
  }
  for (const draft of h3.references) {
    const media = draft.reference.media;
    if (media.authority === "inline" && media.data) {
      records.push({ draftId: draft.draftId, base64: media.data });
    }
  }
  return records;
}

function mediaFromState(state: GenerateFormState): DraftMediaRecord[] {
  const ordinary = [
    ...state.imageAttachments,
    state.endFrame,
    state.identityImage ?? null,
    state.maskImage,
    state.controlImage,
    state.audioFile,
    state.sourceVideo,
    state.extendVideo,
    ...state.keyframes.map((k) => k.image),
  ].filter((m): m is NonNullable<typeof m> => Boolean(m?.base64));
  return [...ordinary, ...h3MediaFromState(state)];
}

let pendingDraftWrites: Promise<unknown>[] = [];
let pendingHydration: Promise<unknown> | null = null;
let pendingPersistence: Promise<void> | null = null;
let persistenceEpoch = 0;

function persistDraftMedia(media: readonly DraftMediaRecord[]) {
  const write = putDurableMediaBatch(media);
  pendingDraftWrites = [write];
  return write;
}

async function hydrateDraftMedia(state: GenerateFormState) {
  const hydrate = async <
    T extends { draftId?: string; base64?: string } | null,
  >(
    media: T,
  ): Promise<T> => {
    if (!media?.draftId || media.base64) return media;
    const stored = await getDraftMedia(media.draftId);
    return stored ? ({ ...media, ...stored } as T) : media;
  };
  const attachmentIds = state.imageAttachments.map((m) => m.draftId ?? "");
  const attachments = await Promise.all(
    state.imageAttachments.map((m) => hydrate(m)),
  );
  if (
    state.imageAttachments.length === attachmentIds.length &&
    state.imageAttachments.every(
      (m, i) => (m.draftId ?? "") === attachmentIds[i],
    )
  ) {
    state.imageAttachments = attachments;
  }

  const endFrameId = state.endFrame?.draftId ?? "";
  const endFrame = await hydrate(state.endFrame);
  if ((state.endFrame?.draftId ?? "") === endFrameId) state.endFrame = endFrame;

  const identityId = state.identityImage?.draftId ?? "";
  const identity = await hydrate(state.identityImage ?? null);
  if ((state.identityImage?.draftId ?? "") === identityId) {
    state.identityImage = identity;
  }

  const maskId = state.maskImage?.draftId ?? "";
  const mask = await hydrate(state.maskImage);
  if ((state.maskImage?.draftId ?? "") === maskId) state.maskImage = mask;

  const controlId = state.controlImage?.draftId ?? "";
  const control = await hydrate(state.controlImage);
  if ((state.controlImage?.draftId ?? "") === controlId)
    state.controlImage = control;

  const audioId = state.audioFile?.draftId ?? "";
  const audio = await hydrate(state.audioFile);
  if ((state.audioFile?.draftId ?? "") === audioId) state.audioFile = audio;

  const sourceVideoId = state.sourceVideo?.draftId ?? "";
  const sourceVideo = await hydrate(state.sourceVideo);
  if ((state.sourceVideo?.draftId ?? "") === sourceVideoId) {
    state.sourceVideo = sourceVideo;
  }

  const keyframeIds = state.keyframes.map(
    (k) => `${k.frame}:${k.image.draftId ?? ""}`,
  );
  const keyframes = await Promise.all(
    state.keyframes.map(async (k) => ({ ...k, image: await hydrate(k.image) })),
  );
  if (
    state.keyframes.length === keyframeIds.length &&
    state.keyframes.every(
      (k, i) => `${k.frame}:${k.image.draftId ?? ""}` === keyframeIds[i],
    )
  ) {
    state.keyframes = keyframes;
  }

  const h3 = state.h3Authoring;
  if (!h3) return;
  const hydrateBoundary = async (
    boundary: MinimaxH3BoundaryImage | null,
  ): Promise<MinimaxH3BoundaryImage | null> => {
    if (!boundary?.draftId || boundary.data) return boundary;
    const stored = await getDraftMedia(boundary.draftId);
    return stored?.base64 ? { ...boundary, data: stored.base64 } : boundary;
  };
  const hydrateReference = async (
    draft: MinimaxH3ReferenceDraft,
  ): Promise<MinimaxH3ReferenceDraft> => {
    if (
      !draft.draftId ||
      (draft.reference.media.authority === "inline" &&
        draft.reference.media.data)
    ) {
      return draft;
    }
    const stored = await getDraftMedia(draft.draftId);
    if (!stored?.base64) return draft;
    return {
      ...draft,
      reference: {
        ...draft.reference,
        media: { authority: "inline", data: stored.base64 },
      },
    };
  };

  const firstSnapshot = h3.firstFrame;
  const lastSnapshot = h3.lastFrame;
  const referencesSnapshot = h3.references;
  const [firstFrame, lastFrame, references] = await Promise.all([
    hydrateBoundary(firstSnapshot),
    hydrateBoundary(lastSnapshot),
    Promise.all(referencesSnapshot.map(hydrateReference)),
  ]);
  const current = state.h3Authoring;
  if (!current) return;
  if (current.firstFrame === firstSnapshot) current.firstFrame = firstFrame;
  if (current.lastFrame === lastSnapshot) current.lastFrame = lastFrame;
  if (current.references === referencesSnapshot)
    current.references = references;
}

function load(): GenerateFormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultForm();
    const parsed = JSON.parse(raw) as Partial<GenerateFormState>;
    if (parsed.version !== FORM_VERSION) {
      return defaultForm();
    }
    const restored = {
      ...defaultForm(),
      ...sanitizePersistedForm({
        ...defaultForm(),
        ...parsed,
      }),
    };
    const camera = normalizeCameraMotionLoraState(
      restored.loras,
      restored.cameraControl,
      (path, scale) => ({ path, scale, trainedWords: [] }),
      MAX_LORA_STACK,
    );
    return { ...restored, ...camera };
  } catch {
    return defaultForm();
  }
}

function persist(state: GenerateFormState) {
  const epoch = ++persistenceEpoch;
  let serialized: string;
  try {
    ensureDraftIds(state);
    // Drop base64 bytes from localStorage — they blow past the quota quickly.
    // Draft IDs reconnect this descriptor state to IndexedDB during hydration.
    serialized = JSON.stringify(sanitizePersistedForm(state));
  } catch {
    return;
  }
  const write = persistDraftMedia(mediaFromState(state));
  pendingPersistence = (async () => {
    // The descriptor is valid only after every referenced byte payload commits
    // in the same transaction. A later edit supersedes this snapshot; a quota,
    // private-mode, or transaction failure leaves the prior valid draft alone
    // and the next form change retries from current in-memory state.
    if (!(await write) || epoch !== persistenceEpoch) return;
    try {
      localStorage.setItem(STORAGE_KEY, serialized);
    } catch {
      // Bytes are durable but no descriptor points at them. The next change
      // safely retries; never publish a descriptor before its media exists.
    }
  })();
}

export interface UseGenerateForm {
  state: Ref<GenerateFormState>;
  reset: () => void;
  /** Restore every generation setting to the selected model's defaults while
   * keeping the prompt, style, and model (`settingsResetPatch`). */
  resetSettings: (model: ModelInfoExtended | null) => void;
  applyModelDefaults: (model: ModelInfoExtended) => void;
  /** Refresh capability-only state for the currently selected model without
   * reapplying model defaults or discarding authored controls. */
  reconcileModelCapabilities: (model: ModelInfoExtended) => void;
  /**
   * Reconcile the restored form's negative-default state against the resolved
   * catalog row for the *currently selected* model once the inventory lands
   * (#787 round 2). A persisted v3 snapshot that predates
   * `negativePromptDefault` restores with an empty default, and the
   * installed-model watcher early-returns because the saved model is still
   * valid — without this, upgraded users never see the prefill and cannot
   * express the "" opt-out until they switch models. Runs the same
   * `negativePromptOnDefaultChange` + default update as the model-switch
   * path, so it is idempotent and never clobbers typed text or an explicit
   * clear recorded against a known default.
   */
  reconcileNegativeDefault: (model: ModelInfoExtended) => void;
  /** Replace the entire LoRA stack. Pass `[]` to clear. */
  setLoras: (loras: LoraSelection[]) => void;
  /** Append a LoRA to the stack, capped by `MAX_LORA_STACK`. No-op if
   * the cap is already reached. */
  addLora: (lora: LoraSelection) => void;
  /** Update a single LoRA's scale in place by index. */
  updateLoraScale: (index: number, scale: number) => void;
  /** Drop the LoRA at `index`. Out-of-range indices are no-ops. */
  removeLora: (index: number) => void;
  /** Append `phrase` to the active prompt with sensible whitespace. */
  appendPromptPhrase: (phrase: string) => void;
  /**
   * Build the wire request. Pass the resolved catalog row when available so
   * checkpoint-specific capabilities can override stale persisted form state.
   */
  toRequest: (model?: ModelInfoExtended | null) => GenerateRequestWire;
  isVideoFamily: (family: string) => boolean;
  supportsNegativePrompt: (family: string) => boolean;
  supportsScheduler: (family: string) => boolean;
  /** Mirrors `mold-tui/src/model_info.rs::capabilities_for_family.supports_lora`
   * and the server-side `require_lora_capable_family` gate. Drives the
   * conditional render of `<LoraPicker>` in the SettingsModal. */
  supportsLora: (family: string) => boolean;
}

let singleton: UseGenerateForm | null = null;
let stopPersistWatch: WatchStopHandle | null = null;
let persistTimer: ReturnType<typeof setTimeout> | null = null;

export function useGenerateForm(): UseGenerateForm {
  if (singleton) return singleton;
  const state = ref<GenerateFormState>(load());
  pendingHydration = hydrateDraftMedia(state.value);

  stopPersistWatch = watch(
    state,
    (v) => {
      if (persistTimer) clearTimeout(persistTimer);
      persistTimer = setTimeout(() => persist(v), 300);
    },
    { deep: true },
  );

  singleton = {
    state,
    reset: () => {
      state.value = defaultForm();
    },
    resetSettings: (model) => {
      state.value = settingsResetPatch(state.value, model);
    },
    setLoras: (loras) => {
      state.value.loras = loras.slice(0, MAX_LORA_STACK);
    },
    addLora: (lora) => {
      if (state.value.loras.length >= MAX_LORA_STACK) return;
      state.value.loras = [...state.value.loras, lora];
    },
    updateLoraScale: (index, scale) => {
      const next = state.value.loras.slice();
      const row = next[index];
      if (!row) return;
      next[index] = { ...row, scale };
      state.value.loras = next;
    },
    removeLora: (index) => {
      if (index < 0 || index >= state.value.loras.length) return;
      const next = state.value.loras.slice();
      next.splice(index, 1);
      state.value.loras = next;
    },
    appendPromptPhrase: (phrase: string) => {
      const trimmed = phrase.trim();
      if (!trimmed) return;
      const current = state.value.prompt;
      // Reuse comma+space if the prompt is non-empty so trigger phrases
      // chain naturally ("a cat, cinematic, dramatic lighting") rather
      // than concatenating into a wall of text. Avoid the comma when the
      // prompt is empty to keep simple cases clean.
      state.value.prompt = current.trim()
        ? `${current.trimEnd()}, ${trimmed}`
        : trimmed;
    },
    applyModelDefaults: (m) => {
      // LoRA support is family-specific. Clear the stack on every model
      // change — even FLUX→FLUX swaps because the LoRA might not target
      // the new variant's tensor layout.
      state.value = modelDefaultsPatch(state.value, m);
    },
    reconcileModelCapabilities: (m) => {
      const s = state.value;
      if (m.name !== s.model) return;
      if (
        m.supports_duration_prediction !== true ||
        m.runtime_ready === false
      ) {
        s.predictDuration = false;
      }
    },
    reconcileNegativeDefault: (m) => {
      const s = state.value;
      // Only the selected model's row is authority for the visible field.
      if (m.name !== s.model) return;
      const nextDefault = effectiveNegativeDefault(m, m.family);
      const nextPrompt = negativePromptOnDefaultChange(
        s.negativePrompt,
        s.negativePromptDefault ?? "",
        nextDefault,
        // A reuse restored before this row landed may carry the explicit
        // `""` opt-out — indistinguishable from untouched without the
        // marker, which keeps the clear instead of prefilling (#787 r3).
        s.negativeExplicitClear ?? false,
      );
      if (nextPrompt !== s.negativePrompt) s.negativePrompt = nextPrompt;
      if ((s.negativePromptDefault ?? "") !== nextDefault) {
        s.negativePromptDefault = nextDefault;
      }
    },
    toRequest: (model = null) => {
      const s = state.value;
      // The wire format strips per-row metadata (trigger phrases) — only
      // path + scale travel to the server. We send `loras` (plural) so
      // multi-LoRA stacks reach the FLUX engine; older single-LoRA
      // clients still set `lora`, which the server coalesces.
      let loras = s.loras.map((l) => ({ path: l.path, scale: l.scale }));
      const capabilities = generationCapabilitiesForFamily(
        selectedFamily(s),
        s.model,
        s.pipeline,
        s.guidanceCapabilities,
        // The resolved catalog row is the final authority at submit time, the
        // same way it is for `enable_audio`; the snapshot covers a form
        // restored before the inventory landed.
        model?.source_image ?? s.sourceImageCapability,
      );
      const attachmentMode = capabilities.sourceImageMode !== "single";
      const attachments = s.imageAttachments ?? [];
      // Wan's first/last-frame render (#779) rides the existing `keyframes`
      // contract: both stills travel there and `source_image` stays home —
      // the engine refuses a request carrying both, and admission counts
      // keyframes as source presence for an I2V-required checkpoint. The
      // closing index is derived from the frame count THIS request carries,
      // so changing the clip length after attaching the end frame can never
      // ship a stale index. A lone source image is an ordinary
      // image-to-video request and sends none.
      const firstLastFrames =
        capabilities.supportsEndFrame && !attachmentMode
          ? firstLastFrameKeyframes(
              attachments[0]?.base64
                ? {
                    base64: attachments[0].base64,
                    filename: attachments[0].filename,
                  }
                : null,
              s.endFrame?.base64
                ? { base64: s.endFrame.base64, filename: s.endFrame.filename }
                : null,
              capabilities.supportsVideo ? s.frames : null,
            )
          : null;
      const requestForcesBatchSizeOne =
        capabilities.forcesBatchSizeOne ||
        (capabilities.sourceImageMode === "references" &&
          attachments.length > 0);
      const controlModel = capabilities.supportsControlNet
        ? s.controlModel.trim()
        : "";
      const upscaleModel = s.upscaleModel.trim();
      const audioPath = s.audioFilePath.trim();
      const sourceVideoPath = s.sourceVideoPath.trim();
      const extendVideoPath = s.extendVideoPath.trim();
      const family = selectedFamily(s);
      const ltx2 = family === "ltx2" || family === "ltx-2";
      const cameraControl = s.cameraControl?.trim();
      if (ltx2 && cameraControl) {
        loras = syncCameraMotionLora(
          loras,
          cameraControl,
          cameraControl,
          (path, scale) => ({ path, scale }),
          MAX_LORA_STACK,
        );
      }
      // The style preset is baked into the OUTGOING request — prompt template
      // plus the preset's curated negative, merged after the user's own
      // fragments and only for families that accept a negative prompt. Neither
      // field on screen is rewritten.
      const styled = composeStyle(s.prompt, s.stylePreset ?? "", {
        supportsNegativePrompt: capabilities.supportsNegativePrompt,
        negative: s.negativePrompt,
      });
      // Stripped at the end: an audio-only pipeline renders no frames, so
      // every conditioning input and upscaler still sitting in the form is
      // something the server refuses. Doing it here rather than on the
      // pipeline transition keeps a user's source media intact if they switch
      // to `t2a` and back.
      const title = validatePrintTitle(s.title ?? "");
      const request: GenerateRequestWire = {
        prompt: styled.prompt,
        // The Create title rides every request this form builds (one-shot,
        // batch siblings, prepared variations). Absent when empty or
        // invalid so older servers and untitled prints behave as before.
        ...(title.ok && title.value ? { title: title.value } : {}),
        ...(s.originalPrompt?.trim() &&
        s.originalPrompt.trim() !== styled.prompt
          ? { original_prompt: s.originalPrompt.trim() }
          : {}),
        // Tri-state (#787): text equal to the advertised default stays
        // absent (older servers behave identically), a cleared defaulted
        // field ships the explicit "" opt-out, typed/styled text travels
        // verbatim. `null` and omission both read as absent server-side.
        negative_prompt: capabilities.supportsNegativePrompt
          ? (negativePromptWireValue(
              styled.negative ?? "",
              s.negativePromptDefault ?? "",
              s.negativeExplicitClear ?? false,
            ) ?? null)
          : null,
        model: s.model,
        width: s.width,
        height: s.height,
        steps: s.steps,
        guidance: effectiveGenerationGuidance(capabilities, s.guidance),
        seed: s.seedMode === "random" ? null : s.seed,
        batch_size: requestForcesBatchSizeOne ? 1 : s.batchSize,
        output_format: s.outputFormat,
        cfg_plus: capabilities.supportsCfgPlus && s.cfgPlus ? true : undefined,
        // "default" is the picker's omit-sentinel, not a wire variant; a
        // stored sentinel must never reach serde (codex review).
        scheduler:
          capabilities.supportsScheduler && s.scheduler !== "default"
            ? s.scheduler
            : undefined,
        ...(attachmentMode
          ? {
              edit_images: attachments.map((image) => image.base64),
            }
          : {
              // A first/last-frame render ships BOTH stills as `keyframes`
              // and no `source_image` — the engine refuses a request
              // carrying both, and admission counts keyframes as source
              // presence for an I2V-required checkpoint. Retained media on
              // a model with no source support stays off the wire entirely.
              source_image:
                firstLastFrames || !capabilities.supportsSourceImage
                  ? null
                  : (attachments[0]?.base64 ?? null),
              source_image_name:
                !firstLastFrames &&
                capabilities.supportsSourceImage &&
                attachments[0]?.base64
                  ? attachments[0].filename
                  : undefined,
              // Wan pins the first frame exactly; it never reads strength.
              strength:
                !firstLastFrames &&
                capabilities.supportsSourceImage &&
                capabilities.supportsStrength &&
                attachments[0]?.base64
                  ? s.strength
                  : undefined,
              // A repaint mask is meaningless without a source image, so it
              // shares the source gate on top of its own capability.
              mask_image:
                firstLastFrames ||
                !capabilities.supportsMask ||
                !capabilities.supportsSourceImage
                  ? undefined
                  : (s.maskImage?.base64 ?? undefined),
              control_image: capabilities.supportsControlNet
                ? (s.controlImage?.base64 ?? undefined)
                : undefined,
              control_model:
                s.controlImage && controlModel ? controlModel : undefined,
              control_scale:
                s.controlImage && controlModel ? s.controlScale : undefined,
              ...(firstLastFrames ? { keyframes: firstLastFrames } : {}),
            }),
        expand: s.expand.enabled || undefined,
        frames:
          capabilities.supportsVideo &&
          !(
            s.predictDuration &&
            model?.supports_duration_prediction === true &&
            model.runtime_ready !== false
          )
            ? s.frames
            : undefined,
        fps: capabilities.supportsVideo ? s.fps : undefined,
        upscale_model: upscaleModel || undefined,
        gif_preview:
          capabilities.supportsVideo && s.gifPreview ? true : undefined,
        placement: s.placement ?? undefined,
        loras: capabilities.supportsLora && loras.length ? loras : undefined,
        // A persisted draft can retain `true` from a previously-selected AV
        // checkpoint. The resolved model row is the final authority at submit
        // time: video-only LTX-2 exports must never reach the server with audio
        // enabled, even if initial model hydration did not reapply defaults.
        enable_audio:
          model?.supports_audio === false
            ? false
            : (s.enableAudio ?? undefined),
        ...(ltx2
          ? {
              audio_file: s.audioFile?.base64 ?? undefined,
              audio_file_path: s.audioFile ? undefined : audioPath || undefined,
              source_video: s.sourceVideo?.base64 ?? undefined,
              source_video_path: s.sourceVideo
                ? undefined
                : sourceVideoPath || undefined,
              keyframes: s.keyframes.length
                ? s.keyframes.map((k) => ({
                    frame: k.frame,
                    image: k.image.base64,
                    name: k.image.filename,
                  }))
                : undefined,
              pipeline: s.icLoraControl
                ? pipelineForControlId(s.icLoraControl)
                : (s.pipeline ?? undefined),
              ic_lora_control: s.icLoraControl ?? undefined,
              retake_range: s.retakeRange ?? undefined,
              spatial_upscale: s.spatialUpscale ?? undefined,
              temporal_upscale: s.temporalUpscale ?? undefined,
              guidance_overrides: guidanceOverridesToWire(s.guidanceOverrides),
            }
          : {}),
        // Continuation is per family, not part of the LTX-2 suite (#783):
        // wan continues by seeding the render with the source clip's final
        // frame, so its fields have to survive outside the `ltx2` spread or
        // an offered wan continuation goes out as a plain text-to-video job.
        ...(familySupportsExtend(family)
          ? {
              extend_video: s.extendVideo?.base64 ?? undefined,
              extend_video_path: s.extendVideo
                ? undefined
                : extendVideoPath || undefined,
              // Only travels with a video to continue — the server rejects a
              // bare overlap — but when it does travel it carries the value
              // the control is showing, resolved by the one shared authority.
              // Leaving it absent handed the host its own default, which for
              // wan is the family-wide 17 `extend_inner` refuses.
              extend_overlap_frames:
                s.extendVideo || extendVideoPath
                  ? resolveExtendOverlapFrames(s.extendOverlapFrames, {
                      family: model?.family ?? family,
                      extend_default_overlap_frames:
                        model?.extend_default_overlap_frames,
                    })
                  : undefined,
            }
          : {}),
        // Spread rather than assigned: an untouched control contributes no
        // key at all, which is what keeps the resolved tier's own shift and
        // distill strengths in place.
        ...wanRecipeToWire(s.wanRecipe, capabilities.wanRecipe),
        // Face identity (#1224). The resolved catalog row is the final
        // authority at submit time exactly as it is for `enable_audio`; the
        // snapshot covers a form restored before the inventory landed. The
        // photo travels untouched — it is never fitted against the canvas.
        ...identityRequestFields({
          supported: model
            ? supportsIdentity(
                effectiveGenerationRecipe(model, s.pipeline),
                model,
              )
            : (s.identitySupported ?? false),
          image: s.identityImage?.base64
            ? {
                base64: s.identityImage.base64,
                filename: s.identityImage.filename,
              }
            : null,
          weight: s.identityWeight ?? null,
          startStep: s.identityStartStep ?? null,
        }),
      };
      const finalized = stripAudioOnlyIncompatibleFields(
        serializeMinimaxH3Authoring(
          request,
          family,
          s.model,
          s.h3Authoring ?? emptyMinimaxH3AuthoringState(),
        ),
      );
      // Crop provenance rides only when the wire actually carries fitted
      // source media (the server echoes it verbatim into OutputMetadata so
      // Reuse settings and running-job selection restore the crop controls).
      if (
        finalized.source_image ||
        finalized.edit_images?.length ||
        finalized.keyframes?.length
      ) {
        finalized.source_fit = s.sourceFitPolicy ?? defaultSourceFitPolicy();
      }
      return finalized;
    },
    isVideoFamily,
    supportsNegativePrompt,
    supportsScheduler,
    supportsLora,
  };
  return singleton;
}

// Scheduler type is re-exported so callers can type-narrow without importing
// both modules.
export type { Scheduler };
export { isQwenImageEditFamily };

export const __testing__ = {
  resetForTest() {
    persistenceEpoch += 1;
    stopPersistWatch?.();
    stopPersistWatch = null;
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = null;
    singleton = null;
    pendingDraftWrites = [];
    pendingHydration = null;
    pendingPersistence = null;
  },
  async flushDraftWrites() {
    await Promise.allSettled(pendingDraftWrites);
    await pendingPersistence;
  },
  async flushHydration() {
    await pendingHydration;
  },
  clearDraftsForTest() {
    clearMemoryDraftsForTest();
  },
};
