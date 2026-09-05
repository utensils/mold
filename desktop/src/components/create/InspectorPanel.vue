<script setup lang="ts">
import { computed, ref, watch } from "vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import ToggleControl from "../settings/ToggleControl.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import Stepper from "@ui/components/Stepper.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { useLastUsedStylesStore } from "@studio/stores/lastUsedStyles";
import {
  defaultClipFrames,
  modelSupportsSequence,
  sequenceMotionTailFrames,
} from "@studio/lib/sequence";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { GenerateForm } from "../../lib/generateForm";
import {
  buildRequest,
  loraBindingMatchesRoute,
  loraHostBinding,
  resetFormToModelDefaults,
  seedMode,
} from "../../lib/generateForm";
import type { Ltx2CameraControlInfo, Ltx2ControlAdapterInfo } from "../../lib/api/types";
import {
  isCameraMotionPreset,
  parseCameraControlAvailability,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import { apiJsonTo } from "../../lib/api/client";
import { findInstalledModel } from "../../lib/generateModels";
import { normalizeTargetHost } from "../../lib/hosts";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import SourceImageWell from "../generate/SourceImageWell.vue";
import LoraStack from "../generate/LoraStack.vue";
import TemplatesPanel from "../generate/TemplatesPanel.vue";
import StarterList from "./StarterList.vue";
import RecentPrints from "./RecentPrints.vue";
import { INSPECTOR_TABS, type InspectorTab } from "./inspectorTabs";
import type { GenerationTemplate } from "../../lib/generationTemplates";
import IdentityWell from "./IdentityWell.vue";
import { advancedActiveCount } from "../../lib/advancedCount";
import { activeQualityPreset, qualityPresets, type QualityPreset } from "../../lib/qualityPresets";
import { meshDetailLadder } from "../../lib/meshDetailLadder";
import { controlNote, effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import {
  intentForCanvas,
  resolveOutputShape,
  sizeForFamily,
  SOURCE_FAMILY_ID,
  type CanvasIntent,
  type OutputShapeInput,
} from "@studio/lib/outputShape";
import { resolveSourceResolution } from "@studio/lib/sourceResolution";
import {
  meshTargetFacesError,
  profileStepsValidationError,
  resolutionValidationError,
  resolutionValidationWarning,
} from "../../lib/generateValidation";
import { randomSeed } from "../../stores/generation";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
import { useGalleryStore, type MergedPrint } from "../../stores/gallery";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";
import { fileUnderAvailable, matchCollection, type FileUnderState } from "@studio/lib/fileUnder";
import FileUnderGroup from "./FileUnderGroup.vue";
import { dragWidth } from "../../lib/panelResize";
import { useStylePicker } from "../../composables/useStylePicker";
import { modelsForOutputKind } from "../../composables/useCreateOutputKind";
import AdvancedSettings from "./AdvancedSettings.vue";
import SequenceAdvancedSettings from "./SequenceAdvancedSettings.vue";
import SequenceOpeningImageWell from "./SequenceOpeningImageWell.vue";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    tab?: InspectorTab;
    /** Recent prints, newest first, for the Recent tab. */
    recent?: MergedPrint[];
    /** Seed of the most recent finished print — powers "lock last seed". */
    lastSeed?: number | null;
    /** Per-model chain caps for the selected model, when Create has them —
     * sizes new clips' default frames on the Output switch. */
    chainLimits?: ChainLimits | null;
    /** Why the canvas holds its current size — the shape resolver's authority. */
    canvasIntent?: CanvasIntent;
  }>(),
  {
    tab: "settings",
    recent: () => [],
    lastSeed: null,
    chainLimits: null,
    canvasIntent: "model-default",
  },
);

const emit = defineEmits<{
  "append-word": [word: string];
  "canvas-intent": [intent: CanvasIntent];
  "reset-settings": [];
  "update:tab": [tab: InspectorTab];
  "load-template": [template: GenerationTemplate];
  /** Recent tab: restore a past print's whole recipe, exactly as the Lightbox
   * does — the door that opens this tab says "Use these settings again". */
  "reuse-print": [print: MergedPrint];
}>();
const durationRoutingRequest = computed(() => buildRequest(props.form));

const formStore = useGenerateFormStore();
const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const appPrefs = useAppPrefsStore();
const gallery = useGalleryStore();
const libraryPrefs = useLibraryPrefsStore();
const loraRoute = computed(() => {
  const binding = loraHostBinding(props.form.loras);
  const selected =
    binding.kind === "bound"
      ? binding.hostId
      : normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all);
  const route = hosts.resolveRoute(selected, props.form.model || null);
  if (binding.kind === "bound" && (!route || !loraBindingMatchesRoute(binding, route))) return null;
  return route;
});
const controlAdapters = ref<Ltx2ControlAdapterInfo[]>([]);
const cameraControls = ref<Ltx2CameraControlInfo[]>([]);
const cameraControlsLoaded = ref(false);
const cameraUnsupportedReason = ref<string | null>(null);
let controlAdaptersEpoch = 0;
watch(
  [
    () => props.form.model,
    () => normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
    () => hosts.all.map((host) => `${host.id}:${host.status}:${host.baseUrl}`).join("|"),
  ],
  async () => {
    const epoch = ++controlAdaptersEpoch;
    // Drop the previous model's reason immediately; keeping it while the
    // new request is in flight shows a stale explanation for the wrong model.
    cameraUnsupportedReason.value = null;
    controlAdapters.value = [];
    cameraControls.value = [];
    cameraControlsLoaded.value = false;
    if (props.form.family !== "ltx2" || !props.form.model) return;
    const route = hosts.resolveRoute(
      normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
      props.form.model,
    );
    if (!route) return;
    const controlsRequest = apiJsonTo<Ltx2ControlAdapterInfo[]>(
      route.target,
      `/api/capabilities/ltx2-control-adapters?model=${encodeURIComponent(props.form.model)}`,
    )
      .then((options) => {
        if (epoch !== controlAdaptersEpoch) return;
        controlAdapters.value = options;
        if (
          props.form.icLoraControl &&
          !options.some((adapter) => adapter.id === props.form.icLoraControl)
        ) {
          props.form.icLoraControl = null;
        }
      })
      .catch(() => {
        if (epoch !== controlAdaptersEpoch) return;
        controlAdapters.value = [];
        props.form.icLoraControl = null;
      });
    const cameraRequest = apiJsonTo<unknown>(
      route.target,
      `/api/capabilities/ltx2-camera-controls?model=${encodeURIComponent(props.form.model)}&detail=1`,
    )
      .then((body) => {
        if (epoch !== controlAdaptersEpoch) return;
        const availability = parseCameraControlAvailability(body);
        const cameras = availability.controls;
        cameraControls.value = cameras;
        cameraUnsupportedReason.value = availability.unsupportedReason;
        cameraControlsLoaded.value = true;
        const compatible = (value: string | null) =>
          !value || !isCameraMotionPreset(value) || cameras.some((camera) => camera.id === value);
        if (!compatible(props.form.cameraControl)) {
          props.form.loras = syncCameraMotionLora(
            props.form.loras,
            props.form.cameraControl,
            null,
            (path, scale) => ({ path, name: path, scale, trainedWords: [] }),
          );
          props.form.cameraControl = null;
        }
        for (const clip of draft.clips) {
          if (!compatible(clip.cameraControl)) clip.cameraControl = null;
        }
      })
      .catch(() => {
        if (epoch !== controlAdaptersEpoch) return;
        cameraControls.value = [];
        cameraUnsupportedReason.value = null;
        cameraControlsLoaded.value = false;
      });
    await Promise.allSettled([controlsRequest, cameraRequest]);
  },
  { immediate: true },
);

// The inspector is docked to the window's right edge, so dragging its left
// handle left grows it. Keep pointer moves local and persist only on commit.
const draftInspectorWidth = ref<number | null>(null);
const inspectorWidth = computed(() => draftInspectorWidth.value ?? appPrefs.generateParamsWidth);

function onInspectorResize(dx: number) {
  draftInspectorWidth.value = dragWidth("generateParams", appPrefs.generateParamsWidth, dx, "left");
}

async function onInspectorCommit() {
  const width = draftInspectorWidth.value;
  if (width === null) return;
  if (width !== appPrefs.generateParamsWidth) {
    await appPrefs.update({ generateParamsWidth: width });
  }
  draftInspectorWidth.value = null;
}

function onInspectorReset() {
  draftInspectorWidth.value = null;
  void appPrefs.update({ generateParamsWidth: null });
}

const caps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    contractModel.value?.guidance_capabilities,
    // Per-model source-image contract (#772): the picked row when we have it,
    // otherwise the form's snapshot of it. Without this the Source image well
    // would render for a text-to-video wan checkpoint that rejects one.
    contractModel.value?.source_image ?? props.form.sourceImageCapability,
    effectiveGenerationRecipe(contractModel.value, props.form.pipeline),
  ),
);
/** The model's image-attachment shape — one shared policy, never a local
 * heuristic. Only `none` hides the primary conditioning editor. */
const sourcePlan = computed(() => sourceMediaPlan(caps.value));
const sequenceSourceImagesSupported = computed(
  () => (contractModel.value?.source_image ?? props.form.sourceImageCapability) !== "unsupported",
);
const showSourceMedia = computed(() => !isSequence.value && sourcePlan.value.kind !== "none");
/** Identity is capability-gated on positive knowledge only: an unread or
 * absent `supports_identity` renders nothing at all rather than a control for
 * a feature this host does not have. Sequence clips carry no identity slot.
 *
 * A staged photo PARKS rather than blocking: `buildRequest` keeps it off the
 * wire, `identityConditioningValidationError` reports nothing for a checkpoint
 * that cannot take it, and selecting a qualified model again brings the well
 * back with the photo still in it — the same treatment staged LTX-2 media
 * gets. Web applies the same rule. */
const showIdentity = computed(() => !isSequence.value && props.form.identitySupported === true);
/* The two doors under the strength slider. Painting a mask belongs to the
 * source well, which alone knows whether this recipe and this attachment can
 * take one, so the well answers and the group only renders the button. */
const sourceWell = ref<InstanceType<typeof SourceImageWell> | null>(null);
const maskDoorAvailable = computed(() => sourceWell.value?.maskAvailable === true);
const identityRevealed = ref(false);
const identityWellOpen = computed(
  () =>
    showIdentity.value &&
    // With no source group there is no door to open it from, so the well is
    // the control itself.
    (!showSourceMedia.value || identityRevealed.value || Boolean(props.form.identityImage)),
);
const activeRecipe = computed(() =>
  effectiveGenerationRecipe(contractModel.value, props.form.pipeline),
);
/* A fixed control explains itself with the profile's own sentence, or with
 * nothing at all. The inspector never composes that copy: the old hard-coded
 * distilled-CFG line was false for H3, whose guidance is pinned at 0. */
const stepsNote = computed(() => controlNote(activeRecipe.value?.steps));
/* Draft / Good / Best are the recipe's own floor, default and ceiling. A
 * recipe that pins its steps offers no choice, so the rows disappear rather
 * than reading as three names for one number. */
const qualityRows = computed(() => qualityPresets(activeRecipe.value?.steps));
const activeQuality = computed(() => activeQualityPreset(qualityRows.value, props.form.steps));
function pickQuality(preset: QualityPreset) {
  props.form.steps = preset.steps;
}
const guidanceNote = computed(() => controlNote(activeRecipe.value?.guidance));

// ── 3-D mesh (canvasless recipes) ───────────────────────────────────────────
// A canvasless recipe advertises `resolution.domain: "none"` with a zero
// canvas, so Shape and Resolution have nothing to bind to and are hidden
// rather than steering a size the request ignores. Everything in the Mesh
// group below is read off the recipe's own `capabilities.mesh` block — the
// octree ladder, the iso-threshold bounds and the face bounds are the
// server's, never a client constant.
const canvasless = computed(() => caps.value.canvasless || outputShape.value.canvasless);
const meshProfile = computed(() => caps.value.mesh ?? null);
/** `null` on the form means "use the profile default", and the default is
 * the segment that reads as chosen. */
const octreeValue = computed(
  () => props.form.mesh.octreeResolution ?? meshProfile.value?.octree_default ?? 0,
);
const octreeOptions = computed(() => {
  const ladder = meshDetailLadder(
    meshProfile.value?.octree_resolutions,
    meshProfile.value?.octree_default,
  );
  // A print made at a rung the three-step ladder skips — 192 or 320, from the
  // CLI, an older client or a reuse — must stay visible rather than leave the
  // control with nothing selected while the truth line reads `octree 192`.
  const current = octreeValue.value;
  if (current <= 0 || ladder.some((step) => step.value === current)) return ladder;
  return [...ladder, { value: current, label: String(current) }].sort((a, b) => a.value - b.value);
});
const thresholdControl = computed(() => meshProfile.value?.threshold ?? null);
const thresholdValue = computed(
  () => props.form.mesh.threshold ?? thresholdControl.value?.default ?? 0,
);
const thresholdNote = computed(() => controlNote(thresholdControl.value));
function setTargetFaces(raw: string) {
  const trimmed = raw.trim();
  if (!trimmed) {
    props.form.mesh.targetFaces = null;
    return;
  }
  const value = Number(trimmed);
  props.form.mesh.targetFaces = Number.isFinite(value) && value > 0 ? Math.round(value) : null;
}
/** A budget outside the advertised bounds is a 422 at admission: name the
 * bounds inline, as Steps does, instead of snapping the typed value. */
const targetFacesError = computed(() =>
  meshTargetFacesError(props.form.mesh.targetFaces, meshProfile.value),
);
// The sequence opening image is primary-form source media, so — exactly like
// the one-shot source well — it never contributes to the Advanced badge.
const advancedCount = computed(() =>
  isSequence.value
    ? Number(
        caps.value.supportsNegativePrompt && draft.clips.some((clip) => clip.negativePrompt.trim()),
      ) + Number(Boolean(draft.clips.some((clip) => clip.cameraControl)))
    : advancedActiveCount(props.form),
);
const showGenerateAudio = computed(() => caps.value.offersAudioControl);
const generateAudio = computed(() =>
  isSequence.value ? draft.enableAudio : props.form.enableAudio,
);
const audioOutputSupported = computed(() =>
  isSequence.value
    ? props.chainLimits?.supports_audio === true
    : caps.value.supportsAudio && selectedModel.value?.supports_audio !== false,
);
const audioOutputUnavailableReason = computed(() => {
  if (!showGenerateAudio.value || audioOutputSupported.value) return null;
  if (isSequence.value) {
    return props.chainLimits?.supports_audio === false
      ? "Generated audio is unavailable for this sequence on the selected host."
      : null;
  }
  if (selectedModel.value?.supports_audio === false) {
    return "Audio assets are not included with this checkpoint. Video generation remains available.";
  }
  return caps.value.outputDeliveryReason ?? "Generated audio is unavailable for this recipe.";
});
function setGenerateAudio(value: boolean) {
  if (isSequence.value) draft.enableAudio = value;
  else props.form.enableAudio = value;
}
const showLoras = computed(() => caps.value.supportsLora && !isSequence.value);
/** One card for every clip control the mock groups together. */
const showClipCard = computed(() => caps.value.supportsVideo);
const advancedExpanded = ref(false);
/** Starters shows pictures; the save/search/sort manager is behind Edit…. */
const managingStarters = ref(false);

// ── Style rows (the ONE style picker lives on the composer's chip) ──────────
// The Settings tab has no style field: the mock puts the picker on the
// composer and nowhere else. These rows are still read here for the Output
// switch, the quality presets, the canvas, and the mesh block.
const { installedModels, stickyTarget, selectedModel, contractModel, targetModels } =
  useStylePicker(() => props.form);

// ── Output (One shot | Sequence) — a setting, not a place ────────────────────
const draft = useSequenceDraftStore();
const lastUsed = useLastUsedStylesStore();
const isSequence = computed(() => draft.output === "sequence");
const canPredictDuration = computed(
  () =>
    !isSequence.value &&
    selectedModel.value?.supports_duration_prediction === true &&
    selectedModel.value.runtime_ready !== false,
);
function setPredictDuration(value: boolean) {
  props.form.predictDuration = value;
  if (!value && !Number.isFinite(props.form.frames)) {
    props.form.frames = selectedModel.value?.default_frames ?? 25;
  }
}
/**
 * The clip styles that can author a sequence, read from the target's WHOLE
 * inventory rather than the picker's rows: the picker is narrowed to the
 * section the view is currently in, which while the user is still looking at
 * Still picture contains no clip style at all.
 */
const sequenceCapableModels = computed(() =>
  stickyTarget.value &&
  stickyTarget.value !== "capable" &&
  (hostModels.byHost[stickyTarget.value]?.fetchedAt ?? 0) === 0
    ? []
    : modelsForOutputKind(targetModels.value, "clip").filter(modelSupportsSequence),
);
/** The styles Still picture can make, for the way back out of a clip draft. */
const stillModels = computed(() => modelsForOutputKind(targetModels.value, "still"));
const defaultFrames = computed(() =>
  defaultClipFrames(
    selectedModel.value,
    props.chainLimits ?? null,
    sequenceMotionTailFrames(selectedModel.value),
  ),
);

function setOutputMode(mode: string | number) {
  const next = mode === "sequence" ? "sequence" : "single";
  if (next === draft.output) return;
  if (next === "sequence") {
    // A non-capable selection is remembered and swapped for the first
    // capable model; switching back restores it.
    const current = selectedModel.value;
    if (!current || !sequenceCapableModels.value.some((m) => m.name === current.name)) {
      draft.lastSingleModel = props.form.model || null;
      const pick = lastUsed.pick("clip", sequenceCapableModels.value);
      if (pick) formStore.applyModel(pick);
      else {
        props.form.model = "";
        props.form.family = "";
      }
    }
  } else {
    // Back to Still picture, which must be left holding a style that section
    // can make: the parked one when it is still a picture style, otherwise the
    // first installed one. A machine with no picture style at all keeps what
    // the form has rather than clearing it.
    const parked = draft.lastSingleModel
      ? findInstalledModel(stillModels.value, draft.lastSingleModel)
      : null;
    const current = selectedModel.value;
    const restored =
      parked ??
      (current && stillModels.value.some((m) => m.name === current.name)
        ? null
        : lastUsed.pick("still", stillModels.value));
    if (restored) formStore.applyModel(restored);
    draft.lastSingleModel = null;
  }
  draft.setOutput(
    next,
    {
      getPrompt: () => props.form.prompt,
      setPrompt: (value) => (props.form.prompt = value),
    },
    defaultFrames.value,
  );
}

// ── Shape + resolution projection ────────────────────────────────────────────
const sourceDimensions = computed(() => {
  if (isSequence.value) {
    // Sequence stage images predate the additive per-model field, so absence
    // stays compatible. Only an explicit unsupported contract parks them.
    if (!sequenceSourceImagesSupported.value) {
      return null;
    }
    const { width, height } = draft.openingImage ?? {};
    return width && height ? { width, height } : null;
  }
  if (!caps.value.supportsSourceImage) return null;
  // Keep a parked image and its dimensions intact across model switches, but
  // do not let them project Source shape/resolution controls for a checkpoint
  // whose request cannot carry that image. Switching back recomputes these
  // controls from the retained dimensions without destructive cleanup.
  return props.form.sourceImageWidth && props.form.sourceImageHeight
    ? {
        width: props.form.sourceImageWidth,
        height: props.form.sourceImageHeight,
      }
    : null;
});
const sourceResolution = computed(() =>
  sourceDimensions.value
    ? resolveSourceResolution(
        sourceDimensions.value,
        contractModel.value ?? props.form.family,
        props.form.pipeline,
      )
    : null,
);
/** One resolver drives the chips, the pills, the badge and the sentence. */
const shapeInput = computed<OutputShapeInput>(() => ({
  model: contractModel.value ?? null,
  family: props.form.family,
  pipeline: props.form.pipeline,
  width: props.form.width,
  height: props.form.height,
  source: sourceDimensions.value,
  intent: props.canvasIntent,
}));
const outputShape = computed(() => resolveOutputShape(shapeInput.value));
const followsSource = computed(
  () =>
    outputShape.value.state === "follows-source" || outputShape.value.state === "matches-source",
);
const shapeOptions = computed(() => outputShape.value.families);
const shapeId = computed(() => outputShape.value.selectedFamilyId);
const shapeApproximate = computed(() => outputShape.value.approximate);
const resolutionRatio = computed(() => props.form.width / props.form.height);
const resolutionOptions = computed(() =>
  outputShape.value.sizes.map((size) => ({
    id: size.id,
    mp: (size.width * size.height) / 1_000_000,
    label: size.label,
    sub: size.mark ? `${size.megapixels} · ${size.mark}` : size.megapixels,
    width: size.width,
    height: size.height,
  })),
);
const resolutionSizeId = computed(() => outputShape.value.selectedSizeId);
const resolutionWarning = computed(() =>
  resolutionValidationWarning(
    props.form.width,
    props.form.height,
    contractModel.value,
    props.form.pipeline,
  ),
);
const resolutionError = computed(() =>
  resolutionValidationError(
    props.form.width,
    props.form.height,
    contractModel.value,
    props.form.pipeline,
  ),
);
const stepsError = computed(() =>
  profileStepsValidationError(props.form.steps, contractModel.value, props.form.pipeline),
);

function onShape(id: string) {
  const size = sizeForFamily(id, shapeInput.value);
  if (!size) return;
  emit("canvas-intent", id === SOURCE_FAMILY_ID ? "source" : "manual");
  props.form.width = size.width;
  props.form.height = size.height;
}
function matchSource() {
  const source = sourceResolution.value;
  if (!source) return;
  emit("canvas-intent", "source-exact");
  props.form.width = source.output.width;
  props.form.height = source.output.height;
}
function onResolution(id: string | number) {
  const size = outputShape.value.sizes.find((candidate) => candidate.id === id);
  if (!size) return;
  emit("canvas-intent", intentForCanvas(shapeInput.value, size));
  props.form.width = size.width;
  props.form.height = size.height;
}

// ── Seed (mode is UI-owned to avoid focus loss — see the previous ParamPanel) ─
const uiSeedMode = ref<"random" | "fixed">(seedMode(props.form.seed));
watch(
  () => props.form.seed,
  (seed) => {
    if (seedMode(seed) === "fixed") uiSeedMode.value = "fixed";
  },
);
function setSeedMode(mode: "random" | "fixed") {
  uiSeedMode.value = mode;
  if (mode === "random") {
    props.form.seed = "";
  } else if (seedMode(props.form.seed) === "random") {
    props.form.seed = String(props.lastSeed ?? randomSeed());
  }
}
/** The mono truth beside the label: the pinned seed, or the last print's. */
const seedReadout = computed(() => {
  const raw = props.form.seed.trim();
  if (uiSeedMode.value === "fixed") return raw === "" ? null : raw;
  return props.lastSeed === null ? null : String(props.lastSeed);
});
const seedHint = computed(() => {
  if (uiSeedMode.value !== "fixed") return null;
  const raw = props.form.seed.trim();
  if (raw === "") return "Empty — a random seed will be used.";
  if (!Number.isFinite(Number(raw))) return "Not a number — a random seed will be used.";
  return null;
});
function rerollSeed() {
  props.form.seed = String(randomSeed());
}

// ── File under (Create-time Library filing) ────────────────────────────────

// Positive knowledge only, exactly like the V3 Library's own gate: an older
// server, `MOLD_DB_DISABLE=1`, and a capability snapshot nobody has read yet
// all answer false and the group stays hidden. A PINNED machine is the one
// that will file this print, so it alone decides; automatic routing could
// land on any machine whose capabilities we have actually read.
const fileUnderHostIds = computed<string[]>(() => {
  const sticky = stickyTarget.value;
  if (sticky && sticky !== "capable") return [sticky];
  return hosts.all.map((host) => host.id);
});
const showFileUnder = computed(() =>
  fileUnderHostIds.value.some((id) => fileUnderAvailable(hosts.capabilities[id])),
);

// Suggestions and the collection picker are the Library's own merged views —
// one tag list and one collection shelf across every connected machine.
const fileUnderTags = computed(() => gallery.mergedTags);
const fileUnderCollections = computed(() => gallery.mergedCollections);

watch(
  showFileUnder,
  (visible) => {
    if (!visible) return;
    gallery.syncBuckets();
    void gallery.fetchCollections();
    void gallery.fetchTags();
  },
  { immediate: true },
);

// The title match is re-derived from the LIVE title on every keystroke, and
// the form carries the winner so `buildRequest` can offer it without knowing
// about stores. Nothing here creates a collection.
watch(
  [() => props.form.title, fileUnderCollections],
  ([title, collections]) => {
    props.form.fileUnderMatch = matchCollection(title, collections);
  },
  { immediate: true },
);

// The preview names the file the print will land as, so it follows the same
// extension `buildRequest` ships.
const fileUnderExtension = computed(() => props.form.outputFormat);

function setFileUnder(next: FileUnderState) {
  props.form.fileUnder = next;
}

// The primary Reset restores every general setting, including Batch. Prompt,
// model, and retained prepared work survive; changing Batch makes that work
// explicitly stale instead of silently discarding it.
function resetSettings() {
  emit("reset-settings");
  resetFormToModelDefaults(props.form, selectedModel.value);
  // The canvas is part of what Reset restores, so its authority resets with
  // it — otherwise the next model change would re-snap the reset canvas back
  // onto the attached source (#1166).
  emit("canvas-intent", "model-default");
  if (isSequence.value) {
    // Reset restores the MODEL's answer, not a flat off: sound is on wherever
    // the clip renders it, so resetting to false handed back a silent draft
    // the user never chose.
    draft.enableAudio = props.chainLimits?.supports_audio === true;
    // `resetFormToModelDefaults` discards one-shot source media wholesale; the
    // sequence's opening image is the same primary-form media, so it goes with
    // it. (`form.strength` / `form.sourceFit` are already reset above.)
    draft.clearOpeningImage();
  }
}
defineExpose({ setOutputMode });
</script>

<template>
  <aside class="ms-inspector" data-test="inspector-panel" :style="{ width: `${inspectorWidth}px` }">
    <PanelResizeHandle
      class="absolute inset-y-0 -left-0.5 z-10"
      label="Resize generation settings"
      @resize="onInspectorResize"
      @commit="onInspectorCommit"
      @reset="onInspectorReset"
    />
    <div class="ms-inspector__tabs" role="tablist" aria-label="Inspector">
      <button
        v-for="t in INSPECTOR_TABS"
        :key="t.id"
        type="button"
        role="tab"
        class="ms-inspector__tab"
        :data-test="`inspector-tab-${t.id}`"
        :aria-selected="tab === t.id"
        :data-on="tab === t.id ? 'true' : undefined"
        @click="emit('update:tab', t.id)"
      >
        {{ t.label }}
      </button>
    </div>
    <div v-if="tab === 'starters'" class="ms-inspector__scroll" data-test="inspector-starters">
      <div class="ms-inspector__head">
        <p class="ms-inspector__lead">
          Pick a starting point and change the words — every setting comes with it.
        </p>
        <button
          type="button"
          class="ms-inspector__reset"
          data-test="edit-starters"
          :aria-pressed="managingStarters"
          :data-on="managingStarters ? 'true' : undefined"
          @click="managingStarters = !managingStarters"
        >
          <Icon name="pencil" :size="12" />
          Edit…
        </button>
      </div>
      <TemplatesPanel
        v-if="managingStarters"
        :form="form"
        :models="installedModels"
        @load="emit('load-template', $event)"
      />
      <StarterList v-else :models="installedModels" @load="emit('load-template', $event)" />
    </div>
    <div v-else-if="tab === 'recent'" class="ms-inspector__scroll" data-test="inspector-recent">
      <p v-if="recent.length === 0" class="ms-inspector__lead">
        Pictures you make show up here, newest first.
      </p>
      <RecentPrints
        :prints="recent"
        :models="installedModels"
        @reuse="emit('reuse-print', $event)"
      />
    </div>
    <div v-else class="ms-inspector__scroll">
      <!-- The mock's Settings tab starts at "Start from a photo": the style
           lives on the composer's chip and has NO field here. The one thing
           this row keeps is the way back to the style's own defaults, which
           the mock has nowhere else. -->
      <div class="ms-inspector__head ms-inspector__head--bare">
        <button
          type="button"
          class="ms-inspector__reset"
          data-test="settings-reset"
          title="Reset to the style's defaults"
          aria-label="Reset to the style's defaults"
          @click="resetSettings"
        >
          <Icon name="refresh" :size="12" />
          Reset
        </button>
      </div>

      <!-- Start from a photo — primary-form image conditioning; the model
           dictates whether (and how) it renders, exactly like resolutions.
           Face conditioning is its own partition, not source media, so it sits
           beside the source wells behind this group's second door and is
           mounted only for a checkpoint that advertises identity support. -->
      <div v-if="showSourceMedia" class="ms-field" data-test="inspector-source-media">
        <div class="ms-group-label uppercase">Start from a photo</div>
        <SourceImageWell ref="sourceWell" :form="form" :selected-model="contractModel" />
        <div v-if="maskDoorAvailable || showIdentity" class="ms-doors">
          <button
            v-if="maskDoorAvailable"
            type="button"
            class="ms-door"
            data-test="source-edit-mask"
            @click="sourceWell?.openMaskEditor()"
          >
            Paint a mask
          </button>
          <button
            v-if="showIdentity"
            type="button"
            class="ms-door"
            data-test="open-identity"
            :data-on="identityWellOpen ? 'true' : undefined"
            :aria-expanded="identityWellOpen"
            @click="identityRevealed = !identityRevealed"
          >
            Use a face
          </button>
        </div>
      </div>

      <div v-if="identityWellOpen" class="ms-field" data-test="inspector-identity">
        <IdentityWell :form="form" />
      </div>

      <!-- The sequence's opening image sits in the same primary slot: staged
           source media is an authoring decision, never an Advanced knob. -->
      <div
        v-if="isSequence && sequenceSourceImagesSupported"
        class="ms-field"
        data-test="inspector-sequence-opening-image"
      >
        <SequenceOpeningImageWell :form="form" :upscalers="models.upscalers" />
      </div>

      <!-- Quality — three rungs of the recipe's own steps range, driving the
           Detail slider below rather than a second setting. -->
      <div v-if="qualityRows.length > 0" class="ms-field" data-test="quality-presets">
        <div class="ms-group-label uppercase">Quality</div>
        <div class="ms-quality">
          <button
            v-for="preset in qualityRows"
            :key="preset.key"
            type="button"
            class="ms-quality__row"
            :data-test="`quality-${preset.key}`"
            :data-on="activeQuality === preset.key ? 'true' : undefined"
            :aria-pressed="activeQuality === preset.key"
            @click="pickQuality(preset)"
          >
            <span class="ms-quality__label">{{ preset.label }}</span>
            <span class="ms-quality__meta">{{ preset.steps }} passes</span>
          </button>
        </div>
      </div>

      <!-- Shape -->
      <div v-if="!canvasless" class="ms-field">
        <div class="ms-field__label">Shape</div>
        <ShapePicker
          :model-value="shapeId"
          :options="shapeOptions"
          :approximate="shapeApproximate"
          label="Aspect ratio"
          @update:model-value="onShape"
        />
      </div>

      <!-- Resolution -->
      <div v-if="!canvasless" class="ms-field">
        <div class="ms-field__label">Resolution</div>
        <ResolutionSelector
          :model-value="resolutionSizeId"
          :ratio="resolutionRatio"
          :options="resolutionOptions"
          :resolved-width="form.width"
          :resolved-height="form.height"
          :custom-label="sourceResolution ? outputShape.badge : undefined"
          :status="outputShape.status"
          @update:model-value="onResolution"
        />
        <button
          v-if="sourceResolution && !followsSource"
          type="button"
          class="ms-field__match-source"
          data-test="match-source-resolution"
          @click="matchSource"
        >
          Match source
        </button>
        <p v-if="resolutionError" class="ms-field__error" role="alert">{{ resolutionError }}</p>
        <p
          v-else-if="resolutionWarning"
          class="ms-field__hint ms-field__hint--warning"
          data-test="resolution-warning"
        >
          {{ resolutionWarning }}
        </p>
      </div>

      <!-- Detail (steps) -->
      <div class="ms-field">
        <SliderRow
          :model-value="form.steps"
          :min="activeRecipe?.steps.min ?? 1"
          :max="activeRecipe?.steps.max ?? 100"
          :step="activeRecipe?.steps.step ?? 1"
          :disabled="activeRecipe?.steps.mode === 'fixed'"
          label="Detail"
          :value-label="`${form.steps} passes`"
          low="Faster"
          high="Finer"
          @update:model-value="form.steps = $event"
        />
        <p v-if="stepsError" class="ms-field__error" role="alert">{{ stepsError }}</p>
        <p
          v-else-if="stepsNote"
          class="ms-field__hint ms-field__hint--after-slider"
          data-test="fixed-steps-hint"
        >
          {{ stepsNote }}
        </p>
      </div>

      <!-- Stick to my words (guidance) -->
      <div class="ms-field">
        <SliderRow
          :model-value="caps.fixedGuidance ?? form.guidance"
          :min="activeRecipe?.guidance.min ?? 0"
          :max="activeRecipe?.guidance.max ?? 100"
          :step="activeRecipe?.guidance.step ?? 0.1"
          label="Stick to my words"
          :value-label="(caps.fixedGuidance ?? form.guidance).toFixed(1)"
          low="Loose"
          high="Literal"
          :disabled="activeRecipe?.guidance.mode === 'fixed' || !caps.guidanceAdjustable"
          @update:model-value="form.guidance = $event"
        />
        <p
          v-if="guidanceNote"
          class="ms-field__hint ms-field__hint--after-slider"
          data-test="fixed-guidance-hint"
        >
          {{ guidanceNote }}
        </p>
      </div>

      <!-- Add-on looks — a main-column group, never an Advanced knob -->
      <div v-if="showLoras" class="ms-field" data-test="inspector-loras">
        <LoraStack
          :form="form"
          :model="form.model"
          :route="loraRoute"
          @append-word="emit('append-word', $event)"
        />
      </div>

      <!-- 3-D object, built entirely from the recipe's advertised `mesh`
           block, so a host that widens the octree ladder or the face bounds
           widens this group with no client release. -->
      <div v-if="meshProfile" class="ms-field ms-card" data-test="mesh-controls">
        <div class="ms-group-label uppercase">3-D object</div>
        <div class="ms-card__row">
          <SegmentedControl
            v-if="octreeOptions.length > 0"
            :model-value="octreeValue"
            :options="octreeOptions"
            label="Surface detail"
            data-test="mesh-octree"
            @update:model-value="form.mesh.octreeResolution = $event"
          />
          <p class="ms-card__truth" data-test="mesh-octree-truth">
            octree {{ octreeValue }} · more detail is slower
          </p>
        </div>
        <SliderRow
          v-if="thresholdControl"
          :model-value="thresholdValue"
          :min="thresholdControl.min"
          :max="thresholdControl.max"
          :step="thresholdControl.step"
          :disabled="thresholdControl.mode === 'fixed'"
          label="How tight to the photo"
          :value-label="thresholdValue.toFixed(2)"
          low="Puffier"
          high="Sharper edges"
          @update:model-value="form.mesh.threshold = $event"
        />
        <p v-if="thresholdNote" class="ms-field__hint" data-test="mesh-threshold-note">
          {{ thresholdNote }}
        </p>
        <div class="ms-field--row">
          <div>
            <label class="ms-field__label ms-field__label--inline" for="mesh-target-faces">
              Simplify to
            </label>
            <p class="ms-field__hint">Fewer faces load faster in other apps</p>
          </div>
          <input
            id="mesh-target-faces"
            data-selectable
            data-test="mesh-target-faces"
            type="number"
            inputmode="numeric"
            :min="meshProfile.target_faces_min"
            :max="meshProfile.target_faces_max"
            placeholder="keep every detail"
            :value="form.mesh.targetFaces ?? ''"
            :aria-invalid="targetFacesError ? 'true' : undefined"
            class="ms-seed__input ms-card__faces font-mono text-micro"
            @input="setTargetFaces(($event.target as HTMLInputElement).value)"
          />
        </div>
        <p
          v-if="targetFacesError"
          class="ms-field__error"
          role="alert"
          data-test="mesh-target-faces-error"
        >
          {{ targetFacesError }}
        </p>
        <p class="ms-field__hint">
          Start from a photo of one object. You'll get a turntable preview and can export .obj or
          .glb. Leave the face budget blank to keep every detail —
          {{ meshProfile.target_faces_min }}–{{ meshProfile.target_faces_max }} triangles when
          simplifying.
        </p>
      </div>

      <!-- Clip — one card for length, smoothness and sound. Duration is the
           human-facing control; exact frames stay in Advanced. -->
      <div v-if="showClipCard" class="ms-field ms-card" data-test="clip-card">
        <div class="ms-group-label uppercase">Clip</div>
        <template v-if="!isSequence">
          <div v-if="canPredictDuration" class="ms-field--row" data-test="predict-duration-control">
            <span class="ms-field__label ms-field__label--inline">Predict duration</span>
            <SwitchToggle
              :model-value="form.predictDuration"
              label="Predict duration from prompt"
              @update:model-value="setPredictDuration"
            />
          </div>
          <VideoDurationSlider
            v-if="!form.predictDuration"
            :frames="form.frames"
            :fps="form.fps"
            :model="contractModel"
            :family="form.family"
            :model-name="form.model"
            :source-image-capability="contractModel?.source_image ?? form.sourceImageCapability"
            :routing-request="durationRoutingRequest"
            @update:frames="form.frames = $event"
          />
          <p v-else class="ms-field__hint" data-test="predicted-duration-hint">
            The host will choose 1–20 seconds from the prompt.
          </p>
        </template>

        <!-- Smoothness — sequence output surfaces it outside Advanced -->
        <div v-if="isSequence" class="ms-field--row" data-test="sequence-fps">
          <span class="ms-field__label ms-field__label--inline">Smoothness</span>
          <Stepper
            :model-value="form.fps"
            :min="
              activeRecipe?.temporal?.fps.mode === 'adjustable' ? activeRecipe.temporal.fps.min : 1
            "
            :max="
              activeRecipe?.temporal?.fps.mode === 'adjustable' ? activeRecipe.temporal.fps.max : 60
            "
            :step="
              activeRecipe?.temporal?.fps.mode === 'adjustable' ? activeRecipe.temporal.fps.step : 1
            "
            :disabled="activeRecipe?.temporal?.fps.mode === 'fixed'"
            label="Frames per second"
            :format="(v: number) => `${v} fps`"
            @update:model-value="form.fps = $event"
          />
        </div>

        <div v-if="showGenerateAudio" class="ms-field--row" data-test="generate-audio-control">
          <span class="ms-field__label ms-field__label--inline">Add sound</span>
          <SwitchToggle
            :model-value="generateAudio"
            :disabled="!audioOutputSupported"
            label="Add sound"
            @update:model-value="setGenerateAudio"
          />
        </div>
        <p v-if="audioOutputUnavailableReason" class="ms-field__hint">
          {{ audioOutputUnavailableReason }}
        </p>

        <p class="ms-field__hint">Clips take a few minutes. You'll get a still preview first.</p>
      </div>

      <!-- Repeat this look (the seed, in plain words) -->
      <div class="ms-field">
        <div class="ms-field__head">
          <span class="ms-field__label ms-field__label--inline">Repeat this look</span>
          <span v-if="seedReadout" class="ms-field__truth">seed {{ seedReadout }}</span>
        </div>
        <div class="ms-seedmode" role="group" aria-label="Repeat this look">
          <button
            type="button"
            data-test="seed-mode-fixed"
            :aria-pressed="uiSeedMode === 'fixed'"
            class="ms-seedmode__btn"
            :data-on="uiSeedMode === 'fixed' ? 'true' : undefined"
            @click="setSeedMode('fixed')"
          >
            Keep
          </button>
          <button
            type="button"
            data-test="seed-mode-random"
            :aria-pressed="uiSeedMode === 'random'"
            class="ms-seedmode__btn"
            :data-on="uiSeedMode === 'random' ? 'true' : undefined"
            @click="setSeedMode('random')"
          >
            Surprise me
          </button>
        </div>
        <div v-if="uiSeedMode === 'fixed'" class="ms-seed__value">
          <input
            v-model="form.seed"
            data-selectable
            data-test="seed-input"
            type="text"
            inputmode="numeric"
            aria-label="Seed number"
            class="ms-seed__input font-mono text-xs"
          />
          <button
            type="button"
            class="ms-seed__reroll"
            title="Reroll this seed"
            aria-label="Reroll this seed"
            @click="rerollSeed"
          >
            <Icon name="reroll" :size="15" />
          </button>
        </div>
        <p v-if="seedHint" data-test="seed-hint" class="ms-field__hint text-accent">
          {{ seedHint }}
        </p>
        <p v-if="uiSeedMode === 'fixed'" class="ms-field__hint">
          Keeping this number reproduces the same look when you tweak the words.
        </p>
        <p v-else class="ms-field__hint">
          New seed every print<template v-if="lastSeed !== null && !isSequence">
            <!-- lock-last is coupled to single prints; hidden for sequences -->
            ·
            <button
              type="button"
              data-test="lock-last-seed"
              class="ms-seed__lock"
              @click="form.seed = String(lastSeed)"
            >
              lock last ({{ lastSeed }})
            </button></template
          >
        </p>
      </div>

      <!-- Save every result — off, the host publishes the print and moves it
           straight to the trash, so a throwaway never clutters My images yet
           stays recoverable until the trash empties. Hidden for a sequence,
           whose stitched clip is the durable job's whole deliverable. -->
      <div v-if="!isSequence" class="ms-field" data-test="save-result-field">
        <div class="ms-field__head">
          <span class="ms-field__label ms-field__label--inline">Save every result</span>
          <ToggleControl
            :model-value="form.saveResult"
            aria-label="Save every result"
            data-test="save-result"
            @commit="form.saveResult = $event"
          />
        </div>
        <p v-if="!form.saveResult" class="ms-field__hint" data-test="save-result-hint">
          Results go straight to the trash, where they stay until it empties.
        </p>
      </div>

      <!-- Where it runs is the view toolbar's last chip (CreateHeader), not
           an inspector row: at the foot of this list nobody found it. -->

      <!-- File under — where this print lands in the Library, decided before
           Generate rather than discovered after it. -->
      <FileUnderGroup
        v-if="showFileUnder"
        :title="form.title"
        :state="form.fileUnder"
        :auto-tag-title="libraryPrefs.autoTagTitle"
        :tags="fileUnderTags"
        :collections="fileUnderCollections"
        :model="form.model"
        :extension="fileUnderExtension"
        :batch-size="isSequence ? 1 : form.batchSize"
        :output-kind="isSequence ? 'sequence' : 'print'"
        @update:state="setFileUnder"
      />

      <!-- Advanced -->
      <button
        type="button"
        class="ms-advanced"
        data-test="open-advanced"
        :aria-expanded="advancedExpanded"
        aria-controls="desktop-inline-advanced"
        @click="advancedExpanded = !advancedExpanded"
      >
        <span class="ms-advanced__label">
          <Icon name="sliders" :size="14" />
          Advanced
        </span>
        <span class="ms-advanced__meta">
          <BadgePill v-if="advancedCount > 0" tone="accent" data-test="advanced-count"
            >{{ advancedCount }} on</BadgePill
          >
          <Icon :name="advancedExpanded ? 'chevron-up' : 'chevron-down'" :size="15" />
        </span>
      </button>
      <SequenceAdvancedSettings
        v-if="advancedExpanded && isSequence"
        id="desktop-inline-advanced"
        :form="form"
        :camera-controls-enabled="form.family === 'ltx2'"
        :camera-controls="cameraControls"
        :camera-controls-loaded="cameraControlsLoaded"
        :camera-unsupported-reason="cameraUnsupportedReason"
      />
      <AdvancedSettings
        v-else-if="advancedExpanded"
        id="desktop-inline-advanced"
        :form="form"
        :selected-model="contractModel"
        :routing-request="durationRoutingRequest"
        :upscalers="models.upscalers"
        :control-adapters="controlAdapters"
        :camera-controls="cameraControls"
        :camera-controls-loaded="cameraControlsLoaded"
        :camera-unsupported-reason="cameraUnsupportedReason"
        @canvas-intent="emit('canvas-intent', $event)"
      />
    </div>
  </aside>
</template>

<style scoped>
.ms-inspector {
  display: flex;
  flex-direction: column;
  position: relative;
  min-height: 0;
  flex: 0 0 auto;
  border-left: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg-deep);
}
.ms-inspector__scroll {
  flex: 1;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  padding: 14px;
}
/* The strip is exactly one view toolbar tall so its bottom rule meets the
   header's at the divider; the tabs sit on the toolbar control height. */
.ms-inspector__tabs {
  height: var(--mold-shell-viewbar-h);
  flex: 0 0 var(--mold-shell-viewbar-h);
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 0 10px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg);
}
.ms-inspector__tab {
  flex: 1;
  height: var(--mold-ctl-md);
  padding: 0;
  border: 0;
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-xs);
  font-weight: 600;
  white-space: nowrap;
  cursor: pointer;
}
.ms-inspector__tab[data-on="true"] {
  background: var(--mold-row-selected);
  color: var(--mold-text);
}
.ms-inspector__lead {
  margin: 0 0 12px;
  font-size: var(--mold-fs-xs);
  line-height: var(--mold-lh-body);
  color: var(--mold-text-2);
}
.ms-inspector__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 16px;
}
.ms-inspector__head .ms-inspector__lead {
  margin-bottom: 0;
}
/* Reset alone, right-aligned over the first group. */
.ms-inspector__head--bare {
  justify-content: flex-end;
  margin-bottom: 8px;
}
.ms-inspector__reset {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  flex-shrink: 0;
  border: var(--mold-bw) solid var(--mold-border-control);
  background: transparent;
  color: var(--mold-text-2);
  padding: 4px 8px;
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-micro);
  font-weight: 600;
  cursor: pointer;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-inspector__reset:hover {
  background: color-mix(in srgb, var(--mold-text) 6%, transparent);
  color: var(--mold-text);
}
.ms-field {
  margin-bottom: 20px;
}
.ms-field--row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.ms-field__label {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.ms-field__label--inline {
  margin-bottom: 0;
}
.ms-field__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}
.ms-field__truth {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  white-space: nowrap;
}
/* A group the mock draws as a card: the 3-D and Clip blocks. */
.ms-card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 11px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
}
.ms-card__row {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.ms-card__truth {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  margin: 0;
}
/* Written against BOTH classes on purpose: this field also carries
   `.ms-seed__input`, whose rule is declared later, so at equal specificity a
   bare `.ms-card__faces` lost its width and height and the face budget
   rendered as a full-width 32px seed field. */
/* Sized to its own placeholder ("keep every detail", 17 mono characters)
   plus padding and WebKit's spin button, as the mock draws it — a fixed cap
   clipped the words at the larger type scales. */
.ms-seed__input.ms-card__faces {
  width: calc(17ch + 16px + 18px);
  max-width: 100%;
  height: var(--mold-ctl-md);
}
.ms-card .ms-field__hint {
  margin-top: 0;
}
/* Paint a mask · Use a face — the source group's two secondary doors. */
.ms-doors {
  display: flex;
  gap: 6px;
  margin-top: 8px;
}
.ms-door {
  flex: 1;
  height: 28px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
  font-weight: 500;
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-door:hover,
.ms-door[data-on="true"] {
  border-color: var(--mold-border-focus);
  color: var(--mold-text);
}
.ms-quality {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.ms-quality__row {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 34px;
  padding: 0 11px;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: transparent;
  color: var(--mold-text-2);
  cursor: pointer;
}
.ms-quality__row[data-on="true"] {
  border-color: transparent;
  background: color-mix(in srgb, var(--mold-blue) 13%, transparent);
  box-shadow: inset 0 0 0 1px var(--mold-blue);
  color: var(--mold-text);
}
.ms-quality__label {
  flex: 1;
  text-align: left;
  font-size: var(--mold-fs-sm);
  font-weight: 600;
  white-space: nowrap;
}
.ms-quality__meta {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  white-space: nowrap;
  opacity: 0.75;
}
.ms-field__hint {
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-dim);
  margin-top: 6px;
  line-height: var(--mold-lh-snug);
}
.ms-field__hint--after-slider {
  margin-top: 12px;
}
/* The accent is ONE thing; a status is a state token. In the accent this
   sentence was indistinguishable from the Match-source link above it. */
.ms-field__hint--warning {
  color: var(--mold-warning);
}
.ms-field__error {
  font-size: var(--mold-fs-micro);
  color: var(--mold-error);
  margin-top: 6px;
}
.ms-field__match-source {
  margin-top: 7px;
  border: 0;
  background: transparent;
  color: var(--mold-blue);
  font-size: var(--mold-fs-micro);
  cursor: pointer;
}
.ms-field__match-source:hover {
  text-decoration: underline;
}
/* Keep | Surprise me. Named for what it is, NOT `.ms-seg`: Vue puts the
   parent's scope id on a child component's ROOT node, so a local `.ms-seg`
   here matched the shared SegmentedControl's root too (the mesh Surface
   detail control) at equal specificity, and which radius and ground won
   depended on the bundler's chunk order rather than on either author.
   `--mold-radius-3` is the WINDOW radius — 16px in Safelight, which turned a
   34px-tall control into a pill; the mock uses radius-2 on the container and
   radius-1 on the buttons. */
.ms-seedmode {
  display: flex;
  gap: 3px;
  padding: 3px;
  background: var(--mold-bg-deep);
  border: var(--mold-bw) solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
}
.ms-seedmode__btn {
  flex: 1;
  border: 0;
  background: transparent;
  color: var(--mold-text-2);
  padding: 7px;
  border-radius: var(--mold-radius-1);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
}
.ms-seedmode__btn[data-on="true"] {
  background: var(--mold-bg);
  color: var(--mold-text);
}
.ms-seed__value {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
}
.ms-seed__input {
  height: 32px;
  width: 100%;
  min-width: 0;
  border: var(--mold-bw) solid var(--mold-border-control);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg-deep);
  padding: 0 8px;
  font-size: var(--mold-fs-sm);
  color: var(--mold-text);
}
.ms-seed__reroll {
  flex-shrink: 0;
  cursor: pointer;
  color: var(--mold-text-dim);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-seed__reroll:hover {
  color: var(--mold-text);
}
.ms-seed__lock {
  color: var(--mold-sapphire);
  cursor: pointer;
}
.ms-seed__lock:hover {
  text-decoration: underline;
}
.ms-advanced {
  width: 100%;
  border: var(--mold-bw) solid var(--mold-border-control);
  background: transparent;
  color: var(--mold-text-2);
  padding: 11px;
  /* A 40px row on the window radius is a pill in Safelight; the card radius
     is what every other in-view control edge uses. */
  border-radius: var(--mold-radius-2);
  font-size: var(--mold-fs-xs);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  cursor: pointer;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-advanced:hover {
  background: color-mix(in srgb, var(--mold-text) 6%, transparent);
}
.ms-advanced[aria-expanded="true"] {
  border-color: color-mix(in srgb, var(--mold-blue) 45%, var(--mold-border-control));
  background: color-mix(in srgb, var(--mold-blue) 7%, transparent);
  color: var(--mold-text);
}
.ms-advanced__label,
.ms-advanced__meta {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ms-advanced__meta {
  color: var(--mold-text-dim);
}
</style>
