<script setup lang="ts">
import { computed, ref, watch } from "vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import Stepper from "@ui/components/Stepper.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { isMinimaxH3Identity } from "@studio/lib/minimaxH3Authoring";
import { defaultClipFrames, modelsForOutput, sequenceMotionTailFrames } from "@studio/lib/sequence";
import { filterRestrictedModels } from "@studio/lib/modelAccess";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { GenerateForm } from "../../lib/generateForm";
import { buildRequest, resetFormToModelDefaults, seedMode } from "../../lib/generateForm";
import type {
  Ltx2CameraControlInfo,
  Ltx2ControlAdapterInfo,
  ModelEntry,
} from "../../lib/api/types";
import {
  isCameraMotionPreset,
  parseCameraControlAvailability,
  syncCameraMotionLora,
} from "@studio/lib/cameraMotion";
import { apiJsonTo } from "../../lib/api/client";
import {
  filterModelsForTarget,
  findInstalledModel,
  mergeInstalledModels,
} from "../../lib/generateModels";
import { normalizeTargetHost } from "../../lib/hosts";
import { modelDisplayName } from "../../lib/models";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import { sourceMediaPlan } from "@studio/lib/sourceMediaPlan";
import SourceImageWell from "../generate/SourceImageWell.vue";
import { advancedActiveCount } from "../../lib/advancedCount";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
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
import { useGalleryStore } from "../../stores/gallery";
import { useLibraryPrefsStore } from "../../stores/libraryPrefs";
import { fileUnderAvailable, matchCollection, type FileUnderState } from "@studio/lib/fileUnder";
import FileUnderGroup from "./FileUnderGroup.vue";
import { dragWidth } from "../../lib/panelResize";
import ModelPicker from "./ModelPicker.vue";
import AdvancedSettings from "./AdvancedSettings.vue";
import SequenceAdvancedSettings from "./SequenceAdvancedSettings.vue";
import { formatGB } from "../../lib/format";
import PanelResizeHandle from "../shell/PanelResizeHandle.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    /** Seed of the most recent finished print — powers "lock last seed". */
    lastSeed?: number | null;
    /** Per-model chain caps for the selected model, when Create has them —
     * sizes new clips' default frames on the Output switch. */
    chainLimits?: ChainLimits | null;
    /** Why the canvas holds its current size — the shape resolver's authority. */
    canvasIntent?: CanvasIntent;
  }>(),
  { lastSeed: null, chainLimits: null, canvasIntent: "model-default" },
);

const emit = defineEmits<{
  "append-word": [word: string];
  "canvas-intent": [intent: CanvasIntent];
  /** The picker's "Not installed" row: offer the pull for this exact id. */
  "pull-missing-model": [model: string];
}>();
const durationRoutingRequest = computed(() => buildRequest(props.form));

const formStore = useGenerateFormStore();
const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const appPrefs = useAppPrefsStore();
const gallery = useGalleryStore();
const libraryPrefs = useLibraryPrefsStore();
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
    selectedModel.value?.guidance_capabilities,
    // Per-model source-image contract (#772): the picked row when we have it,
    // otherwise the form's snapshot of it. Without this the Source image well
    // would render for a text-to-video wan checkpoint that rejects one.
    selectedModel.value?.source_image ?? props.form.sourceImageCapability,
    effectiveGenerationRecipe(selectedModel.value, props.form.pipeline),
  ),
);
/** The model's image-attachment shape — one shared policy, never a local
 * heuristic. `none` hides the well outright; `h3-references` keeps the H3
 * ordered-reference editor in Advanced. */
const sourcePlan = computed(() => sourceMediaPlan(caps.value));
const sequenceSourceImagesSupported = computed(
  () => (selectedModel.value?.source_image ?? props.form.sourceImageCapability) !== "unsupported",
);
const showSourceMedia = computed(
  () =>
    !isSequence.value &&
    sourcePlan.value.kind !== "none" &&
    sourcePlan.value.kind !== "h3-references",
);
const activeRecipe = computed(() =>
  effectiveGenerationRecipe(selectedModel.value, props.form.pipeline),
);
const advancedCount = computed(() =>
  isSequence.value
    ? Number(sequenceSourceImagesSupported.value && Boolean(draft.openingImage)) +
      Number(
        caps.value.supportsNegativePrompt && draft.clips.some((clip) => clip.negativePrompt.trim()),
      ) +
      Number(Boolean(draft.clips.some((clip) => clip.cameraControl)))
    : advancedActiveCount(props.form),
);
const showGenerateAudio = computed(() =>
  isSequence.value
    ? props.chainLimits?.supports_audio === true
    : caps.value.supportsAudio && !isMinimaxH3Identity(props.form.family, props.form.model),
);
const generateAudio = computed(() =>
  isSequence.value ? draft.enableAudio : props.form.enableAudio,
);
const audioOutputSupported = computed(() => selectedModel.value?.supports_audio !== false);
function setGenerateAudio(value: boolean) {
  if (isSequence.value) draft.enableAudio = value;
  else props.form.enableAudio = value;
}
const advancedExpanded = ref(false);

// ── Model picker (the shared ModelPicker; chains uses the same control) ──────
const installedModels = computed(() =>
  mergeInstalledModels(
    filterRestrictedModels(models.installed, hosts.capabilities.local),
    hostModels.unionInstalled,
  ),
);
const stickyTarget = computed<string | null>(() =>
  normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
);

const selectedModel = computed<ModelEntry | null>(() =>
  hostModels.installedEntryForTarget(props.form.model, stickyTarget.value),
);

/**
 * The form's model when no machine has it installed. Restoring a print whose
 * checkpoint is gone must keep the id visible with a Not installed tag rather
 * than reading "Choose a model" — the raw id stays in `form.model` and in the
 * request either way.
 */
const missingModelId = computed<string | null>(() =>
  props.form.model && !selectedModel.value ? props.form.model : null,
);

const pickerModels = computed<ModelEntry[]>(() => {
  const target = stickyTarget.value;
  const fetched = target && target !== "capable" && (hostModels.byHost[target]?.fetchedAt ?? 0) > 0;
  const forTarget = filterModelsForTarget(
    installedModels.value,
    target,
    fetched ? new Set(hostModels.installedOn(target).map((m) => m.name)) : null,
  );
  // Sequence output narrows the picker to chain-capable video models.
  return modelsForOutput(forTarget, draft.output);
});

// ── Output (One shot | Sequence) — a setting, not a place ────────────────────
const draft = useSequenceDraftStore();
const isSequence = computed(() => draft.output === "sequence");
const sequenceCapableModels = computed(() =>
  stickyTarget.value &&
  stickyTarget.value !== "capable" &&
  (hostModels.byHost[stickyTarget.value]?.fetchedAt ?? 0) === 0
    ? []
    : modelsForOutput(pickerModels.value, "sequence"),
);
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
      const pick = sequenceCapableModels.value[0];
      if (pick) formStore.applyModel(pick);
      else {
        props.form.model = "";
        props.form.family = "";
      }
    }
  } else if (draft.lastSingleModel) {
    const restored = findInstalledModel(installedModels.value, draft.lastSingleModel);
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

const stickyHostMissingModel = computed<string | null>(() => {
  const sel = stickyTarget.value;
  if (!sel || sel === "capable" || !props.form.model) return null;
  const host = hosts.all.find((h) => h.id === sel);
  if (!host) return null;
  const ids = hostModels.hostsFor(props.form.model);
  if (ids.length === 0 || ids.includes(sel)) return null;
  return host.label;
});

const modelDescription = computed(() => {
  const m = selectedModel.value;
  if (!m) return null;
  const parts: string[] = [];
  if (m.description && modelDisplayName(m) === m.name) parts.push(m.description);
  if (m.disk_usage_bytes) parts.push(formatGB(m.disk_usage_bytes));
  if (m.is_loaded) parts.push("loaded");
  return parts.length ? parts.join(" · ") : null;
});

function pickModel(m: ModelEntry) {
  formStore.applyModel(m);
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
        selectedModel.value ?? props.form.family,
        props.form.pipeline,
      )
    : null,
);
/** One resolver drives the chips, the pills, the badge and the sentence. */
const shapeInput = computed<OutputShapeInput>(() => ({
  model: selectedModel.value ?? null,
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
    selectedModel.value,
    props.form.pipeline,
  ),
);
const resolutionError = computed(() =>
  resolutionValidationError(
    props.form.width,
    props.form.height,
    selectedModel.value,
    props.form.pipeline,
  ),
);
const stepsError = computed(() =>
  profileStepsValidationError(props.form.steps, selectedModel.value, props.form.pipeline),
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

const MAX_BATCH_SIZE = 10_000;
const batchLocked = computed(
  () =>
    caps.value.forcesBatchSizeOne ||
    (caps.value.sourceImageMode === "references" && props.form.imageAttachments.length > 0),
);
const batchMax = computed(() => (batchLocked.value ? 1 : MAX_BATCH_SIZE));

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

// Same contract as the Advanced pane's Reset, surfaced without opening it:
// the prompt, the model, and any prepared batch size survive.
function resetSettings() {
  resetFormToModelDefaults(props.form, selectedModel.value);
  // The canvas is part of what Reset restores, so its authority resets with
  // it — otherwise the next model change would re-snap the reset canvas back
  // onto the attached source (#1166).
  emit("canvas-intent", "model-default");
  if (isSequence.value) draft.enableAudio = false;
}
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
    <div class="ms-inspector__scroll">
      <div class="ms-inspector__head">
        <span class="ms-inspector__kicker">Settings</span>
        <button
          type="button"
          class="ms-inspector__reset"
          data-test="settings-reset"
          title="Reset settings to model defaults"
          aria-label="Reset settings to model defaults"
          @click="resetSettings"
        >
          ↺ Reset
        </button>
      </div>

      <!-- Model -->
      <div class="ms-field">
        <div class="ms-field__label">Model</div>
        <ModelPicker
          :models="pickerModels"
          :selected="selectedModel"
          :missing-model="missingModelId"
          :show-availability="!stickyTarget || stickyTarget === 'capable'"
          :browse-target="caps.supportsVideo ? '/models?type=video' : '/models'"
          @pick="pickModel"
          @pick-missing="emit('pull-missing-model', $event)"
        />
        <p v-if="modelDescription" class="ms-field__hint">{{ modelDescription }}</p>
        <p v-if="stickyHostMissingModel" class="ms-field__hint">
          Not on {{ stickyHostMissingModel }} — will download there.
        </p>
      </div>

      <!-- Output — a highlighted card: sequence is a setting of Create, not a place -->
      <div class="ms-field">
        <div class="ms-output" data-test="output-card">
          <div class="ms-field__label">Output</div>
          <SegmentedControl
            :model-value="draft.output"
            :options="[
              { value: 'single', label: 'One shot' },
              { value: 'sequence', label: 'Sequence' },
            ]"
            label="Output"
            data-test="output-mode"
            @update:model-value="setOutputMode"
          />
          <p v-if="isSequence" class="ms-field__hint">
            {{ draft.clips.length }} clips on the composer rail · one-shot and sequence prompts stay
            separate.
          </p>
        </div>
      </div>

      <!-- Source media — primary-form image conditioning; the model dictates
           whether (and how) it renders, exactly like resolutions. -->
      <div v-if="showSourceMedia" class="ms-field" data-test="inspector-source-media">
        <SourceImageWell :form="form" :selected-model="selectedModel" />
      </div>

      <!-- Shape -->
      <div class="ms-field">
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
      <div class="ms-field">
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
          :value-label="`${form.steps} steps`"
          @update:model-value="form.steps = $event"
        />
        <p v-if="stepsError" class="ms-field__error" role="alert">{{ stepsError }}</p>
      </div>

      <!-- Prompt strength (guidance) -->
      <div class="ms-field">
        <SliderRow
          :model-value="caps.fixedGuidance ?? form.guidance"
          :min="activeRecipe?.guidance.min ?? 0"
          :max="activeRecipe?.guidance.max ?? 100"
          :step="activeRecipe?.guidance.step ?? 0.1"
          label="Prompt strength"
          :value-label="(caps.fixedGuidance ?? form.guidance).toFixed(1)"
          :disabled="activeRecipe?.guidance.mode === 'fixed' || !caps.guidanceAdjustable"
          @update:model-value="form.guidance = $event"
        />
        <p
          v-if="!caps.guidanceAdjustable"
          class="ms-field__hint ms-field__hint--after-slider"
          data-test="fixed-guidance-hint"
        >
          Distilled recipe fixes CFG at 1.0. Choose a Dev checkpoint with Auto or a guided pipeline
          to adjust it.
        </p>
      </div>

      <!-- Duration is the human-facing video control; exact frames/FPS stay in Advanced. -->
      <div v-if="caps.supportsVideo && !isSequence" class="ms-field">
        <VideoDurationSlider
          :frames="form.frames"
          :fps="form.fps"
          :model="selectedModel"
          :family="form.family"
          :model-name="form.model"
          :source-image-capability="selectedModel?.source_image ?? form.sourceImageCapability"
          :routing-request="durationRoutingRequest"
          @update:frames="form.frames = $event"
        />
      </div>

      <!-- Frame rate — sequence output surfaces it outside Advanced -->
      <div v-if="isSequence" class="ms-field ms-field--row" data-test="sequence-fps">
        <span class="ms-field__label ms-field__label--inline">Frame rate</span>
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

      <div
        v-if="showGenerateAudio"
        class="ms-field ms-field--row"
        data-test="generate-audio-control"
      >
        <span class="ms-field__label ms-field__label--inline">Generate audio</span>
        <SwitchToggle
          :model-value="generateAudio"
          :disabled="!isSequence && !audioOutputSupported"
          label="Generate audio"
          @update:model-value="setGenerateAudio"
        />
      </div>
      <p
        v-if="showGenerateAudio && !isSequence && !audioOutputSupported"
        class="ms-field__hint -mt-2"
      >
        Audio assets are not included with this checkpoint. Video generation remains available.
      </p>

      <!-- Seed -->
      <div class="ms-field">
        <div class="ms-field__label">Seed</div>
        <div class="ms-seg" role="group" aria-label="Seed mode">
          <button
            type="button"
            data-test="seed-mode-random"
            :aria-pressed="uiSeedMode === 'random'"
            class="ms-seg__btn"
            :data-on="uiSeedMode === 'random' ? 'true' : undefined"
            @click="setSeedMode('random')"
          >
            Random
          </button>
          <button
            type="button"
            data-test="seed-mode-fixed"
            :aria-pressed="uiSeedMode === 'fixed'"
            class="ms-seg__btn"
            :data-on="uiSeedMode === 'fixed' ? 'true' : undefined"
            @click="setSeedMode('fixed')"
          >
            Fixed
          </button>
        </div>
        <div v-if="uiSeedMode === 'fixed'" class="ms-seed__value">
          <input
            v-model="form.seed"
            data-selectable
            data-test="seed-input"
            type="text"
            inputmode="numeric"
            aria-label="Seed value"
            class="ms-seed__input data-mono"
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
        <p v-if="seedHint" data-test="seed-hint" class="ms-field__hint text-safelight">
          {{ seedHint }}
        </p>
        <p v-if="uiSeedMode === 'random'" class="ms-field__hint">
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

      <!-- Batch -->
      <div class="ms-field ms-field--row">
        <span class="ms-field__label ms-field__label--inline">Batch</span>
        <Stepper
          v-if="!isSequence"
          :model-value="batchLocked ? 1 : form.batchSize"
          :min="1"
          :max="batchMax"
          :editable="!batchLocked"
          label="Batch size"
          @update:model-value="form.batchSize = $event"
        />
        <span v-else class="data-mono text-ink-2" data-test="batch-locked">1</span>
      </div>
      <p v-if="isSequence" class="ms-field__hint -mt-2">a sequence renders one timeline</p>
      <p v-else-if="batchLocked" class="ms-field__hint -mt-2">
        Locked to 1 — edit models render one at a time.
      </p>

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
        :upscalers="models.upscalers"
        :camera-controls-enabled="form.family === 'ltx2'"
        :camera-controls="cameraControls"
        :camera-controls-loaded="cameraControlsLoaded"
        :camera-unsupported-reason="cameraUnsupportedReason"
      />
      <AdvancedSettings
        v-else-if="advancedExpanded"
        id="desktop-inline-advanced"
        :form="form"
        :selected-model="selectedModel"
        :routing-request="durationRoutingRequest"
        :upscalers="models.upscalers"
        :control-adapters="controlAdapters"
        :camera-controls="cameraControls"
        :camera-controls-loaded="cameraControlsLoaded"
        :camera-unsupported-reason="cameraUnsupportedReason"
        @append-word="emit('append-word', $event)"
        @canvas-intent="emit('canvas-intent', $event)"
      />
    </div>
  </aside>
</template>

<style scoped>
.ms-inspector {
  position: relative;
  min-height: 0;
  flex: 0 0 auto;
  border-left: 1px solid var(--edge);
  background: var(--bench);
}
.ms-inspector__scroll {
  height: 100%;
  overflow-x: hidden;
  overflow-y: auto;
  padding: 20px 18px;
}
.ms-inspector__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 16px;
}
.ms-inspector__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.ms-inspector__reset {
  flex-shrink: 0;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 4px 8px;
  border-radius: 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.ms-inspector__reset:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
  color: var(--rebate);
}
.ms-field {
  margin-bottom: 20px;
}
/* Highlighted per mockup 1c: the Output choice reads as a mode, not a knob. */
.ms-output {
  border: 1px solid color-mix(in srgb, var(--safelight) 45%, var(--ce));
  background: color-mix(in srgb, var(--safelight) 7%, transparent);
  border-radius: 9px;
  padding: 11px;
}
.ms-output .ms-field__hint {
  margin-top: 8px;
}
.ms-field--row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.ms-field__label {
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.ms-field__label--inline {
  margin-bottom: 0;
}
.ms-field__hint {
  font-size: 11px;
  color: var(--ink-3);
  margin-top: 6px;
  line-height: 1.4;
}
.ms-field__hint--after-slider {
  margin-top: 12px;
}
.ms-field__hint--warning {
  color: var(--safelight);
}
.ms-field__error {
  font-size: 11px;
  color: var(--stop);
  margin-top: 6px;
}
.ms-field__match-source {
  margin-top: 7px;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-size: 11px;
  cursor: pointer;
}
.ms-field__match-source:hover {
  text-decoration: underline;
}
.ms-seg {
  display: flex;
  gap: 3px;
  padding: 3px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: 9px;
}
.ms-seg__btn {
  flex: 1;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  padding: 7px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
}
.ms-seg__btn[data-on="true"] {
  background: var(--bench);
  color: var(--rebate);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
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
  border: 1px solid var(--ce);
  border-radius: 6px;
  background: var(--bath);
  padding: 0 8px;
  font-size: 13px;
  color: var(--rebate);
}
.ms-seed__reroll {
  flex-shrink: 0;
  color: var(--ink-3);
  background: transparent;
  border: 0;
  cursor: pointer;
}
.ms-seed__reroll:hover {
  color: var(--rebate);
}
.ms-seed__lock {
  color: var(--halide);
}
.ms-seed__lock:hover {
  text-decoration: underline;
}
.ms-advanced {
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: 9px;
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    border-color var(--dur-quick) var(--ease);
}
.ms-advanced:hover {
  background: color-mix(in srgb, var(--rebate) 6%, transparent);
}
.ms-advanced[aria-expanded="true"] {
  border-color: color-mix(in srgb, var(--safelight) 45%, var(--ce));
  background: color-mix(in srgb, var(--safelight) 7%, transparent);
  color: var(--rebate);
}
.ms-advanced__label,
.ms-advanced__meta {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ms-advanced__meta {
  color: var(--ink-3);
}
</style>
