<script setup lang="ts">
import { computed, ref, watch } from "vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import Stepper from "@ui/components/Stepper.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { defaultClipFrames, modelsForOutput, sequenceMotionTailFrames } from "@studio/lib/sequence";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { GenerateForm } from "../../lib/generateForm";
import { resetFormToModelDefaults, seedMode } from "../../lib/generateForm";
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
import { advancedActiveCount } from "../../lib/advancedCount";
import { ASPECTS } from "@ui/lib/resolution";
import {
  aspectIdFor,
  closestResolutionPreset,
  megapixelLabel,
  presetsForModel,
  presetsNearRatio,
} from "../../lib/resolutions";
import {
  canvasMatchesSourceResolution,
  resolveSourceResolution,
  sourceResolutionStatus,
} from "@studio/lib/sourceResolution";
import { resolutionValidationError, stepsValidationError } from "../../lib/generateValidation";
import { randomSeed } from "../../stores/generation";
import { useGenerateFormStore } from "../../stores/generateForm";
import { useModelStore } from "../../stores/models";
import { useHostModelsStore } from "../../stores/hostModels";
import { useHostsStore } from "../../stores/hosts";
import { useAppPrefsStore } from "../../stores/appPrefs";
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
  }>(),
  { lastSeed: null, chainLimits: null },
);

const emit = defineEmits<{ "append-word": [word: string] }>();

const formStore = useGenerateFormStore();
const models = useModelStore();
const hostModels = useHostModelsStore();
const hosts = useHostsStore();
const appPrefs = useAppPrefsStore();
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

const caps = computed(() => generationCapabilitiesForFamily(props.form.family, props.form.model));
const advancedCount = computed(() =>
  isSequence.value
    ? Number(Boolean(draft.openingImage)) +
      Number(Boolean(draft.clips.some((clip) => clip.negativePrompt.trim()))) +
      Number(Boolean(draft.clips.some((clip) => clip.cameraControl))) +
      Number(draft.enableAudio)
    : advancedActiveCount(props.form),
);
const advancedExpanded = ref(false);

// ── Model picker (the shared ModelPicker; chains uses the same control) ──────
const installedModels = computed(() =>
  mergeInstalledModels(models.installed, hostModels.unionInstalled),
);
const selectedModel = computed<ModelEntry | null>(() =>
  findInstalledModel(installedModels.value, props.form.model),
);

const stickyTarget = computed<string | null>(() =>
  normalizeTargetHost(appPrefs.settings?.generateTargetHost ?? null, hosts.all),
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
    const { width, height } = draft.openingImage ?? {};
    return width && height ? { width, height } : null;
  }
  return props.form.sourceImageWidth && props.form.sourceImageHeight
    ? {
        width: props.form.sourceImageWidth,
        height: props.form.sourceImageHeight,
      }
    : null;
});
const sourceResolution = computed(() =>
  sourceDimensions.value
    ? resolveSourceResolution(sourceDimensions.value, selectedModel.value ?? props.form.family)
    : null,
);
const followsSource = computed(
  () =>
    sourceResolution.value !== null &&
    canvasMatchesSourceResolution(
      { width: props.form.width, height: props.form.height },
      sourceResolution.value,
    ),
);
const sourceStatus = computed(() =>
  sourceResolution.value ? sourceResolutionStatus(sourceResolution.value) : null,
);
const shapeOptions = computed(() => {
  const source = sourceResolution.value;
  return source
    ? [
        ...ASPECTS,
        {
          id: "source",
          label: "Source",
          ratio: source.output.width / source.output.height,
        },
      ]
    : ASPECTS;
});
const shapeId = computed(() =>
  followsSource.value ? "source" : (aspectIdFor(props.form.width, props.form.height) ?? ""),
);
const resolutionRatio = computed(() => props.form.width / props.form.height);
const resolutionPresets = computed(() =>
  presetsNearRatio(
    presetsForModel(selectedModel.value ?? props.form.family),
    resolutionRatio.value,
  ),
);
const resolutionOptions = computed(() => {
  const presets = resolutionPresets.value.map((preset, index, all) => ({
    mp: (preset.width * preset.height) / 1_000_000,
    label: megapixelLabel(preset.width, preset.height),
    sub:
      all.length === 1
        ? "Native"
        : index === 0
          ? "Small"
          : index === all.length - 1
            ? "Native"
            : "Standard",
    width: preset.width,
    height: preset.height,
  }));
  const exact = presets.some(
    (option) => option.width === props.form.width && option.height === props.form.height,
  );
  if (exact) return presets;
  const currentMp = (props.form.width * props.form.height) / 1_000_000;
  return [
    {
      mp: currentMp,
      label: followsSource.value ? (sourceStatus.value?.label ?? "Source") : "Custom",
      sub: followsSource.value ? "Matched" : "Manual",
      width: props.form.width,
      height: props.form.height,
    },
    ...presets.filter((option) => Math.abs(option.mp - currentMp) > Number.EPSILON),
  ];
});
const resolutionMp = computed(() => {
  const exact = resolutionOptions.value.find(
    (option) => option.width === props.form.width && option.height === props.form.height,
  );
  if (exact) return exact.mp;
  const current = (props.form.width * props.form.height) / 1_000_000;
  return (
    [...resolutionOptions.value].sort(
      (a, b) => Math.abs(a.mp - current) - Math.abs(b.mp - current),
    )[0]?.mp ?? current
  );
});
const resolutionError = computed(() =>
  resolutionValidationError(props.form.width, props.form.height, selectedModel.value),
);
const stepsError = computed(() => stepsValidationError(props.form.steps));

function onShape(id: string) {
  if (id === "source") {
    matchSource();
    return;
  }
  const aspect = ASPECTS.find((option) => option.id === id);
  if (!aspect) return;
  const preset = closestResolutionPreset(
    presetsNearRatio(presetsForModel(selectedModel.value ?? props.form.family), aspect.ratio),
    props.form.width,
    props.form.height,
  );
  if (preset) {
    props.form.width = preset.width;
    props.form.height = preset.height;
  }
}
function matchSource() {
  const source = sourceResolution.value;
  if (!source) return;
  props.form.width = source.output.width;
  props.form.height = source.output.height;
}
function onResolution(mp: number) {
  const preset = resolutionPresets.value.find(
    (candidate) => (candidate.width * candidate.height) / 1_000_000 === mp,
  );
  if (preset) {
    props.form.width = preset.width;
    props.form.height = preset.height;
  }
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

// Same contract as the Advanced pane's Reset, surfaced without opening it:
// the prompt, the model, and any prepared batch size survive.
function resetSettings() {
  resetFormToModelDefaults(props.form, selectedModel.value);
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
          :show-availability="!stickyTarget || stickyTarget === 'capable'"
          :browse-target="caps.supportsVideo ? '/models?type=video' : '/models'"
          @pick="pickModel"
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

      <!-- Shape -->
      <div class="ms-field">
        <div class="ms-field__label">Shape</div>
        <ShapePicker
          :model-value="shapeId"
          :options="shapeOptions"
          label="Aspect ratio"
          @update:model-value="onShape"
        />
      </div>

      <!-- Resolution -->
      <div class="ms-field">
        <div class="ms-field__label">Resolution</div>
        <ResolutionSelector
          :model-value="resolutionMp"
          :ratio="resolutionRatio"
          :options="resolutionOptions"
          :resolved-width="form.width"
          :resolved-height="form.height"
          :custom-label="sourceStatus?.label"
          :status="
            sourceStatus
              ? followsSource
                ? sourceStatus.detail
                : `${sourceStatus.detail} · manual output`
              : undefined
          "
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
      </div>

      <!-- Detail (steps) -->
      <div class="ms-field">
        <SliderRow
          :model-value="form.steps"
          :min="1"
          :max="60"
          label="Detail"
          :value-label="`${form.steps} steps`"
          @update:model-value="form.steps = $event"
        />
        <p v-if="stepsError" class="ms-field__error" role="alert">{{ stepsError }}</p>
      </div>

      <!-- Prompt strength (guidance) -->
      <div class="ms-field">
        <SliderRow
          :model-value="form.guidance"
          :min="0"
          :max="12"
          :step="0.1"
          label="Prompt strength"
          :value-label="form.guidance.toFixed(1)"
          @update:model-value="form.guidance = $event"
        />
      </div>

      <!-- Frame rate — sequence output surfaces it outside Advanced -->
      <div v-if="isSequence" class="ms-field ms-field--row" data-test="sequence-fps">
        <span class="ms-field__label ms-field__label--inline">Frame rate</span>
        <Stepper
          :model-value="form.fps"
          :min="1"
          :max="60"
          label="Frames per second"
          :format="(v: number) => `${v} fps`"
          @update:model-value="form.fps = $event"
        />
      </div>

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
        :chain-limits="chainLimits"
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
        :upscalers="models.upscalers"
        :control-adapters="controlAdapters"
        :camera-controls="cameraControls"
        :camera-controls-loaded="cameraControlsLoaded"
        :camera-unsupported-reason="cameraUnsupportedReason"
        @append-word="emit('append-word', $event)"
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
