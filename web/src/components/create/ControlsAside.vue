<script setup lang="ts">
/*
 * Controls aside (Mold Studio Create) — the right rail. width/height stay the
 * persisted source of truth (see `useGenerateForm`); Shape and Resolution are
 * PROJECTIONS of those pixels. Detail (steps) and Prompt strength (guidance)
 * are direct sliders, Seed exposes Random/Fixed (increment stays reachable in
 * Advanced → Output & seed), Batch is a stepper, and the Advanced button
 * surfaces the "N on" badge and opens the drawer.
 */
import { computed } from "vue";
import { useRouter } from "vue-router";
import { conditioningForRequest } from "@studio/lib/sourceMediaPlan";
import ShapePicker from "@ui/components/ShapePicker.vue";
import ResolutionSelector from "@ui/components/ResolutionSelector.vue";
import SliderRow from "@ui/components/SliderRow.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import Stepper from "@ui/components/Stepper.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import BadgePill from "@ui/components/BadgePill.vue";
import Icon from "@ui/components/Icon.vue";
import {
  controlNote,
  effectiveGenerationRecipe,
  resolutionProfileFinding,
} from "@studio/lib/generationProfile";
import { emptyMeshForm, type MeshFormState } from "@studio/lib/meshControls";
import type { GenerateFormState, ModelInfoExtended } from "../../types";
import type { GenerateRoutingRequest } from "@studio/lib/chainRouting";
import { generationCapabilitiesForFamily } from "../../lib/generateCapabilities";
import {
  intentForCanvas,
  resolveOutputShape,
  sizeForFamily,
  SOURCE_FAMILY_ID,
  type CanvasIntent,
  type OutputShapeInput,
} from "@studio/lib/outputShape";
import {
  resolveSourceResolution,
  type SourceDimensions,
} from "@studio/lib/sourceResolution";
import HostRoutingPicker from "./HostRoutingPicker.vue";
import { useHostRouting } from "../../composables/useHostRouting";

const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    family: string;
    model?: ModelInfoExtended | null;
    sourceDimensions?: SourceDimensions | null;
    /** Why the canvas holds its current size — the shape resolver's authority. */
    canvasIntent?: CanvasIntent;
    /** Count of active advanced fields (drives the badge). */
    advCount?: number;
    /** Phone surface: the Advanced sheet button shows here; on tablet+ the
     * Advanced sections render inline in the controls region instead. */
    mobile?: boolean;
    /** Seed of the most recent finished print — powers "lock last seed"
     * (desktop InspectorPanel parity). */
    lastSeed?: number | null;
    routingRequest?: Partial<GenerateRoutingRequest> | null | undefined;
  }>(),
  {
    advCount: 0,
    mobile: false,
    lastSeed: null,
    model: null,
    sourceDimensions: null,
    canvasIntent: "model-default",
  },
);

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  "open-advanced": [];
  /* The rail only knows the form, not the catalog row behind it, so the page
   * owns the actual reset (model defaults + the undo offer). */
  "reset-settings": [];
  "canvas-intent": [intent: CanvasIntent];
}>();

// The fifth argument is the selected row's advertised source-image contract
// (#772). The rail does not render a source well itself, but resolving the
// same five inputs as the drawer and the submit gate is what stops the three
// from disagreeing about the selected checkpoint.
const capabilities = computed(() =>
  generationCapabilitiesForFamily(
    props.family,
    props.model?.name ?? props.modelValue.model,
    props.modelValue.pipeline,
    props.model?.guidance_capabilities,
    props.model?.source_image ?? props.modelValue.sourceImageCapability,
    effectiveGenerationRecipe(props.model, props.modelValue.pipeline),
  ),
);
const activeRecipe = computed(() =>
  effectiveGenerationRecipe(props.model, props.modelValue.pipeline),
);
/** A fixed control explains itself with the server's own sentence, or with
 * nothing at all — this surface never composes copy for a value the profile
 * pinned (an older server sends no note, and H3 is not a distilled FLUX). */
const stepsNote = computed(() => controlNote(activeRecipe.value?.steps));
const guidanceNote = computed(() => controlNote(activeRecipe.value?.guidance));

// ── 3-D mesh (Hunyuan3D) ──────────────────────────────────────────────
// Every control below is built from the recipe's own `capabilities.mesh`
// block — the octree ladder, the iso-threshold bounds and the face bounds are
// the server's, never a client constant, so a host that widens them widens
// the rail with no client release.
/** The recipe renders no pixel canvas: Shape and Resolution have nothing to
 * bind to and are hidden rather than steering a size the request ignores. */
const canvasless = computed(
  () => capabilities.value.canvasless || outputShape.value.canvasless,
);
const meshProfile = computed(() => capabilities.value.mesh ?? null);
const meshForm = computed(() => props.modelValue.mesh ?? emptyMeshForm());
const octreeOptions = computed(() =>
  (meshProfile.value?.octree_resolutions ?? []).map((value) => ({
    value,
    label: String(value),
  })),
);
const octreeValue = computed(
  () =>
    meshForm.value.octreeResolution ?? meshProfile.value?.octree_default ?? 0,
);
const thresholdControl = computed(() => meshProfile.value?.threshold ?? null);
const thresholdValue = computed(
  () => meshForm.value.threshold ?? thresholdControl.value?.default ?? 0,
);
const thresholdNote = computed(() => controlNote(thresholdControl.value));
const targetFacesValue = computed(() => meshForm.value.targetFaces);
/** Advisory only, like the resolution warning: the server is the authority
 * and refuses an out-of-range value at admission (422), so the rail says so
 * here instead of letting Generate fail with no explanation. */
const targetFacesWarning = computed(() => {
  const profile = meshProfile.value;
  const value = targetFacesValue.value;
  if (!profile || value === null || value === undefined) return null;
  if (value >= profile.target_faces_min && value <= profile.target_faces_max) {
    return null;
  }
  const count = (n: number) => n.toLocaleString("en-US");
  return `${count(value)} is outside this model's ${count(profile.target_faces_min)}–${count(profile.target_faces_max)} face range; the host will refuse it.`;
});
function patchMesh(next: Partial<MeshFormState>) {
  patch({ mesh: { ...meshForm.value, ...next } });
}
function setTargetFaces(raw: string) {
  const trimmed = raw.trim();
  if (!trimmed) {
    patchMesh({ targetFaces: null });
    return;
  }
  const value = Number(trimmed);
  patchMesh({
    targetFaces: Number.isFinite(value) && value > 0 ? Math.round(value) : null,
  });
}
/** Every size constraint is advisory — the server is the authority, so a
 * custom size renders a warning here rather than blocking Generate. */
const resolutionWarning = computed(() => {
  const finding = resolutionProfileFinding(
    props.modelValue.width,
    props.modelValue.height,
    activeRecipe.value?.resolution,
  );
  return finding?.level === "warn" ? finding.message : null;
});
const canPredictDuration = computed(
  () =>
    props.model?.supports_duration_prediction === true &&
    props.model.runtime_ready !== false,
);
const predictDuration = computed(
  () => props.modelValue.predictDuration === true,
);
function setPredictDuration(value: boolean) {
  patch({
    predictDuration: value,
    frames: value
      ? null
      : (props.modelValue.frames ?? props.model?.default_frames ?? 25),
  });
}
const showGenerateAudio = computed(() => capabilities.value.offersAudioControl);
const generateAudio = computed(() => props.modelValue.enableAudio !== false);
const audioOutputSupported = computed(
  () =>
    capabilities.value.supportsAudio && props.model?.supports_audio !== false,
);
const audioOutputUnavailableReason = computed(() => {
  if (!showGenerateAudio.value || audioOutputSupported.value) return null;
  if (props.model?.supports_audio === false) {
    return "Audio assets are not included with this checkpoint. Video generation remains available.";
  }
  return (
    capabilities.value.outputDeliveryReason ??
    "Generated audio is unavailable for this recipe."
  );
});
function setGenerateAudio(value: boolean) {
  patch({ enableAudio: value });
}
// Edit families (Qwen image edit) render one print at a time.
const batchLocked = computed(
  () =>
    capabilities.value.forcesBatchSizeOne ||
    conditioningForRequest(capabilities.value.sourceImageMode, {
      hasSource: Boolean(props.modelValue.imageAttachments[0]?.base64),
      referenceCount:
        capabilities.value.sourceImageMode === "single-or-references"
          ? (props.modelValue.referenceImages?.length ?? 0)
          : props.modelValue.imageAttachments.length,
      lastWrite: props.modelValue.exclusiveWell ?? null,
    }) === "references",
);

// Reroll: a fresh random seed for the next print without leaving Fixed mode —
// mirrors the desktop inspector's reroll. Switches to Random so the server
// draws a new seed each generate.
function reroll() {
  patch({ seedMode: "random", seed: null });
}

// Generation target (spec §08 multi-host). The picker owns the routing choice —
// Auto, Most capable, or a sticky host — and the submit path in CreatePage
// resolves the same persisted pick through the same singleton, so what the row
// claims and where the job lands can't drift apart.
const router = useRouter();
const routing = useHostRouting();
function openMachines() {
  void router?.push("/machines");
}

const sourceResolution = computed(() =>
  props.sourceDimensions
    ? resolveSourceResolution(
        props.sourceDimensions,
        props.model ?? props.family,
        props.modelValue.pipeline,
      )
    : null,
);
/** One resolver drives the chips, the pills, the badge and the sentence. */
const shapeInput = computed<OutputShapeInput>(() => ({
  model: props.model ?? null,
  family: props.family,
  pipeline: props.modelValue.pipeline,
  width: props.modelValue.width,
  height: props.modelValue.height,
  source: props.sourceDimensions ?? null,
  intent: props.canvasIntent,
}));
const outputShape = computed(() => resolveOutputShape(shapeInput.value));
const followsSource = computed(
  () =>
    outputShape.value.state === "follows-source" ||
    outputShape.value.state === "matches-source",
);
const shapeOptions = computed(() => outputShape.value.families);
const aspectId = computed(() => outputShape.value.selectedFamilyId);
const aspectApproximate = computed(() => outputShape.value.approximate);
const currentRatio = computed(() => {
  const { width, height } = props.modelValue;
  return height ? width / height : 1;
});

function patch(patch: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...patch });
}

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
const selectedSizeId = computed(() => outputShape.value.selectedSizeId);

function setAspect(id: string) {
  const size = sizeForFamily(id, shapeInput.value);
  if (!size) return;
  emit("canvas-intent", id === SOURCE_FAMILY_ID ? "source" : "manual");
  patch({ width: size.width, height: size.height });
}

function matchSource() {
  const source = sourceResolution.value;
  if (!source) return;
  emit("canvas-intent", "source-exact");
  patch({ width: source.output.width, height: source.output.height });
}

function setSize(id: string | number) {
  const size = outputShape.value.sizes.find((candidate) => candidate.id === id);
  if (!size) return;
  emit("canvas-intent", intentForCanvas(shapeInput.value, size));
  patch({ width: size.width, height: size.height });
}

// Seed: Random / Fixed segments. "Fixed" covers the persisted static and
// increment modes so an increment set in Advanced survives a re-click.
const seedSegments = [
  { value: "random", label: "Random" },
  { value: "fixed", label: "Fixed" },
] as const;
const seedSegment = computed(() =>
  props.modelValue.seedMode === "random" ? "random" : "fixed",
);
function setSeedSegment(v: "random" | "fixed") {
  if (v === "random") {
    patch({ seedMode: "random" });
  } else if (props.modelValue.seedMode === "random") {
    patch({ seedMode: "static" });
  }
}
function setSeed(value: number) {
  patch({ seed: Number.isFinite(value) ? value : null });
}
// Lock last seed (desktop parity): pin the previous print's seed so the next
// generate reproduces it. Switching to the Fixed segment reveals the input.
function lockLastSeed() {
  if (props.lastSeed === null) return;
  patch({ seedMode: "static", seed: props.lastSeed });
}
</script>

<template>
  <aside class="controls" data-test="controls-aside">
    <div class="controls__head">
      <span class="controls__kicker">Settings</span>
      <button
        type="button"
        class="controls__reset"
        data-test="settings-reset"
        aria-label="Reset settings to model defaults"
        title="Reset settings to model defaults"
        @click="emit('reset-settings')"
      >
        ↺ Reset
      </button>
    </div>

    <div v-if="!canvasless" class="controls__group">
      <div class="controls__label">Shape</div>
      <ShapePicker
        :model-value="aspectId"
        :options="shapeOptions"
        :approximate="aspectApproximate"
        label="Shape"
        @update:model-value="setAspect"
      />
    </div>

    <div v-if="!canvasless" class="controls__group">
      <div class="controls__label">Resolution</div>
      <ResolutionSelector
        :model-value="selectedSizeId"
        :ratio="currentRatio"
        :options="resolutionOptions"
        :resolved-width="modelValue.width"
        :resolved-height="modelValue.height"
        :custom-label="sourceResolution ? outputShape.badge : undefined"
        :status="outputShape.status"
        @update:model-value="setSize"
      />
      <button
        v-if="sourceResolution && !followsSource"
        type="button"
        class="controls__match-source"
        data-test="match-source-resolution"
        @click="matchSource"
      >
        Match source
      </button>
      <p
        v-if="resolutionWarning"
        class="controls__hint controls__hint--warning"
        data-test="resolution-warning"
      >
        {{ resolutionWarning }}
      </p>
    </div>

    <div class="controls__group">
      <SliderRow
        label="Detail"
        :model-value="modelValue.steps"
        :min="activeRecipe?.steps.min ?? 1"
        :max="activeRecipe?.steps.max ?? 100"
        :step="activeRecipe?.steps.step ?? 1"
        :disabled="activeRecipe?.steps.mode === 'fixed'"
        :value-label="`${modelValue.steps} steps`"
        @update:model-value="patch({ steps: $event })"
      />
      <p v-if="stepsNote" class="controls__hint" data-test="fixed-steps-hint">
        {{ stepsNote }}
      </p>
    </div>

    <!-- 3-D geometry. Built entirely from the recipe's advertised `mesh`
         block, so a host that widens the octree ladder or the face bounds
         widens this group with no client release. -->
    <div v-if="meshProfile" class="controls__group" data-test="mesh-controls">
      <div class="controls__label">Mesh</div>
      <SegmentedControl
        v-if="octreeOptions.length > 0"
        data-test="mesh-octree"
        wrap
        :model-value="octreeValue"
        :options="octreeOptions"
        label="Octree detail"
        @update:model-value="patchMesh({ octreeResolution: $event })"
      />
      <SliderRow
        v-if="thresholdControl"
        class="controls__mesh-slider"
        label="Iso threshold"
        :model-value="thresholdValue"
        :min="thresholdControl.min"
        :max="thresholdControl.max"
        :step="thresholdControl.step"
        :disabled="thresholdControl.mode === 'fixed'"
        :value-label="thresholdValue.toFixed(2)"
        @update:model-value="patchMesh({ threshold: $event })"
      />
      <p
        v-if="thresholdNote"
        class="controls__hint"
        data-test="mesh-threshold-note"
      >
        {{ thresholdNote }}
      </p>
      <div class="controls__mesh-faces">
        <label class="controls__label controls__label--inline" for="mesh-faces"
          >Target faces</label
        >
        <input
          id="mesh-faces"
          class="controls__seed"
          data-test="mesh-target-faces"
          type="number"
          :min="meshProfile.target_faces_min"
          :max="meshProfile.target_faces_max"
          placeholder="keep raw surface"
          :value="targetFacesValue ?? ''"
          @input="setTargetFaces(($event.target as HTMLInputElement).value)"
        />
      </div>
      <p class="controls__hint">
        Leave blank to keep the raw surface —
        {{ meshProfile.target_faces_min }}–{{ meshProfile.target_faces_max }}
        triangles when decimating.
      </p>
      <p
        v-if="targetFacesWarning"
        class="controls__hint controls__hint--warning"
        data-test="mesh-target-faces-warning"
      >
        {{ targetFacesWarning }}
      </p>
    </div>

    <div v-if="capabilities.supportsVideo" class="controls__group">
      <div
        v-if="canPredictDuration"
        class="controls__toggle"
        data-test="predict-duration-control"
      >
        <span class="controls__label controls__label--inline"
          >Predict duration</span
        >
        <SwitchToggle
          :model-value="predictDuration"
          label="Predict duration from prompt"
          @update:model-value="setPredictDuration"
        />
      </div>
      <VideoDurationSlider
        v-if="!predictDuration || !canPredictDuration"
        :frames="modelValue.frames ?? model?.default_frames ?? 25"
        :fps="modelValue.fps ?? model?.default_fps ?? 24"
        :model="model"
        :family="family"
        :model-name="modelValue.model"
        :source-image-capability="
          model?.source_image ?? modelValue.sourceImageCapability
        "
        :routing-request="routingRequest"
        @update:frames="patch({ frames: $event })"
      />
      <p
        v-else-if="canPredictDuration"
        class="controls__hint"
        data-test="predicted-duration-hint"
      >
        The host will choose 1–20 seconds from the prompt.
      </p>
    </div>

    <div
      v-if="showGenerateAudio"
      class="controls__group controls__toggle"
      data-test="generate-audio-control"
    >
      <span class="controls__label controls__label--inline"
        >Generate audio</span
      >
      <SwitchToggle
        :model-value="generateAudio"
        :disabled="!audioOutputSupported"
        label="Generate audio"
        @update:model-value="setGenerateAudio"
      />
      <p
        v-if="audioOutputUnavailableReason"
        class="controls__hint controls__hint--full"
      >
        {{ audioOutputUnavailableReason }}
      </p>
    </div>

    <div class="controls__group">
      <SliderRow
        label="Prompt strength"
        :model-value="capabilities.fixedGuidance ?? modelValue.guidance"
        :min="activeRecipe?.guidance.min ?? 0"
        :max="activeRecipe?.guidance.max ?? 100"
        :step="activeRecipe?.guidance.step ?? 0.1"
        :value-label="
          (capabilities.fixedGuidance ?? modelValue.guidance).toFixed(1)
        "
        :disabled="
          activeRecipe?.guidance.mode === 'fixed' ||
          !capabilities.guidanceAdjustable
        "
        @update:model-value="patch({ guidance: $event })"
      />
      <p
        v-if="guidanceNote"
        class="controls__hint"
        data-test="fixed-guidance-hint"
      >
        {{ guidanceNote }}
      </p>
    </div>

    <div class="controls__group">
      <div class="controls__seed-head">
        <span class="controls__label controls__label--inline">Seed</span>
        <button
          type="button"
          class="controls__reroll"
          data-test="seed-reroll"
          title="New random seed next print"
          @click="reroll"
        >
          <Icon name="reroll" :size="13" />
          reroll
        </button>
      </div>
      <SegmentedControl
        data-test="seed-seg"
        :model-value="seedSegment"
        :options="seedSegments"
        label="Seed mode"
        @update:model-value="setSeedSegment"
      />
      <input
        v-if="seedSegment === 'fixed'"
        class="controls__seed"
        data-test="controls-seed"
        type="number"
        min="0"
        placeholder="Seed"
        :value="modelValue.seed ?? ''"
        @input="setSeed(Number(($event.target as HTMLInputElement).value))"
      />
      <p v-if="seedSegment === 'random'" class="controls__hint">
        New seed every print<template v-if="lastSeed !== null">
          ·
          <button
            type="button"
            data-test="lock-last-seed"
            class="controls__lock"
            @click="lockLastSeed"
          >
            lock last ({{ lastSeed }})
          </button></template
        >
      </p>
    </div>

    <div class="controls__group">
      <div class="controls__batch">
        <span class="controls__label controls__label--inline">Batch</span>
        <Stepper
          :model-value="batchLocked ? 1 : modelValue.batchSize"
          :min="1"
          :max="batchLocked ? 1 : 10_000"
          editable
          label="Batch size"
          @update:model-value="patch({ batchSize: $event })"
        />
      </div>
      <p v-if="batchLocked" class="controls__hint" data-test="batch-locked">
        locked to 1 — edit models render one print at a time.
      </p>
    </div>

    <!-- "File under" (Create-time Library organization): after the
         essentials, above Advanced on every width. The page owns the state
         and the capability gate; the rail owns only its position, so the
         phone sheet and the tablet+ column can't order it differently. -->
    <slot name="file-under" />

    <button
      v-if="mobile"
      type="button"
      class="controls__advanced"
      data-test="open-advanced"
      @click="emit('open-advanced')"
    >
      <Icon name="sliders" :size="14" />
      Advanced
      <BadgePill v-if="advCount > 0" data-test="adv-badge"
        >{{ advCount }} on</BadgePill
      >
    </button>

    <HostRoutingPicker
      :hosts="routing.hosts.value"
      :target-id="routing.targetId.value"
      @select="routing.setTarget"
      @open-machines="openMachines"
    />
  </aside>
</template>

<style scoped>
.controls {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 18px;
}

.controls__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
  margin-bottom: 16px;
}

.controls__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}

/* Matches the Advanced card's inline Reset pill so the two read as one
 * family of "put this section back" actions. */
.controls__reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 4px 11px;
  border-radius: var(--radius-pill);
  font-size: 11.5px;
  font-weight: 600;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}
.controls__reset:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}

.controls__group {
  margin-bottom: 20px;
}

.controls__label {
  font-size: 12px;
  color: var(--ink-2);
  margin-bottom: 9px;
  font-weight: 600;
}

.controls__label--inline {
  margin-bottom: 0;
}

.controls__toggle {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 8px 12px;
}
.controls__hint--full {
  grid-column: 1 / -1;
}

.controls__seed {
  width: 100%;
  box-sizing: border-box;
  margin-top: 9px;
  height: 38px;
  background: var(--bath);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 13px;
  padding: 0 12px;
  outline: none;
}

.controls__batch {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.controls__mesh-slider {
  margin-top: 12px;
}

.controls__mesh-faces {
  margin-top: 12px;
}

.controls__seed-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 9px;
}

.controls__reroll {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  cursor: pointer;
  padding: 0;
}
.controls__reroll:hover {
  color: var(--safelight);
}

.controls__hint {
  margin: 8px 0 0;
  font-size: 11px;
  color: var(--ink-3);
  line-height: 1.4;
}

.controls__hint--warning {
  color: var(--safelight);
}

.controls__match-source {
  margin-top: 7px;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-size: 11px;
  cursor: pointer;
}
.controls__match-source:hover {
  text-decoration: underline;
}

.controls__lock {
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
  text-decoration: underline;
  text-underline-offset: 2px;
  cursor: pointer;
  padding: 0;
}
.controls__lock:hover {
  color: var(--safelight);
}

.controls__advanced {
  width: 100%;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 11px;
  border-radius: var(--radius-control);
  font-size: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 16px;
  cursor: pointer;
}
</style>
