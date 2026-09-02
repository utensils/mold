<script setup lang="ts">
/*
 * The generation params BOTH outputs read: size, frame rate, steps, guidance,
 * and seed. Extracted so One shot and Sequence render the identical controls
 * over the identical `GenerateForm` — the sequence bench borrows them through
 * its settings disclosure instead of keeping the private copies that used to
 * drift from what the user could see.
 */
import { computed } from "vue";
import { fpsValidationError, meshTargetFacesError } from "../lib/generateValidation";
import { buildRequest, type GenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";
import MobileSeedPicker from "./MobileSeedPicker.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import { emptyMeshForm } from "@studio/lib/meshControls";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { controlNote, effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import type { CanvasIntent } from "@studio/lib/outputShape";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    model?: ModelEntry | null;
    /** Timing metadata without changing legacy resolution projection behavior. */
    durationModel?: ModelEntry | null;
    lastSeed: number | null;
    disabled?: boolean;
    /** Sequence output surfaces frame rate outside the Advanced sheet. */
    showFps?: boolean;
    stepsError?: string | null;
    guidanceError?: string | null;
    canvasIntent?: CanvasIntent;
  }>(),
  {
    disabled: false,
    showFps: false,
    stepsError: null,
    guidanceError: null,
    model: null,
    durationModel: null,
    canvasIntent: "model-default",
  },
);

const emit = defineEmits<{
  "resolution-validity": [valid: boolean];
  "seed-validity": [valid: boolean];
  "canvas-intent": [intent: CanvasIntent];
}>();

const fpsError = computed(() => (props.showFps ? fpsValidationError(props.form.fps) : null));
const supportsVideo = computed(
  () =>
    generationCapabilitiesForFamily(
      props.form.family,
      props.form.model,
      props.form.pipeline,
      null,
      null,
      effectiveGenerationRecipe(props.model, props.form.pipeline),
    ).supportsVideo,
);
const durationCapabilityModel = computed(() => props.durationModel ?? props.model);
const canPredictDuration = computed(
  () =>
    !props.showFps &&
    durationCapabilityModel.value?.supports_duration_prediction === true &&
    durationCapabilityModel.value.runtime_ready !== false,
);
const guidanceCaps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.model?.guidance_capabilities,
    props.model?.source_image ?? props.form.sourceImageCapability,
    effectiveGenerationRecipe(props.model, props.form.pipeline),
  ),
);
const recipe = computed(() => effectiveGenerationRecipe(props.model, props.form.pipeline));
const stepsControl = computed(() => recipe.value?.steps);
const guidanceControl = computed(() => recipe.value?.guidance);
/* A fixed control explains itself with the host's own sentence, or with
 * nothing at all — the phone never composes copy for a value the profile
 * pinned, and an older host simply sends no note. */
const stepsNote = computed(() => controlNote(stepsControl.value));
const guidanceNote = computed(() => controlNote(guidanceControl.value));
const fpsControl = computed(() => recipe.value?.temporal?.fps);
/**
 * The 3-D controls, exactly as the recipe advertises them. The phone carries
 * no octree ladder, threshold range or face budget of its own: an absent
 * `mesh` block means the group does not render, and a control left untouched
 * stays `null` so `buildRequest` omits it and the engine's own default is
 * what renders (and what the print records).
 */
const meshCaps = computed(() => guidanceCaps.value.mesh ?? null);
/** Read-only view: a pre-mesh form restored without the slot reads as empty
 * here, and only a user edit (below) creates the slot on the owner's form —
 * a computed must never mutate its prop. */
const meshForm = computed(() => props.form.mesh ?? emptyMeshForm());
function writableMeshForm(): NonNullable<GenerateForm["mesh"]> {
  props.form.mesh ??= emptyMeshForm();
  return props.form.mesh;
}
const octreeSegments = computed(() =>
  (meshCaps.value?.octree_resolutions ?? []).map((resolution) => ({
    value: resolution,
    label: String(resolution),
  })),
);
/** The default is LIT while the control is untouched — the value that renders. */
const octreeSelected = computed(
  () => meshForm.value.octreeResolution ?? meshCaps.value?.octree_default ?? 0,
);
const thresholdControl = computed(() => meshCaps.value?.threshold ?? null);
const thresholdNote = computed(() => controlNote(thresholdControl.value));
const thresholdValue = computed(
  () => meshForm.value.threshold ?? thresholdControl.value?.default ?? 0,
);

function setOctreeResolution(value: string | number): void {
  writableMeshForm().octreeResolution = Number(value);
}

function setThreshold(event: Event): void {
  if (thresholdControl.value?.mode === "fixed") return;
  writableMeshForm().threshold = Number((event.target as HTMLInputElement).value);
}

/**
 * Blank is the raw decimated surface the engine produces on its own, which is
 * why this control is optional rather than pre-filled. A typed budget stands
 * as typed: one outside the advertised bounds is named inline (below) and
 * holds Develop, the way a bad steps value does, rather than being snapped
 * behind the user's back or shipped for the server to refuse.
 */
function setTargetFaces(event: Event): void {
  const raw = (event.target as HTMLInputElement).value.trim();
  if (!raw) {
    writableMeshForm().targetFaces = null;
    return;
  }
  const parsed = Math.round(Number(raw));
  writableMeshForm().targetFaces = Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}
const targetFacesError = computed(() =>
  meshTargetFacesError(meshForm.value.targetFaces, meshCaps.value),
);

const draft = useSequenceDraftStore();
const sourceDimensions = computed(() => {
  if (props.showFps) {
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
</script>

<template>
  <MobileResolutionPicker
    v-model:width="form.width"
    v-model:height="form.height"
    :family="form.family"
    :model="model"
    :pipeline="form.pipeline"
    :source-dimensions="sourceDimensions"
    :canvas-intent="canvasIntent"
    :disabled="disabled"
    @validity-change="emit('resolution-validity', $event)"
    @canvas-intent="emit('canvas-intent', $event)"
  />
  <div
    v-if="canPredictDuration"
    class="field mobile-predict-duration"
    data-test="mobile-predict-duration"
  >
    <span>Predict duration</span>
    <SwitchToggle
      :model-value="form.predictDuration"
      label="Predict duration from prompt"
      @update:model-value="form.predictDuration = $event"
    />
  </div>
  <VideoDurationSlider
    v-if="supportsVideo && !showFps && !form.predictDuration"
    class="mobile-duration-field"
    :frames="form.frames"
    :fps="form.fps"
    :model="durationModel ?? model"
    :family="form.family"
    :model-name="form.model"
    :source-image-capability="
      durationModel?.source_image ?? model?.source_image ?? form.sourceImageCapability
    "
    :routing-request="buildRequest(form)"
    touch-friendly
    data-test="mobile-duration"
    @update:frames="form.frames = $event"
  />
  <p
    v-else-if="supportsVideo && !showFps && form.predictDuration"
    class="field-hint"
    data-test="mobile-predicted-duration-hint"
  >
    The host will choose 1–20 seconds from the prompt.
  </p>
  <label v-if="showFps" class="field" data-test="mobile-sequence-fps">
    <span>FPS</span>
    <input
      v-model.number="form.fps"
      class="control"
      type="number"
      inputmode="numeric"
      :min="fpsControl?.mode === 'adjustable' ? fpsControl.min : 1"
      :max="fpsControl?.mode === 'adjustable' ? fpsControl.max : 60"
      :step="fpsControl?.mode === 'adjustable' ? fpsControl.step : 1"
      :disabled="fpsControl?.mode === 'fixed'"
      :aria-invalid="fpsError ? 'true' : undefined"
    />
  </label>
  <p
    v-if="fpsError"
    class="mobile-generate-validation"
    role="alert"
    data-test="mobile-sequence-fps-error"
  >
    {{ fpsError }}
  </p>
  <div class="field-grid">
    <label class="field" :class="{ 'field--with-note': stepsNote }">
      <span>Steps</span>
      <input
        v-model.number="form.steps"
        class="control"
        type="number"
        inputmode="numeric"
        :min="stepsControl?.min ?? 1"
        :max="stepsControl?.max ?? 100"
        :step="stepsControl?.step ?? 1"
        :disabled="stepsControl?.mode === 'fixed'"
        :aria-invalid="stepsError ? 'true' : undefined"
      />
      <small v-if="stepsNote" class="mobile-generate-hint" data-test="mobile-fixed-steps-hint">
        {{ stepsNote }}
      </small>
    </label>
    <label class="field" :class="{ 'field--with-note': guidanceNote }">
      <span>Guidance</span>
      <input
        :value="guidanceCaps.fixedGuidance ?? form.guidance"
        class="control"
        type="number"
        inputmode="decimal"
        :step="guidanceControl?.step ?? 0.1"
        :min="guidanceControl?.min ?? 0"
        :max="guidanceControl?.max ?? 100"
        :disabled="guidanceControl?.mode === 'fixed' || !guidanceCaps.guidanceAdjustable"
        :aria-invalid="guidanceError ? 'true' : undefined"
        @input="
          guidanceCaps.guidanceAdjustable &&
          (form.guidance = Number(($event.target as HTMLInputElement).value))
        "
      />
      <small
        v-if="guidanceNote"
        class="mobile-generate-hint"
        data-test="mobile-fixed-guidance-hint"
      >
        {{ guidanceNote }}
      </small>
    </label>
  </div>
  <p
    v-if="stepsError || guidanceError"
    class="mobile-generate-validation"
    role="alert"
    data-test="mobile-basic-parameter-error"
  >
    {{ stepsError || guidanceError }}
  </p>
  <!-- 3-D: rendered only for a recipe that advertises a `mesh` block, which
       is also the only request the server accepts `mesh` on. -->
  <fieldset
    v-if="meshCaps"
    class="mobile-mesh-controls"
    :disabled="disabled"
    data-test="mobile-mesh-controls"
  >
    <legend class="mobile-mesh-legend">Mesh</legend>
    <div v-if="octreeSegments.length" class="mobile-mesh-group">
      <span class="mobile-mesh-label">Octree detail</span>
      <SegmentedControl
        wrap
        data-test="mobile-mesh-octree"
        :model-value="octreeSelected"
        :options="octreeSegments"
        label="Octree detail"
        :disabled="disabled"
        @update:model-value="setOctreeResolution"
      />
    </div>
    <label
      v-if="thresholdControl"
      class="mobile-range-field"
      :class="{ 'field--with-note': thresholdNote }"
    >
      <span
        >Iso threshold <output>{{ thresholdValue.toFixed(2) }}</output></span
      >
      <input
        type="range"
        :value="thresholdValue"
        :min="thresholdControl.min"
        :max="thresholdControl.max"
        :step="thresholdControl.step"
        :disabled="thresholdControl.mode === 'fixed'"
        aria-label="Iso threshold"
        data-test="mobile-mesh-threshold"
        @input="setThreshold"
      />
      <small
        v-if="thresholdNote"
        class="mobile-generate-hint"
        data-test="mobile-mesh-threshold-note"
      >
        {{ thresholdNote }}
      </small>
    </label>
    <label class="field">
      <span>Target faces</span>
      <input
        class="control"
        type="number"
        inputmode="numeric"
        placeholder="Leave blank for the raw surface"
        :value="meshForm.targetFaces ?? ''"
        :min="meshCaps.target_faces_min"
        :max="meshCaps.target_faces_max"
        step="1"
        :aria-invalid="targetFacesError ? 'true' : undefined"
        data-test="mobile-mesh-target-faces"
        @change="setTargetFaces"
      />
      <small class="mobile-generate-hint">
        Optional — decimates to this budget, between
        {{ meshCaps.target_faces_min.toLocaleString("en-US") }} and
        {{ meshCaps.target_faces_max.toLocaleString("en-US") }} triangles.
      </small>
    </label>
    <p
      v-if="targetFacesError"
      class="mobile-generate-validation"
      role="alert"
      data-test="mobile-mesh-target-faces-error"
    >
      {{ targetFacesError }}
    </p>
  </fieldset>
  <MobileSeedPicker
    :model-value="form.seed"
    :last-seed="lastSeed"
    @update:model-value="form.seed = $event"
    @validity-change="emit('seed-validity', $event)"
  />
</template>
