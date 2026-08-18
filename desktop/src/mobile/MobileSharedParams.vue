<script setup lang="ts">
/*
 * The generation params BOTH outputs read: size, frame rate, steps, guidance,
 * and seed. Extracted so One shot and Sequence render the identical controls
 * over the identical `GenerateForm` — the sequence bench borrows them through
 * its settings disclosure instead of keeping the private copies that used to
 * drift from what the user could see.
 */
import { computed } from "vue";
import { fpsValidationError } from "../lib/generateValidation";
import { buildRequest, type GenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";
import MobileSeedPicker from "./MobileSeedPicker.vue";
import VideoDurationSlider from "@ui/components/VideoDurationSlider.vue";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";

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
  }>(),
  {
    disabled: false,
    showFps: false,
    stepsError: null,
    guidanceError: null,
    model: null,
    durationModel: null,
  },
);

const emit = defineEmits<{
  "resolution-validity": [valid: boolean];
  "seed-validity": [valid: boolean];
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
const fpsControl = computed(() => recipe.value?.temporal?.fps);
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
    :disabled="disabled"
    @validity-change="emit('resolution-validity', $event)"
  />
  <VideoDurationSlider
    v-if="supportsVideo && !showFps"
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
    <label class="field">
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
    </label>
    <label class="field"
      ><span>Guidance</span
      ><input
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
    /></label>
  </div>
  <p
    v-if="!guidanceCaps.guidanceAdjustable"
    class="mobile-generate-hint"
    data-test="mobile-fixed-guidance-hint"
  >
    Distilled recipe fixes CFG at 1.0. Choose a Dev checkpoint with Auto or a guided pipeline to
    adjust it.
  </p>
  <p
    v-if="stepsError || guidanceError"
    class="mobile-generate-validation"
    role="alert"
    data-test="mobile-basic-parameter-error"
  >
    {{ stepsError || guidanceError }}
  </p>
  <MobileSeedPicker
    :model-value="form.seed"
    :last-seed="lastSeed"
    @update:model-value="form.seed = $event"
    @validity-change="emit('seed-validity', $event)"
  />
</template>
