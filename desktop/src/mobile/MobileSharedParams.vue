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
import type { GenerateForm } from "../lib/generateForm";
import type { ModelEntry } from "../lib/api/types";
import MobileResolutionPicker from "./MobileResolutionPicker.vue";
import MobileSeedPicker from "./MobileSeedPicker.vue";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    model?: ModelEntry | null;
    lastSeed: number | null;
    disabled?: boolean;
    /** Sequence output surfaces frame rate outside the Advanced sheet. */
    showFps?: boolean;
    stepsError?: string | null;
    guidanceError?: string | null;
  }>(),
  { disabled: false, showFps: false, stepsError: null, guidanceError: null, model: null },
);

const emit = defineEmits<{
  "resolution-validity": [valid: boolean];
  "seed-validity": [valid: boolean];
}>();

const fpsError = computed(() => (props.showFps ? fpsValidationError(props.form.fps) : null));
</script>

<template>
  <MobileResolutionPicker
    v-model:width="form.width"
    v-model:height="form.height"
    :family="form.family"
    :model="model"
    :disabled="disabled"
    @validity-change="emit('resolution-validity', $event)"
  />
  <label v-if="showFps" class="field" data-test="mobile-sequence-fps">
    <span>FPS</span>
    <input
      v-model.number="form.fps"
      class="control"
      type="number"
      inputmode="numeric"
      min="1"
      max="60"
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
        min="1"
        max="100"
        :aria-invalid="stepsError ? 'true' : undefined"
      />
    </label>
    <label class="field"
      ><span>Guidance</span
      ><input
        v-model.number="form.guidance"
        class="control"
        type="number"
        inputmode="decimal"
        step="0.1"
        min="0"
        max="100"
        :aria-invalid="guidanceError ? 'true' : undefined"
    /></label>
  </div>
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
