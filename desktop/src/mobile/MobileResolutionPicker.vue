<script setup lang="ts">
import { computed, ref, watch } from "vue";
import ShapePicker from "@ui/components/ShapePicker.vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import { resolutionValidationError, resolutionValidationWarning } from "../lib/generateValidation";
import { orientationLabel } from "../lib/resolutions";
import type { ModelEntry } from "../lib/api/types";
import { resolveSourceResolution, type SourceDimensions } from "@studio/lib/sourceResolution";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import {
  intentForCanvas,
  resolveOutputShape,
  sizeForFamily,
  SOURCE_FAMILY_ID,
  type CanvasIntent,
  type OutputShapeInput,
  type OutputShapeSize,
} from "@studio/lib/outputShape";
import { snapMobileDimension } from "./resolutionPicker";

const props = withDefaults(
  defineProps<{
    family: string;
    model?: ModelEntry | null;
    /** The user's explicit LTX-2 pipeline; a non-refining one lowers the ceiling. */
    pipeline?: string | null;
    sourceDimensions?: SourceDimensions | null;
    /** Why the canvas holds its current size — the shape resolver's authority. */
    canvasIntent?: CanvasIntent;
    disabled?: boolean;
  }>(),
  {
    disabled: false,
    model: null,
    pipeline: null,
    sourceDimensions: null,
    canvasIntent: "model-default",
  },
);

const width = defineModel<number>("width", { required: true });
const height = defineModel<number>("height", { required: true });
const emit = defineEmits<{
  "validity-change": [valid: boolean];
  "canvas-intent": [intent: CanvasIntent];
}>();

const manualOpen = ref(false);
const recipe = computed(() => effectiveGenerationRecipe(props.model, props.pipeline));
const alignment = computed(
  () => recipe.value?.resolution.alignment ?? props.model?.dimension_alignment ?? 16,
);
/** One resolver drives the chips, the pills, the badge and the sentence. */
const shapeInput = computed<OutputShapeInput>(() => ({
  model: props.model ?? null,
  family: props.family,
  pipeline: props.pipeline,
  width: width.value,
  height: height.value,
  source: props.sourceDimensions ?? null,
  intent: props.canvasIntent,
}));
const outputShape = computed(() => resolveOutputShape(shapeInput.value));
const currentOrientation = computed(() => orientationLabel(width.value, height.value));
const currentAspect = computed(() => outputShape.value.family.label);
/** The active family has authored sizes; a lone custom entry is not a ladder. */
const hasLadder = computed(() => outputShape.value.sizes.some((size) => size.tier !== "custom"));
const onLadder = computed(() =>
  outputShape.value.sizes.some(
    (size) => size.width === width.value && size.height === height.value && size.tier !== "custom",
  ),
);
const customVisible = computed(() => manualOpen.value || !onLadder.value);
const resolutionError = computed(() =>
  resolutionValidationError(width.value, height.value, props.model, props.pipeline),
);
/** Warn-policy bucket recipes admit this size but are not tuned for it. */
const resolutionWarning = computed(() =>
  resolutionValidationWarning(width.value, height.value, props.model, props.pipeline),
);
const sourceResolution = computed(() =>
  props.sourceDimensions
    ? resolveSourceResolution(props.sourceDimensions, props.model ?? props.family, props.pipeline)
    : null,
);
const followsSource = computed(
  () =>
    outputShape.value.state === "follows-source" || outputShape.value.state === "matches-source",
);
watch(resolutionError, (next) => emit("validity-change", !next), { immediate: true });
const shapeOptions = computed(() => outputShape.value.families);
const shapeId = computed(() => outputShape.value.selectedFamilyId);
const shapeApproximate = computed(() => outputShape.value.approximate);

function applyResolution(size: SourceDimensions | null): void {
  if (!size) return;
  width.value = size.width;
  height.value = size.height;
  manualOpen.value = false;
}

function setShape(id: string): void {
  const size = sizeForFamily(id, shapeInput.value);
  if (!size) return;
  emit("canvas-intent", id === SOURCE_FAMILY_ID ? "source" : "manual");
  applyResolution(size);
}

function setSize(id: string | number): void {
  const size = outputShape.value.sizes.find((candidate) => candidate.id === id);
  if (!size) return;
  emit("canvas-intent", intentForCanvas(shapeInput.value, size));
  applyResolution(size);
}

/** Pixels are the label; megapixels and any authored mark stay secondary. */
const sizeSegments = computed(() =>
  outputShape.value.sizes.map((size: OutputShapeSize) => ({
    value: size.id,
    label: size.label,
    sub: size.mark ? `${size.megapixels} · ${size.mark}` : size.megapixels,
  })),
);

function changedDimension(event: Event): number {
  return snapMobileDimension(Number((event.target as HTMLInputElement).value), 64, alignment.value);
}

function snapWidth(event: Event): void {
  emit("canvas-intent", "manual");
  width.value = changedDimension(event);
}

function snapHeight(event: Event): void {
  emit("canvas-intent", "manual");
  height.value = changedDimension(event);
}

function swapDimensions(): void {
  emit("canvas-intent", "manual");
  [width.value, height.value] = [height.value, width.value];
}

function matchSource(): void {
  const source = sourceResolution.value;
  if (!source) return;
  emit("canvas-intent", "source-exact");
  width.value = source.output.width;
  height.value = source.output.height;
  manualOpen.value = true;
}
</script>

<template>
  <fieldset class="mobile-resolution-picker" :disabled="disabled">
    <legend class="mobile-resolution-legend">Resolution</legend>

    <p
      class="sr-only"
      data-test="mobile-resolution-announcement"
      aria-live="polite"
      aria-atomic="true"
    >
      Selected resolution: {{ width }} by {{ height }} pixels, {{ currentAspect }},
      {{ currentOrientation }}.
    </p>

    <div
      v-if="sourceResolution"
      class="mobile-resolution-source"
      data-test="mobile-source-resolution-status"
      role="status"
    >
      <strong>{{ outputShape.badge }}</strong>
      <span>{{ outputShape.status }}</span>
      <button
        v-if="!followsSource"
        type="button"
        class="secondary-button"
        data-test="mobile-match-source-resolution"
        @click="matchSource"
      >
        Match source
      </button>
    </div>

    <div class="mobile-resolution-group">
      <span class="mobile-resolution-label">Shape</span>
      <ShapePicker
        :model-value="shapeId"
        :options="shapeOptions"
        :approximate="shapeApproximate"
        :disabled="disabled"
        label="Aspect ratio"
        data-test="mobile-resolution-shape"
        @update:model-value="setShape"
      />
    </div>

    <div v-if="hasLadder" class="mobile-resolution-group mobile-resolution-tier">
      <span class="mobile-resolution-label">Size</span>
      <SegmentedControl
        data-test="mobile-resolution-tier"
        :model-value="outputShape.selectedSizeId"
        :options="sizeSegments"
        label="Size"
        @update:model-value="setSize"
      />
      <p class="mobile-resolution-tier-dims" data-test="mobile-resolution-tier-dims">
        {{ width }} × {{ height }} px
      </p>
    </div>

    <button
      v-if="onLadder"
      type="button"
      class="secondary-button mobile-resolution-custom-toggle"
      data-test="mobile-resolution-custom-toggle"
      :aria-expanded="manualOpen"
      @click="manualOpen = !manualOpen"
    >
      {{ manualOpen ? "Hide custom size" : "Custom size" }}
    </button>

    <div v-if="customVisible" class="mobile-resolution-custom" data-test="mobile-resolution-custom">
      <label class="field">
        <span>Width</span>
        <input
          v-model.number="width"
          class="control"
          type="number"
          inputmode="numeric"
          min="64"
          :step="alignment"
          aria-label="Custom width"
          @change="snapWidth"
        />
      </label>
      <button
        type="button"
        class="secondary-button mobile-resolution-swap"
        aria-label="Swap width and height"
        @click="swapDimensions"
      >
        ⇄
      </button>
      <label class="field">
        <span>Height</span>
        <input
          v-model.number="height"
          class="control"
          type="number"
          inputmode="numeric"
          min="64"
          :step="alignment"
          aria-label="Custom height"
          @change="snapHeight"
        />
      </label>
    </div>
    <p v-if="customVisible" class="mobile-resolution-note">
      Custom dimensions snap to multiples of {{ alignment }} for model compatibility.
    </p>
    <p
      v-if="resolutionError"
      class="mobile-generate-validation"
      role="alert"
      data-test="mobile-resolution-error"
    >
      {{ resolutionError }}
    </p>
    <p
      v-else-if="resolutionWarning"
      class="mobile-resolution-note mobile-resolution-note--warning"
      data-test="mobile-resolution-warning"
    >
      {{ resolutionWarning }}
    </p>
  </fieldset>
</template>
