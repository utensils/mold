<script setup lang="ts">
import { computed, ref, watch } from "vue";
import SegmentedControl from "@ui/components/SegmentedControl.vue";
import {
  resolutionValidationError,
  resolutionValidationWarning,
} from "../lib/generateValidation";
import {
  aspectRatioLabel,
  matchPreset,
  orientationLabel,
  presetsForModel,
  type ResolutionPreset,
} from "../lib/resolutions";
import type { ModelEntry } from "../lib/api/types";
import {
  canvasMatchesSourceResolution,
  resolveSourceResolution,
  sourceResolutionStatus,
  type SourceDimensions,
} from "@studio/lib/sourceResolution";
import { effectiveGenerationRecipe } from "@studio/lib/generationProfile";
import {
  aspectsForOrientation,
  closestResolutionPreset,
  MOBILE_RESOLUTION_ORIENTATIONS,
  presetsForOrientation,
  resolutionTierLabel,
  snapMobileDimension,
  sortedResolutionTiers,
  type ResolutionOrientation,
} from "./resolutionPicker";

const props = withDefaults(
  defineProps<{
    family: string;
    model?: ModelEntry | null;
    /** The user's explicit LTX-2 pipeline; a non-refining one lowers the ceiling. */
    pipeline?: string | null;
    sourceDimensions?: SourceDimensions | null;
    disabled?: boolean;
  }>(),
  { disabled: false, model: null, pipeline: null, sourceDimensions: null },
);

const width = defineModel<number>("width", { required: true });
const height = defineModel<number>("height", { required: true });
const emit = defineEmits<{ "validity-change": [valid: boolean] }>();

const manualOpen = ref(false);
const recipe = computed(() => effectiveGenerationRecipe(props.model, props.pipeline));
const alignment = computed(
  () => recipe.value?.resolution.alignment ?? props.model?.dimension_alignment ?? 16,
);
const presets = computed(() => presetsForModel(props.model ?? props.family, props.pipeline));
const currentPreset = computed(() =>
  matchPreset(width.value, height.value, props.model ?? props.family, props.pipeline),
);
const currentOrientation = computed(() => orientationLabel(width.value, height.value));
const currentAspect = computed(() =>
  aspectRatioLabel(width.value, height.value, props.model ?? props.family, props.pipeline),
);
const customVisible = computed(() => manualOpen.value || !currentPreset.value);
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
    sourceResolution.value !== null &&
    canvasMatchesSourceResolution(
      { width: width.value, height: height.value },
      sourceResolution.value,
    ),
);
const sourceStatus = computed(() =>
  sourceResolution.value ? sourceResolutionStatus(sourceResolution.value) : null,
);
watch(resolutionError, (next) => emit("validity-change", !next), { immediate: true });
const aspectOptions = computed(() =>
  aspectsForOrientation(presets.value, currentOrientation.value),
);
const tierOptions = computed(() =>
  currentPreset.value ? sortedResolutionTiers(presets.value, currentPreset.value.aspect) : [],
);

function proportionalShapeStyle(
  shapeWidth: number,
  shapeHeight: number,
  maxWidth: number,
  maxHeight: number,
): Record<string, string> {
  const safeWidth = Math.max(1, shapeWidth);
  const safeHeight = Math.max(1, shapeHeight);
  const scale = Math.min(maxWidth / safeWidth, maxHeight / safeHeight);
  return {
    width: `${(safeWidth * scale).toFixed(2)}px`,
    height: `${(safeHeight * scale).toFixed(2)}px`,
  };
}

function aspectDimensions(aspect: string): [number, number] {
  const preset = presets.value.find((candidate) => candidate.aspect === aspect);
  if (preset) return [preset.width, preset.height];
  const match = aspect.replace(/^≈/, "").match(/^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/);
  if (!match) return [1, 1];
  return [Number(match[1]), Number(match[2])];
}

function aspectOptionShapeStyle(aspect: string): Record<string, string> {
  const [shapeWidth, shapeHeight] = aspectDimensions(aspect);
  return proportionalShapeStyle(shapeWidth, shapeHeight, 30, 28);
}

function aspectAccessibleLabel(aspect: string): string {
  const approximate = aspect.startsWith("≈");
  const [left, right] = aspect.replace(/^≈/, "").split(":");
  return `${approximate ? "Approximately " : ""}${left} by ${right} aspect ratio`;
}

function isOrientationAvailable(orientation: ResolutionOrientation): boolean {
  return presetsForOrientation(presets.value, orientation).length > 0;
}

function applyResolution(preset: ResolutionPreset | null): void {
  if (!preset) return;
  width.value = preset.width;
  height.value = preset.height;
  manualOpen.value = false;
}

function setOrientation(orientation: ResolutionOrientation): void {
  applyResolution(
    closestResolutionPreset(
      presetsForOrientation(presets.value, orientation),
      width.value,
      height.value,
    ),
  );
}

function setAspect(aspect: string): void {
  applyResolution(
    closestResolutionPreset(
      presets.value.filter((preset) => preset.aspect === aspect),
      width.value,
      height.value,
    ),
  );
}

function setTier(label: string): void {
  applyResolution(tierOptions.value.find((preset) => preset.label === label) ?? null);
}

/** "0.6 MP" with the web treatment's trailing-.0 strip ("1.0" → "1 MP"). */
function tierMegapixels(preset: ResolutionPreset): string {
  const megapixels = ((preset.width * preset.height) / 1_000_000).toFixed(1).replace(/\.0$/, "");
  return `${megapixels} MP`;
}

const tierSegments = computed(() =>
  tierOptions.value.map((preset, index) => ({
    value: preset.label,
    label: tierMegapixels(preset),
    sub: resolutionTierLabel(index, tierOptions.value.length),
  })),
);

function changedDimension(event: Event): number {
  return snapMobileDimension(Number((event.target as HTMLInputElement).value), 64, alignment.value);
}

function snapWidth(event: Event): void {
  width.value = changedDimension(event);
}

function snapHeight(event: Event): void {
  height.value = changedDimension(event);
}

function swapDimensions(): void {
  [width.value, height.value] = [height.value, width.value];
}

function matchSource(): void {
  const source = sourceResolution.value;
  if (!source) return;
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
      v-if="sourceStatus"
      class="mobile-resolution-source"
      data-test="mobile-source-resolution-status"
      role="status"
    >
      <strong>{{ followsSource ? sourceStatus.label : "Manual" }}</strong>
      <span>{{
        followsSource
          ? sourceStatus.detail
          : `${sourceStatus.detail} · output is ${width}×${height}`
      }}</span>
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
      <span class="mobile-resolution-label">Orientation</span>
      <div class="mobile-resolution-segments" role="group" aria-label="Orientation">
        <button
          v-for="orientation in MOBILE_RESOLUTION_ORIENTATIONS"
          :key="orientation"
          type="button"
          class="mobile-resolution-segment"
          :class="{ 'is-selected': currentOrientation === orientation }"
          :aria-pressed="currentOrientation === orientation"
          :disabled="!isOrientationAvailable(orientation)"
          :data-orientation="orientation.toLowerCase()"
          @click="setOrientation(orientation)"
        >
          {{ orientation }}
        </button>
      </div>
    </div>

    <div class="mobile-resolution-group">
      <span class="mobile-resolution-label">Aspect ratio</span>
      <div class="mobile-resolution-aspects" role="group" aria-label="Aspect ratio presets">
        <button
          v-for="aspect in aspectOptions"
          :key="aspect"
          type="button"
          class="mobile-resolution-aspect"
          :class="{ 'is-selected': currentPreset?.aspect === aspect }"
          :aria-pressed="currentPreset?.aspect === aspect"
          :aria-label="aspectAccessibleLabel(aspect)"
          :data-aspect="aspect"
          @click="setAspect(aspect)"
        >
          <span class="mobile-resolution-aspect-visual" aria-hidden="true">
            <span
              class="mobile-resolution-aspect-shape"
              data-test="mobile-resolution-aspect-shape"
              aria-hidden="true"
              :style="aspectOptionShapeStyle(aspect)"
            />
          </span>
          <span class="mobile-resolution-aspect-label">{{ aspect }}</span>
        </button>
      </div>
    </div>

    <div v-if="currentPreset" class="mobile-resolution-group mobile-resolution-tier">
      <span class="mobile-resolution-label">Resolution tier</span>
      <SegmentedControl
        data-test="mobile-resolution-tier"
        :model-value="currentPreset.label"
        :options="tierSegments"
        label="Resolution tier"
        @update:model-value="setTier"
      />
      <p class="mobile-resolution-tier-dims" data-test="mobile-resolution-tier-dims">
        {{ currentPreset.width }} × {{ currentPreset.height }} px
      </p>
    </div>

    <button
      v-if="currentPreset"
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
