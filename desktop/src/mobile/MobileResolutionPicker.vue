<script setup lang="ts">
import { computed, ref } from "vue";
import {
  aspectRatioLabel,
  matchPreset,
  orientationLabel,
  presetsForFamily,
  type ResolutionPreset,
} from "../lib/resolutions";
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
    disabled?: boolean;
  }>(),
  { disabled: false },
);

const width = defineModel<number>("width", { required: true });
const height = defineModel<number>("height", { required: true });

const manualOpen = ref(false);
const presets = computed(() => presetsForFamily(props.family));
const currentPreset = computed(() => matchPreset(width.value, height.value, props.family));
const currentOrientation = computed(() => orientationLabel(width.value, height.value));
const currentAspect = computed(() => aspectRatioLabel(width.value, height.value, props.family));
const customVisible = computed(() => manualOpen.value || !currentPreset.value);
const aspectOptions = computed(() =>
  aspectsForOrientation(presets.value, currentOrientation.value),
);
const tierOptions = computed(() =>
  currentPreset.value ? sortedResolutionTiers(presets.value, currentPreset.value.aspect) : [],
);
const aspectShapeStyle = computed(() => ({
  aspectRatio: `${width.value} / ${height.value}`,
  ...(width.value > height.value ? { width: "34px" } : { height: "30px" }),
}));

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

function setTier(event: Event): void {
  const label = (event.target as HTMLSelectElement).value;
  applyResolution(tierOptions.value.find((preset) => preset.label === label) ?? null);
}

function tierOptionLabel(preset: ResolutionPreset, index: number): string {
  const tier = resolutionTierLabel(index, tierOptions.value.length);
  const megapixels = ((preset.width * preset.height) / 1_000_000).toFixed(1);
  return `${tier} · ${preset.width} × ${preset.height} · ${megapixels} MP`;
}

function changedDimension(event: Event): number {
  return snapMobileDimension(Number((event.target as HTMLInputElement).value));
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
</script>

<template>
  <fieldset class="mobile-resolution-picker" :disabled="disabled">
    <legend class="mobile-resolution-legend">Resolution</legend>

    <div
      class="mobile-resolution-summary"
      data-test="mobile-resolution-summary"
      role="status"
      aria-live="polite"
    >
      <div class="mobile-resolution-preview" aria-hidden="true">
        <span data-test="mobile-resolution-shape" :style="aspectShapeStyle" />
      </div>
      <div class="mobile-resolution-copy">
        <strong>{{ width }} × {{ height }}</strong>
        <span>{{ currentAspect }} · {{ currentOrientation }}</span>
      </div>
      <span v-if="!currentPreset" class="mobile-resolution-custom-badge">Custom</span>
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
      <div class="mobile-resolution-aspects" aria-label="Aspect ratio presets">
        <button
          v-for="aspect in aspectOptions"
          :key="aspect"
          type="button"
          class="mobile-resolution-aspect"
          :class="{ 'is-selected': currentPreset?.aspect === aspect }"
          :aria-pressed="currentPreset?.aspect === aspect"
          :data-aspect="aspect"
          @click="setAspect(aspect)"
        >
          {{ aspect }}
        </button>
      </div>
    </div>

    <label v-if="currentPreset" class="field mobile-resolution-tier">
      <span>Resolution tier</span>
      <select
        class="control"
        data-test="mobile-resolution-tier"
        :value="currentPreset.label"
        @change="setTier"
      >
        <option v-for="(preset, index) in tierOptions" :key="preset.label" :value="preset.label">
          {{ tierOptionLabel(preset, index) }}
        </option>
      </select>
    </label>

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
          step="16"
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
          step="16"
          aria-label="Custom height"
          @change="snapHeight"
        />
      </label>
    </div>
    <p v-if="customVisible" class="mobile-resolution-note">
      Custom dimensions snap to multiples of 16 for model compatibility.
    </p>
  </fieldset>
</template>
