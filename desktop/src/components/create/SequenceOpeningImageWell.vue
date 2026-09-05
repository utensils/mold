<script setup lang="ts">
import { computed, ref } from "vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { GenerateForm, PickedImage } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import {
  SOURCE_FIT_OPTIONS,
  coerceSourceFitForMaskless,
  defaultSourceFitPolicy,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { strengthSemantics } from "@studio/lib/strengthSemantics";
import { fileToBase64 } from "../../lib/image";
import { useToastStore } from "../../stores/toasts";

/**
 * The sequence's opening image, its source strength, and its fit — primary-form
 * source media, exactly like the one-shot `SourceImageWell` it sits in place of.
 * It is deliberately NOT an Advanced control: attaching a starting frame is a
 * first-class authoring decision, and the inspector's ↺ Reset clears it with the
 * rest of the primary form (`InspectorPanel.resetSettings`).
 */
const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    upscalers?: ModelEntry[];
  }>(),
  { upscalers: () => [] },
);

const draft = useSequenceDraftStore();
const toasts = useToastStore();
const pickerOpen = ref(false);
const openingImageMime = computed(() =>
  /\.jpe?g$/i.test(draft.openingImage?.filename ?? "") ? "image/jpeg" : "image/png",
);
const fitOptions = SOURCE_FIT_OPTIONS.filter((option) => option.value !== "pad-repaint");
const fitMode = computed(() => coerceSourceFitForMaskless(props.form.sourceFit).mode);
const upscalerAvailable = computed(() =>
  Boolean(props.form.upscaleModel || props.upscalers[0]?.name),
);
/** The one label policy for the shared `strength` field — and the only thing
 *  that knows which end of the track keeps the photo (LTX-2 inverts it). */
const strength = computed(() => strengthSemantics(props.form.family));

function setSourceFit(mode: SourceFitMode) {
  props.form.sourceFit = sourceFitPolicyForMode(mode, {
    supportsMask: false,
    upscalerModel: props.form.upscaleModel || props.upscalers[0]?.name || "",
  });
}

function onPickImage(images: PickedImage[]) {
  const image = images[0];
  pickerOpen.value = false;
  if (!image) return;
  draft.openingImage = { filename: image.filename, base64: image.base64 };
  props.form.sourceFit = defaultSourceFitPolicy();
}

async function onOpeningImageFile(file: File) {
  try {
    const base64 = await fileToBase64(file);
    const dimensions = imageDimensionsFromBase64(base64);
    if (!dimensions) {
      toasts.push("Only PNG or JPEG images can be used here.", "error");
      return;
    }
    draft.openingImage = {
      filename: file.name,
      base64,
      width: dimensions.width,
      height: dimensions.height,
    };
    props.form.sourceFit = defaultSourceFitPolicy();
  } catch {
    toasts.push("Couldn't read the image.", "error");
  }
}
</script>

<template>
  <div class="ms-opening" data-test="sequence-opening-image-field">
    <div class="ms-field__label">Opening image</div>
    <ImageDropWell
      :image="draft.openingImage?.base64 ?? null"
      :mime-type="openingImageMime"
      :filename="draft.openingImage?.filename ?? null"
      placeholder="Drop an image or click to pick the original starting frame"
      accept="image/png,image/jpeg"
      gallery
      alt="Opening sequence image"
      test-id="sequence-opening-image"
      drop-target="opening"
      @file="onOpeningImageFile"
      @gallery="pickerOpen = true"
      @clear="draft.openingImage = null"
    />
    <div v-if="draft.openingImage" class="ms-source-controls">
      <label class="ms-range">
        <span>
          {{ strength.label }}
          <output class="font-mono text-xs">{{ form.strength.toFixed(2) }}</output>
        </span>
        <input
          v-model.number="form.strength"
          type="range"
          min="0"
          max="1"
          step="0.01"
          data-test="sequence-source-strength"
        />
      </label>
      <p class="ms-source-hint" data-test="sequence-source-strength-hint">
        {{ strength.hint }}
      </p>
      <label class="ms-field">
        <span>Fit to video frame</span>
        <select
          class="ms-input"
          :value="fitMode"
          data-test="sequence-source-fit"
          @change="setSourceFit(($event.target as HTMLSelectElement).value as SourceFitMode)"
        >
          <option
            v-for="option in fitOptions"
            :key="option.value"
            :value="option.value"
            :disabled="option.value === 'upscale-then-fit' && !upscalerAvailable"
          >
            {{ option.label }}
          </option>
        </select>
      </label>
      <p v-if="!upscalerAvailable" class="ms-hint">Install an upscaler to enable Upscale + crop.</p>
      <p class="ms-hint">Applied to the opening image before clip 1 renders.</p>
    </div>

    <ImagePickerModal
      :open="pickerOpen"
      title="Opening sequence image"
      :multiple="false"
      @pick="onPickImage"
      @close="pickerOpen = false"
    />
  </div>
</template>

<style scoped>
.ms-field__label {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.ms-source-controls {
  display: grid;
  gap: 10px;
  margin-top: 12px;
}
.ms-range,
.ms-field {
  display: grid;
  gap: 6px;
  color: var(--mold-text-2);
  font-size: var(--mold-fs-micro);
}
.ms-range > span {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
.ms-range input {
  width: 100%;
  accent-color: var(--mold-blue);
}
.ms-source-hint {
  margin: -4px 0 0;
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
}
.ms-input {
  width: 100%;
  box-sizing: border-box;
  min-height: 36px;
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  padding: 8px 10px;
}
.ms-hint {
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
  line-height: 1.45;
}
</style>
