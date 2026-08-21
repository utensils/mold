<script setup lang="ts">
import { computed, ref } from "vue";
import SliderRow from "@ui/components/SliderRow.vue";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  SOURCE_FIT_OPTIONS,
  coerceSourceFitForMaskless,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import { blobToBase64 } from "../../lib/base64";
import type { GenerateFormState } from "../../types";

/**
 * The sequence twin of `SourceMediaPanel`: the opening frame is source media,
 * so it belongs in the primary form beside the one-shot well — not behind the
 * Advanced toggle, and never counted in the Advanced badge. The bytes live on
 * the shared sequence draft (they ride the chain request, not the form); the
 * fit policy and strength are ordinary form fields, patched exactly as every
 * other control here does.
 */
const props = defineProps<{ modelValue: GenerateFormState }>();

const emit = defineEmits<{
  "update:modelValue": [value: GenerateFormState];
  "open-picker": [];
}>();

function patch(next: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...next });
}

const draft = useSequenceDraftStore();
const uploadError = ref<string | null>(null);
const mimeType = computed(() =>
  /\.jpe?g$/i.test(draft.openingImage?.filename ?? "")
    ? "image/jpeg"
    : "image/png",
);

/** Decoding the header is also the format gate — drag-and-drop bypasses the
 * file input's accept filter, and the engine reads nothing but PNG/JPEG. */
async function onFile(file: File) {
  const base64 = await blobToBase64(file);
  const dimensions = imageDimensionsFromBase64(base64);
  if (!dimensions) {
    uploadError.value = "Only PNG or JPEG images can be used here.";
    return;
  }
  uploadError.value = null;
  draft.openingImage = {
    filename: file.name,
    base64,
    width: dimensions.width,
    height: dimensions.height,
  };
  patch({
    sourceFitPolicy: coerceSourceFitForMaskless(
      props.modelValue.sourceFitPolicy ?? { mode: "crop-fill" },
    ),
  });
}

function clearOpeningImage() {
  uploadError.value = null;
  draft.openingImage = null;
}

function openGalleryPicker() {
  uploadError.value = null;
  emit("open-picker");
}

// No mask exists for a sequence opening frame, so the repaint mode has no
// meaning here and never renders.
const fitOptions = SOURCE_FIT_OPTIONS.filter(
  (option) => option.value !== "pad-repaint",
);
const fitMode = computed(
  () =>
    coerceSourceFitForMaskless(
      props.modelValue.sourceFitPolicy ?? { mode: "crop-fill" },
    ).mode,
);
function setFit(mode: SourceFitMode) {
  patch({
    sourceFitPolicy: sourceFitPolicyForMode(mode, {
      supportsMask: false,
      upscalerModel: props.modelValue.upscaleModel || "real-esrgan-x4plus:fp16",
    }),
  });
}
</script>

<template>
  <section class="soi" data-test="sequence-opening-image-panel">
    <div class="soi__head">
      <span class="soi__kicker">Opening image</span>
    </div>

    <ImageDropWell
      :image="draft.openingImage?.base64 ?? null"
      :mime-type="mimeType"
      :filename="draft.openingImage?.filename ?? null"
      placeholder="Drop an image or click to pick the original starting frame"
      accept="image/png,image/jpeg"
      gallery
      alt="Opening sequence image"
      test-id="sequence-opening-image"
      @file="onFile"
      @gallery="openGalleryPicker"
      @clear="clearOpeningImage"
    />
    <p
      v-if="uploadError"
      class="soi__error"
      role="alert"
      data-test="sequence-opening-image-error"
    >
      {{ uploadError }}
    </p>

    <template v-if="draft.openingImage">
      <SliderRow
        label="Source strength"
        :model-value="modelValue.strength"
        :min="0"
        :max="1"
        :step="0.01"
        :value-label="modelValue.strength.toFixed(2)"
        data-test="sequence-source-strength"
        @update:model-value="patch({ strength: $event })"
      />
      <div class="soi__field">
        <label class="soi__label" for="sequence-source-fit"
          >Fit to video frame</label
        >
        <select
          id="sequence-source-fit"
          class="soi__input"
          data-test="sequence-source-fit"
          :value="fitMode"
          @change="
            setFit(($event.target as HTMLSelectElement).value as SourceFitMode)
          "
        >
          <option
            v-for="option in fitOptions"
            :key="option.value"
            :value="option.value"
          >
            {{ option.label }}
          </option>
        </select>
        <p class="soi__hint">
          Applied to the opening image before the first clip renders.
        </p>
      </div>
    </template>
    <p v-else class="soi__hint">
      Optional — the original starting frame for the first clip.
    </p>
  </section>
</template>

<style scoped>
.soi {
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card-lg);
  box-shadow: inset 0 1px 0 var(--card-hi);
  padding: 18px;
}
.soi__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
  margin-bottom: 12px;
}
.soi__kicker {
  font-family: var(--f-mono);
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.soi__field {
  margin-top: 12px;
}
.soi__label {
  display: block;
  font-size: 11.5px;
  font-weight: 600;
  color: var(--ink-2);
  margin-bottom: 6px;
}
.soi__input {
  width: 100%;
  border: 1px solid var(--ce);
  background: var(--bath);
  color: var(--ink);
  border-radius: var(--radius-control);
  padding: 8px 10px;
  font-size: 13px;
}
.soi__hint {
  font-size: 10.5px;
  color: var(--ink-3);
  margin-top: 6px;
}
.soi__error {
  font-size: 11px;
  font-weight: 600;
  color: var(--stop);
  margin-top: 8px;
}
</style>
