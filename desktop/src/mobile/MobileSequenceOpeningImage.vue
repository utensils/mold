<script setup lang="ts">
/*
 * The sequence's opening still — the frame clip 1 is conditioned on.
 *
 * It lives in the PRIMARY Create stack, exactly where one-shot output keeps
 * `MobileSourceControls`: source media is what the print is made of, not a
 * fine control, and burying it in the Advanced sheet also put it behind that
 * sheet's Reset (which is scoped to model-owned knobs and must never discard
 * staged media). The clip list still owns the image itself — it is written to
 * the shared @studio sequence draft — so this component holds no source state
 * of its own beyond the picker's open flag.
 *
 * Every control here takes its iPhone sizing from the shared mobile.css
 * primitives — `secondary-button` (46px), `control` (44px), and the global
 * 16px editable-text floor — so this component introduces no tokens of its
 * own and cannot shrink a target by restating one.
 */
import { computed, ref } from "vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  SOURCE_FIT_OPTIONS,
  coerceSourceFitForMaskless,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import type { ApiTarget } from "../lib/api/client";
import type { ModelEntry } from "../lib/api/types";
import type { GenerateForm } from "../lib/generateForm";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import MobileImagePickerSheet, {
  type MobileGallerySource,
  type MobilePickedImage,
} from "./MobileImagePickerSheet.vue";

const props = withDefaults(
  defineProps<{
    /** Strength and fit are ordinary form fields — the chain request reads
     *  them from the same form a one-shot does. */
    form: GenerateForm;
    target: ApiTarget | null;
    gallerySources?: MobileGallerySource[];
    upscalers?: ModelEntry[];
    /** A submit or a durable job is in flight. */
    locked?: boolean;
  }>(),
  { target: null, gallerySources: () => [], upscalers: () => [], locked: false },
);

const draft = useSequenceDraftStore();
const imagePickerOpen = ref(false);

// Sequences have no mask well, so the repaint fit is not offered.
const fitOptions = SOURCE_FIT_OPTIONS.filter((option) => option.value !== "pad-repaint");
const fitMode = computed(() => coerceSourceFitForMaskless(props.form.sourceFit).mode);
const upscalerAvailable = computed(() =>
  Boolean(props.form.upscaleModel || props.upscalers[0]?.name),
);

function setSourceFit(mode: SourceFitMode): void {
  props.form.sourceFit = sourceFitPolicyForMode(mode, {
    supportsMask: false,
    upscalerModel: props.form.upscaleModel || props.upscalers[0]?.name || "",
  });
}

function setOpeningImage(image: MobilePickedImage): void {
  draft.openingImage = { filename: image.filename, base64: image.base64 };
  props.form.sourceFit = coerceSourceFitForMaskless(props.form.sourceFit);
  imagePickerOpen.value = false;
}

function sourceImageMime(filename: string): string {
  return /\.jpe?g$/i.test(filename.trim()) ? "image/jpeg" : "image/png";
}
</script>

<template>
  <div class="mobile-sequence-opening">
    <ImageDropWell
      :image="draft.openingImage?.base64 ?? null"
      :mime-type="draft.openingImage ? sourceImageMime(draft.openingImage.filename) : null"
      :filename="draft.openingImage?.filename ?? null"
      placeholder="Attach opening image"
      alt="Sequence opening image"
      test-id="mobile-sequence-source"
      touch-friendly
      :touch-target-size="48"
      native-picker
      :disabled="locked"
      :pick-disabled="!target"
      @pick="imagePickerOpen = true"
      @clear="draft.openingImage = null"
    />
    <template v-if="draft.openingImage">
      <label class="mobile-range-field">
        <span>
          Source strength
          <output>{{ form.strength.toFixed(2) }}</output>
        </span>
        <input
          v-model.number="form.strength"
          type="range"
          min="0"
          max="1"
          step="0.01"
          aria-label="Sequence source strength"
          data-test="mobile-sequence-source-strength"
          :disabled="locked"
        />
      </label>
      <label class="field">
        <span>Fit to video frame</span>
        <select
          class="control"
          :value="fitMode"
          data-test="mobile-sequence-source-fit"
          :disabled="locked"
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
      <p v-if="!upscalerAvailable" class="mobile-source-note">
        Install an upscaler to enable Upscale + crop.
      </p>
      <p class="mobile-source-note">Applied to the opening image before clip 1 renders.</p>
    </template>
    <MobileImagePickerSheet
      :open="imagePickerOpen"
      :target="target"
      :gallery-sources="gallerySources"
      @pick="setOpeningImage"
      @close="imagePickerOpen = false"
    />
  </div>
</template>
