<script setup lang="ts">
import { computed, ref } from "vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import Chip from "@ui/components/Chip.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { cameraMotionMode } from "@studio/lib/cameraMotion";
import type { GenerateForm, PickedImage } from "../../lib/generateForm";
import type { Ltx2CameraControlInfo, ModelEntry } from "../../lib/api/types";
import {
  SOURCE_FIT_OPTIONS,
  coerceSourceFitForMaskless,
  sourceFitPolicyForMode,
  type SourceFitMode,
} from "@studio/lib/sourceFit";
import ImagePickerModal from "../generate/ImagePickerModal.vue";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";
import ImageDropWell from "@studio/components/ImageDropWell.vue";
import { imageDimensionsFromBase64 } from "@studio/lib/imageDimensions";
import { fileToBase64 } from "../../lib/image";
import { useToastStore } from "../../stores/toasts";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    cameraControlsEnabled?: boolean;
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
    upscalers?: ModelEntry[];
    cameraUnsupportedReason?: string | null;
  }>(),
  {
    cameraControlsEnabled: false,
    cameraControls: () => [],
    cameraControlsLoaded: false,
    upscalers: () => [],
    cameraUnsupportedReason: null,
  },
);

const draft = useSequenceDraftStore();
const toasts = useToastStore();
const pickerOpen = ref(false);
const openingImageMime = computed(() =>
  /\.jpe?g$/i.test(draft.openingImage?.filename ?? "") ? "image/jpeg" : "image/png",
);
const activeClip = computed(
  () => draft.clips.find((clip) => clip.id === draft.activeClipId) ?? draft.clips[0] ?? null,
);
const activeIndex = computed(() =>
  activeClip.value ? draft.clips.findIndex((clip) => clip.id === activeClip.value?.id) : -1,
);
const guidanceCaps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.form.guidanceCapabilities,
  ),
);
// Keep old/unknown servers compatible; only an explicit per-checkpoint
// rejection parks the opening image and its controls.
const supportsOpeningImage = computed(() => props.form.sourceImageCapability !== "unsupported");
const activeCount = computed(
  () =>
    Number(supportsOpeningImage.value && Boolean(draft.openingImage)) +
    Number(
      guidanceCaps.value.supportsNegativePrompt && Boolean(activeClip.value?.negativePrompt.trim()),
    ) +
    Number(props.cameraControlsEnabled && Boolean(activeClip.value?.cameraControl)),
);
const fitOptions = SOURCE_FIT_OPTIONS.filter((option) => option.value !== "pad-repaint");
const fitMode = computed(() => coerceSourceFitForMaskless(props.form.sourceFit).mode);
const upscalerAvailable = computed(() =>
  Boolean(props.form.upscaleModel || props.upscalers[0]?.name),
);

function setSourceFit(mode: SourceFitMode) {
  props.form.sourceFit = sourceFitPolicyForMode(mode, {
    supportsMask: false,
    upscalerModel: props.form.upscaleModel || props.upscalers[0]?.name || "",
  });
}

const NEGATIVE_QUICK_ADDS = [
  "blurry",
  "extra fingers",
  "watermark",
  "low quality",
  "oversaturated",
];

function addNegative(word: string) {
  const clip = activeClip.value;
  if (!clip) return;
  const current = clip.negativePrompt.trim();
  clip.negativePrompt = current ? `${current}, ${word}` : word;
}

function setCameraMode(mode: string) {
  const clip = activeClip.value;
  if (!clip) return;
  if (mode === "custom") {
    if (cameraMotionMode(clip.cameraControl) !== "custom") clip.cameraControl = "";
  } else {
    clip.cameraControl = mode || null;
  }
}

function onPickImage(images: PickedImage[]) {
  const image = images[0];
  pickerOpen.value = false;
  if (!image) return;
  draft.openingImage = { filename: image.filename, base64: image.base64 };
  props.form.sourceFit = coerceSourceFitForMaskless(props.form.sourceFit);
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
    props.form.sourceFit = coerceSourceFitForMaskless(props.form.sourceFit);
  } catch {
    toasts.push("Couldn't read the image.", "error");
  }
}

// Reset clears sequence-advanced knobs only; the opening image and its
// strength/fit are staged source media and survive (web parity).
function reset() {
  for (const clip of draft.clips) {
    clip.negativePrompt = "";
    clip.cameraControl = null;
  }
}
</script>

<template>
  <section class="ms-adv" data-test="sequence-inline-advanced">
    <div class="ms-adv__toolbar">
      <span class="ms-adv__summary">
        {{ activeCount > 0 ? `${activeCount} active` : "Sequence controls" }}
      </span>
      <button
        type="button"
        class="ms-adv__reset"
        data-test="sequence-advanced-reset"
        @click="reset"
      >
        Reset
      </button>
    </div>

    <div class="ms-adv__list">
      <AccordionSection
        v-if="supportsOpeningImage"
        icon="image"
        title="Opening sequence image"
        :summary="draft.openingImage?.filename ?? 'Optional original starting frame'"
        :open="true"
        :header-interactive="false"
        data-test="sequence-section-opening-image"
      >
        <ImageDropWell
          :image="draft.openingImage?.base64 ?? null"
          :mime-type="openingImageMime"
          :filename="draft.openingImage?.filename ?? null"
          placeholder="Drop an image or click to pick the original starting frame"
          accept="image/png,image/jpeg"
          gallery
          alt="Opening sequence image"
          test-id="sequence-opening-image"
          @file="onOpeningImageFile"
          @gallery="pickerOpen = true"
          @clear="draft.openingImage = null"
        />
        <div v-if="draft.openingImage" class="ms-source-controls">
          <label class="ms-range">
            <span>
              Source strength
              <output class="data-mono">{{ form.strength.toFixed(2) }}</output>
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
          <p v-if="!upscalerAvailable" class="ms-hint">
            Install an upscaler to enable Upscale + crop.
          </p>
          <p class="ms-hint">Applied to the opening image before clip 1 renders.</p>
        </div>
      </AccordionSection>

      <AccordionSection
        v-if="activeClip"
        icon="negative"
        :title="`Clip ${activeIndex + 1} negative prompt`"
        summary="What to steer away from in this clip"
        :open="true"
        :header-interactive="false"
        data-test="sequence-section-negative"
      >
        <textarea
          v-model="activeClip.negativePrompt"
          :disabled="!guidanceCaps.supportsNegativePrompt"
          data-selectable
          rows="2"
          placeholder="blurry, low quality, deformed…"
          class="ms-textarea"
          aria-label="Active clip negative prompt"
        />
        <p
          v-if="!guidanceCaps.supportsNegativePrompt"
          class="ms-hint"
          data-test="sequence-negative-unavailable-hint"
        >
          Saved for reuse, but this distilled recipe does not use negative-prompt guidance.
        </p>
        <div v-if="guidanceCaps.supportsNegativePrompt" class="ms-chips">
          <Chip v-for="word in NEGATIVE_QUICK_ADDS" :key="word" @click="addNegative(word)"
            >+ {{ word }}</Chip
          >
        </div>
      </AccordionSection>

      <AccordionSection
        v-if="activeClip && cameraControlsEnabled"
        icon="video"
        :title="`Clip ${activeIndex + 1} camera motion`"
        :summary="
          cameraControls.find((control) => control.id === activeClip?.cameraControl)?.label ??
          activeClip.cameraControl ??
          'None'
        "
        :open="true"
        :header-interactive="false"
        data-test="sequence-section-camera"
      >
        <select
          class="ms-input"
          data-test="sequence-camera-motion"
          aria-label="Active clip camera motion"
          :value="cameraMotionMode(activeClip.cameraControl)"
          @change="setCameraMode(($event.target as HTMLSelectElement).value)"
        >
          <option value="">None</option>
          <option v-for="control in cameraControls" :key="control.id" :value="control.id">
            {{ control.label }}{{ control.installed ? "" : " · downloads on first use" }}
          </option>
          <option value="custom">Custom LoRA path…</option>
        </select>
        <input
          v-if="cameraMotionMode(activeClip.cameraControl) === 'custom'"
          v-model="activeClip.cameraControl"
          class="ms-input ms-camera-path"
          data-test="sequence-camera-motion-custom"
          aria-label="Active clip camera motion LoRA path"
          placeholder="/path/to/lora.safetensors"
        />
        <p v-if="cameraControlsLoaded && cameraControls.length === 0" class="ms-hint">
          {{
            cameraUnsupportedReason ??
            "Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA path."
          }}
        </p>
      </AccordionSection>
    </div>

    <ImagePickerModal
      v-if="supportsOpeningImage"
      :open="pickerOpen"
      title="Opening sequence image"
      :multiple="false"
      @pick="onPickImage"
      @close="pickerOpen = false"
    />
  </section>
</template>

<style scoped>
.ms-adv {
  padding-top: 10px;
}
.ms-adv__toolbar,
.ms-switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}
.ms-adv__toolbar {
  margin-bottom: 10px;
}
.ms-adv__summary {
  color: var(--ink-3);
  font-family: var(--f-mono);
  font-size: 10px;
}
.ms-adv__reset {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 8px;
  cursor: pointer;
}
.ms-adv__reset {
  padding: 5px 9px;
  font-size: 11px;
}
.ms-source-controls {
  display: grid;
  gap: 10px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--ce);
}
.ms-range,
.ms-field {
  display: grid;
  gap: 6px;
  color: var(--ink-2);
  font-size: 11px;
}
.ms-range > span {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}
.ms-range input {
  width: 100%;
  accent-color: var(--safelight);
}
.ms-adv__list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.ms-textarea {
  width: 100%;
  box-sizing: border-box;
  resize: none;
  border: 1px solid var(--ce);
  border-radius: 8px;
  background: var(--bath);
  color: var(--rebate);
  padding: 9px 10px;
}
.ms-input {
  width: 100%;
  box-sizing: border-box;
  min-height: 36px;
  border: 1px solid var(--ce);
  border-radius: 8px;
  background: var(--bath);
  color: var(--rebate);
  padding: 8px 10px;
}
.ms-camera-path,
.ms-hint {
  margin-top: 9px;
}
.ms-hint {
  color: var(--ink-3);
  font-size: 10px;
  line-height: 1.45;
}
.ms-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 9px;
}
.ms-switch-row {
  color: var(--ink-2);
  font-size: 12px;
}
</style>
