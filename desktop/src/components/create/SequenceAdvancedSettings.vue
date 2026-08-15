<script setup lang="ts">
import { computed, ref } from "vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import Chip from "@ui/components/Chip.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
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

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    chainLimits?: ChainLimits | null;
    cameraControlsEnabled?: boolean;
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
    upscalers?: ModelEntry[];
    cameraUnsupportedReason?: string | null;
  }>(),
  {
    chainLimits: null,
    cameraControlsEnabled: false,
    cameraControls: () => [],
    cameraControlsLoaded: false,
    upscalers: () => [],
    cameraUnsupportedReason: null,
  },
);

const draft = useSequenceDraftStore();
const pickerOpen = ref(false);
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
const activeCount = computed(
  () =>
    Number(Boolean(draft.openingImage)) +
    Number(
      guidanceCaps.value.supportsNegativePrompt && Boolean(activeClip.value?.negativePrompt.trim()),
    ) +
    Number(props.cameraControlsEnabled && Boolean(activeClip.value?.cameraControl)) +
    Number(draft.enableAudio),
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

// Reset clears sequence-advanced knobs only; the opening image and its
// strength/fit are staged source media and survive (web parity).
function reset() {
  draft.enableAudio = false;
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
        icon="image"
        title="Opening sequence image"
        :summary="draft.openingImage?.filename ?? 'Optional original starting frame'"
        :open="true"
        :header-interactive="false"
        data-test="sequence-section-opening-image"
      >
        <button
          type="button"
          class="ms-dropzone"
          data-test="sequence-opening-image-pick"
          @click="pickerOpen = true"
        >
          {{
            draft.openingImage
              ? `Replace ${draft.openingImage.filename}`
              : "Drop an image or click to choose the original starting frame"
          }}
        </button>
        <button
          v-if="draft.openingImage"
          type="button"
          class="ms-remove"
          data-test="sequence-opening-image-clear"
          @click="draft.openingImage = null"
        >
          Remove opening image
        </button>
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

      <AccordionSection
        v-if="props.chainLimits?.supports_audio"
        icon="video"
        title="Sequence audio"
        summary="Generate and mux audio for this timeline"
        :open="true"
        :header-interactive="false"
        data-test="sequence-section-audio"
      >
        <div class="ms-switch-row">
          <span>Generate audio</span>
          <SwitchToggle
            :model-value="draft.enableAudio"
            label="Generate sequence audio"
            @update:model-value="draft.enableAudio = $event"
          />
        </div>
      </AccordionSection>
    </div>

    <ImagePickerModal
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
.ms-adv__reset,
.ms-remove,
.ms-dropzone {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 8px;
  cursor: pointer;
}
.ms-adv__reset,
.ms-remove {
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
.ms-dropzone {
  width: 100%;
  border-style: dashed;
  padding: 22px 12px;
  font-size: 12px;
}
.ms-remove {
  margin-top: 9px;
  color: var(--stop);
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
