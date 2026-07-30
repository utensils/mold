<script setup lang="ts">
import { computed, ref } from "vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import Chip from "@ui/components/Chip.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { PickedImage } from "../../lib/generateForm";
import ImagePickerModal from "../generate/ImagePickerModal.vue";

const props = withDefaults(defineProps<{ chainLimits?: ChainLimits | null }>(), {
  chainLimits: null,
});

const draft = useSequenceDraftStore();
const pickerOpen = ref(false);
const activeClip = computed(
  () => draft.clips.find((clip) => clip.id === draft.activeClipId) ?? draft.clips[0] ?? null,
);
const activeIndex = computed(() =>
  activeClip.value ? draft.clips.findIndex((clip) => clip.id === activeClip.value?.id) : -1,
);
const activeCount = computed(
  () =>
    Number(Boolean(draft.openingImage)) +
    Number(Boolean(activeClip.value?.negativePrompt.trim())) +
    Number(draft.enableAudio),
);

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

function onPickImage(images: PickedImage[]) {
  const image = images[0];
  pickerOpen.value = false;
  if (!image) return;
  draft.openingImage = { filename: image.filename, base64: image.base64 };
}

function reset() {
  draft.openingImage = null;
  draft.enableAudio = false;
  for (const clip of draft.clips) clip.negativePrompt = "";
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
          data-selectable
          rows="2"
          placeholder="blurry, low quality, deformed…"
          class="ms-textarea"
          aria-label="Active clip negative prompt"
        />
        <div class="ms-chips">
          <Chip v-for="word in NEGATIVE_QUICK_ADDS" :key="word" @click="addNegative(word)"
            >+ {{ word }}</Chip
          >
        </div>
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
