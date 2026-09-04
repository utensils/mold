<script setup lang="ts">
import { computed } from "vue";
import AccordionSection from "@ui/components/AccordionSection.vue";
import Chip from "@ui/components/Chip.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { cameraMotionMode } from "@studio/lib/cameraMotion";
import type { GenerateForm } from "../../lib/generateForm";
import type { Ltx2CameraControlInfo } from "../../lib/api/types";
import { generationCapabilitiesForFamily } from "../../lib/capabilities";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    cameraControlsEnabled?: boolean;
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
    cameraUnsupportedReason?: string | null;
  }>(),
  {
    cameraControlsEnabled: false,
    cameraControls: () => [],
    cameraControlsLoaded: false,
    cameraUnsupportedReason: null,
  },
);

const draft = useSequenceDraftStore();
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
// The opening image is primary-form source media (`SequenceOpeningImageWell`),
// so it is deliberately absent from this pane and from its active count.
const activeCount = computed(
  () =>
    Number(
      guidanceCaps.value.supportsNegativePrompt && Boolean(activeClip.value?.negativePrompt.trim()),
    ) + Number(props.cameraControlsEnabled && Boolean(activeClip.value?.cameraControl)),
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

function setCameraMode(mode: string) {
  const clip = activeClip.value;
  if (!clip) return;
  if (mode === "custom") {
    if (cameraMotionMode(clip.cameraControl) !== "custom") clip.cameraControl = "";
  } else {
    clip.cameraControl = mode || null;
  }
}

// Reset clears sequence-advanced knobs only; the opening image and its
// strength/fit are staged source media owned by the primary form, so they
// survive this Reset (the inspector header's ↺ Reset is what clears them).
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
        v-if="activeClip"
        icon="negative"
        :title="`Scene ${activeIndex + 1} — what to steer away from`"
        summary="What this scene should keep out"
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
          aria-label="What the selected scene should steer away from"
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
        :title="`Scene ${activeIndex + 1} camera motion`"
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
  color: var(--mold-text-dim);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
}
.ms-adv__reset {
  border: 1px solid var(--mold-border-control);
  background: transparent;
  color: var(--mold-text-2);
  border-radius: var(--mold-radius-3);
  cursor: pointer;
}
.ms-adv__reset {
  padding: 5px 9px;
  font-size: var(--mold-fs-micro);
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
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-3);
  background: var(--mold-bg-deep);
  color: var(--mold-text);
  padding: 9px 10px;
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
.ms-camera-path,
.ms-hint {
  margin-top: 9px;
}
.ms-hint {
  color: var(--mold-text-dim);
  font-size: var(--mold-fs-micro);
  line-height: 1.45;
}
.ms-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 9px;
}
.ms-switch-row {
  color: var(--mold-text-2);
  font-size: var(--mold-fs-xs);
}
</style>
