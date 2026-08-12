<script setup lang="ts">
/*
 * "Continue a video" — the extend field of the Advanced drawer's Video
 * section.
 *
 * It lived inside the LTX-2 suite while LTX-2 was the only family that could
 * continue a clip. Wan can too (#783) and has no LTX-2 suite, so the control
 * is its own component the drawer renders whenever the selected model
 * advertises `supports_extend`; the family only decides the overlap rule.
 * Holds no local copy — it patches the parent form via v-model, like the
 * other advanced controls.
 */
import { computed } from "vue";
import type { GenerateFormState, SourceMediaState } from "../../../types";
import { blobToBase64 } from "../../../lib/base64";
import {
  DEFAULT_EXTEND_OVERLAP_FRAMES,
  extendNewFrames,
  extendOverlapOptions,
  extendValidationError,
  resolveExtendOverlapFrames,
} from "@studio/lib/extend";

const props = withDefaults(
  defineProps<{
    modelValue: GenerateFormState;
    /** Selected model's family — the overlap grid is per family. */
    family?: string;
    defaultOverlapFrames?: number;
  }>(),
  {
    family: "",
    defaultOverlapFrames: DEFAULT_EXTEND_OVERLAP_FRAMES,
  },
);
const emit = defineEmits<{ "update:modelValue": [value: GenerateFormState] }>();

function patch(part: Partial<GenerateFormState>) {
  emit("update:modelValue", { ...props.modelValue, ...part });
}

async function readMedia(event: Event): Promise<SourceMediaState | null> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0] ?? null;
  input.value = "";
  if (!file) return null;
  return {
    kind: "upload",
    filename: file.name,
    base64: await blobToBase64(file),
  };
}

async function onExtendVideo(event: Event) {
  const extendVideo = await readMedia(event);
  if (!extendVideo) return;
  patch({ extendVideo, extendVideoPath: "" });
}
function clearExtendVideo() {
  // Drop the overlap with the clip so a later continuation starts from the
  // server default rather than inheriting a stale, possibly invalid value.
  patch({ extendVideo: null, extendOverlapFrames: null });
}

// The same resolution the request builder submits, so the number on screen is
// the number on the wire. `defaultOverlapFrames` already arrives clamped by
// `serverExtendOverlapDefault`, hence the family-free shape here.
const extendOverlap = computed(() =>
  resolveExtendOverlapFrames(props.modelValue.extendOverlapFrames, {
    extend_default_overlap_frames: props.defaultOverlapFrames,
  }),
);
const isExtending = computed(
  () =>
    props.modelValue.extendVideo !== null ||
    props.modelValue.extendVideoPath.trim() !== "",
);
const extendOverlapChoices = computed(() =>
  extendOverlapOptions(props.modelValue.frames, props.family),
);
const extendError = computed(() =>
  isExtending.value
    ? extendValidationError({
        overlapFrames: extendOverlap.value,
        frames: props.modelValue.frames,
        family: props.family,
        hasSourceImage: props.modelValue.imageAttachments.length > 0,
        hasSourceVideo:
          props.modelValue.sourceVideo !== null ||
          props.modelValue.sourceVideoPath.trim() !== "",
        hasKeyframes: props.modelValue.keyframes.length > 0,
      })
    : null,
);
const extendSummary = computed(() => {
  const added = extendNewFrames(props.modelValue.frames, extendOverlap.value);
  if (added === null) return "";
  return `Appends ${added} new frames after the source clip.`;
});
</script>

<template>
  <div class="extend" data-test="extend-field">
    <label class="extend__label">Continue a video</label>
    <label v-if="!modelValue.extendVideo" class="extend__file">
      <span>Attach video to continue</span>
      <input
        type="file"
        accept="video/*"
        class="extend__file-input"
        data-test="extend-attach"
        @change="onExtendVideo"
      />
    </label>
    <div v-else class="extend__source-row">
      <span class="extend__source-name">{{
        modelValue.extendVideo.filename
      }}</span>
      <button
        type="button"
        class="extend__remove"
        data-test="extend-clear"
        @click="clearExtendVideo"
      >
        Remove
      </button>
    </div>
    <input
      class="extend__input"
      data-test="extend-path"
      placeholder="or server video path"
      :value="modelValue.extendVideoPath"
      :disabled="modelValue.extendVideo !== null"
      @input="
        patch({ extendVideoPath: ($event.target as HTMLInputElement).value })
      "
    />
    <template v-if="isExtending">
      <label class="extend__label">Overlap (frames of motion context)</label>
      <select
        class="extend__input"
        data-test="extend-overlap"
        :value="String(extendOverlap)"
        @change="
          patch({
            extendOverlapFrames: Number(
              ($event.target as HTMLSelectElement).value,
            ),
          })
        "
      >
        <option
          v-for="option in extendOverlapChoices"
          :key="option"
          :value="String(option)"
        >
          {{ option }}
        </option>
      </select>
      <p v-if="extendError" class="extend__error" data-test="extend-error">
        {{ extendError }}
      </p>
      <p v-else class="extend__hint" data-test="extend-summary">
        {{ extendSummary }}
      </p>
    </template>
  </div>
</template>

<style scoped>
.extend {
  margin-bottom: 14px;
}
.extend__label {
  display: block;
  font-size: 12px;
  color: var(--ink-2);
  font-weight: 600;
  margin-bottom: 8px;
}
.extend__input {
  width: 100%;
  box-sizing: border-box;
  height: 40px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-control);
  color: var(--rebate);
  padding: 0 12px;
  font-size: 13px;
  font-family: var(--f-mono);
}
.extend__input:disabled {
  opacity: 0.5;
}
.extend__file {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  border: 1.5px dashed var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: var(--radius-control);
  padding: 14px;
  font-size: 12.5px;
  cursor: pointer;
  margin-bottom: 8px;
}
.extend__file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  opacity: 0;
  pointer-events: none;
}
.extend__source-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.extend__source-name {
  font-size: 13px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.extend__remove {
  border: 0;
  background: transparent;
  color: var(--stop);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}
.extend__hint {
  margin: 6px 0 0;
  font-size: 11.5px;
  color: var(--ink-2);
}
.extend__error {
  margin: 6px 0 0;
  font-size: 11.5px;
  color: var(--stop);
}
</style>
