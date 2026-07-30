<script setup lang="ts">
/*
 * iPhone sequence bench (mockup 3b). Full-width clip cards with a 44pt seam
 * pill between consecutive cards; tapping a seam opens MobileSeamSheet.
 *
 * The clip list lives in the shared @studio sequence draft — this component
 * owns NO clip state. The old composer kept a private reactive form behind
 * the Create `v-if`, so every tab switch silently erased every clip prompt,
 * and its private width/steps/guidance/seed copies drifted from the form the
 * user could actually see. Shared params stay in MobileApp's generate form
 * and are read at submit time via `buildChainRequest`; `fps` arrives as a
 * prop only so the duration summary can be honest.
 */
import { computed, ref } from "vue";
import SeamPill from "@ui/components/SeamPill.vue";
import {
  defaultClipFrames,
  formatFrameDuration,
  friendlySequenceError,
  sequenceDuration,
  sequenceFrameOptions,
  sequenceMotionTailFrames,
  sequenceValidation,
  type SequenceStage,
  type SequenceTransition,
} from "@studio/lib/sequence";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import { sequenceOpeningImageError } from "@studio/lib/sequenceForm";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { ApiTarget } from "../lib/api/client";
import type { ModelEntry } from "../lib/api/types";
import { base64ToDataUrl } from "../lib/image";
import MobileImagePickerSheet, { type MobilePickedImage } from "./MobileImagePickerSheet.vue";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";
import MobileSeamSheet from "./MobileSeamSheet.vue";

const props = withDefaults(
  defineProps<{
    selectedModel: ModelEntry | null;
    chainLimits: ChainLimits | null;
    target: ApiTarget | null;
    /** The generate form's frame rate — shown, never stored here. */
    fps: number;
    submitting?: boolean;
    /** A create/amend failure from the host, raw. */
    error?: string;
    /** A durable job is already starting on this host. */
    busy?: boolean;
    /** Collapsed caption for the shared-parameter disclosure. */
    settingsSummary?: string;
  }>(),
  { target: null, submitting: false, error: "", busy: false, settingsSummary: "" },
);

const emit = defineEmits<{ submit: [] }>();

const draft = useSequenceDraftStore();

const motionTail = computed(() => sequenceMotionTailFrames(props.selectedModel));
const maxStages = computed(() => props.chainLimits?.max_stages ?? 16);
const fadeFramesMax = computed(() => props.chainLimits?.fade_frames_max ?? 32);
const newClipFrames = computed(() =>
  defaultClipFrames(props.selectedModel, props.chainLimits, motionTail.value),
);
const locked = computed(() => props.submitting || props.busy);
const imagePickerOpen = ref(false);
const advancedOpen = ref(false);
const activeClip = computed(
  () => draft.clips.find((clip) => clip.id === draft.activeClipId) ?? draft.clips[0] ?? null,
);
const activeIndex = computed(() =>
  activeClip.value ? draft.clips.findIndex((clip) => clip.id === activeClip.value?.id) : -1,
);
const advancedCount = computed(
  () =>
    Number(Boolean(draft.openingImage)) +
    Number(Boolean(activeClip.value?.negativePrompt.trim())) +
    Number(draft.enableAudio),
);

/** Durations are the 8n+1 grid up to the cap, strictly above the motion tail;
 *  an off-grid loaded value stays visible rather than silently re-snapping. */
function frameOptionsFor(frames: number): number[] {
  const options = sequenceFrameOptions(
    props.chainLimits?.frames_per_clip_cap ?? 97,
    motionTail.value,
  );
  if (!options.includes(frames)) options.push(frames);
  return options.sort((a, b) => a - b);
}

const stages = computed<SequenceStage[]>(() =>
  draft.clips.map((clip) => ({
    prompt: clip.prompt,
    frames: clip.frames,
    transition: clip.transition,
    fade_frames: clip.fadeFrames,
  })),
);
const duration = computed(() => sequenceDuration(stages.value, props.fps, motionTail.value));
const validation = computed(() =>
  sequenceValidation(stages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: props.chainLimits?.max_total_frames ?? Number.MAX_SAFE_INTEGER,
    motionTailFrames: motionTail.value,
  }),
);
/** Desktop parity: a model the server says can't chain names its reason
 *  inline instead of being filtered out of the picker without explanation. */
const unsupportedReason = computed(() =>
  props.chainLimits && props.chainLimits.supports_sequence === false
    ? (props.chainLimits.sequence_unsupported_reason ?? "This model can't render a clip sequence.")
    : null,
);
const blockingReason = computed(
  () =>
    unsupportedReason.value ??
    sequenceOpeningImageError(draft.openingImage, draft.mediaRestoring) ??
    validation.value[0] ??
    null,
);
const submitError = computed(() => (props.error ? friendlySequenceError(props.error) : ""));

const clipLabel = (index: number) => (index === 0 ? "opening" : `clip ${index + 1}`);

// ── Seam sheet ───────────────────────────────────────────────────────────────
const openSeamId = ref<string | null>(null);
const seamIndex = computed(() => draft.clips.findIndex((clip) => clip.id === openSeamId.value));
const seamClip = computed(() => draft.clips[seamIndex.value] ?? null);

function toggleSeam(id: string): void {
  openSeamId.value = openSeamId.value === id ? null : id;
}

function setSeamTransition(transition: SequenceTransition): void {
  if (openSeamId.value) draft.setTransition(openSeamId.value, transition);
}

function setSeamFade(frames: number): void {
  const clip = seamClip.value;
  if (clip) draft.setTransition(clip.id, clip.transition, frames);
}

function addClip(): void {
  if (draft.clips.length >= maxStages.value) return;
  draft.addClip(newClipFrames.value);
}

function removeClip(id: string): void {
  if (openSeamId.value === id) openSeamId.value = null;
  draft.removeClip(id);
}

function submit(): void {
  if (locked.value || blockingReason.value) return;
  emit("submit");
}

function setOpeningImage(image: MobilePickedImage): void {
  draft.openingImage = { filename: image.filename, base64: image.base64 };
  imagePickerOpen.value = false;
}

function sourceImageMime(filename: string): string {
  return /\.jpe?g$/i.test(filename.trim()) ? "image/jpeg" : "image/png";
}
</script>

<template>
  <section class="mobile-sequence" data-test="mobile-sequence-composer">
    <div class="mobile-sequence-clips">
      <template v-for="(clip, index) in draft.clips" :key="clip.id">
        <div v-if="index > 0" class="mobile-sequence-seam">
          <span class="mobile-sequence-seam-line" aria-hidden="true" />
          <SeamPill
            large
            :transition="clip.transition"
            :motion-tail="motionTail"
            :fade-frames="clip.fadeFrames"
            :active="openSeamId === clip.id"
            :disabled="locked"
            @click="toggleSeam(clip.id)"
          />
          <span class="mobile-sequence-seam-line" aria-hidden="true" />
        </div>
        <article class="mobile-sequence-clip" data-test="mobile-sequence-clip">
          <div class="mobile-sequence-clip-head">
            <strong>{{ index === 0 ? "Opening clip" : `Clip ${index + 1}` }}</strong>
            <button
              v-if="draft.clips.length > 2"
              type="button"
              data-test="mobile-sequence-remove"
              :aria-label="`Remove clip ${index + 1}`"
              :disabled="locked"
              @click="removeClip(clip.id)"
            >
              Remove
            </button>
          </div>
          <textarea
            v-model="clip.prompt"
            class="control mobile-sequence-prompt"
            :aria-label="index === 0 ? 'Opening clip prompt' : `Clip ${index + 1} prompt`"
            :placeholder="index === 0 ? 'How does the sequence begin?' : 'What happens next?'"
            :disabled="locked"
          />
          <details class="mobile-native-disclosure">
            <summary>
              <span>Clip tools</span>
              <small>{{ formatFrameDuration(clip.frames, fps) }}</small>
            </summary>
            <label class="field">
              <span>Duration</span>
              <select
                v-model.number="clip.frames"
                class="control"
                data-test="mobile-sequence-frames"
                :disabled="locked"
              >
                <option
                  v-for="frames in frameOptionsFor(clip.frames)"
                  :key="frames"
                  :value="frames"
                >
                  {{ formatFrameDuration(frames, fps) }}
                </option>
              </select>
            </label>
          </details>
        </article>
      </template>
    </div>

    <button
      class="secondary-button mobile-sequence-add"
      type="button"
      data-test="mobile-sequence-add"
      :disabled="locked || draft.clips.length >= maxStages"
      @click="addClip"
    >
      + Add clip
    </button>

    <p class="mobile-sequence-duration" data-test="mobile-sequence-duration">
      {{ formatFrameDuration(duration.frames, fps) }} @ {{ fps }}fps
    </p>

    <!-- Shared generation params are OWNED by the host form (one source of
         truth for both outputs); the composer only lends them a place to sit
         so the clips stay at the top of the phone's scroll. -->
    <details
      v-if="$slots.settings"
      class="mobile-native-disclosure"
      data-test="mobile-sequence-settings"
    >
      <summary>
        <span>Sequence settings</span>
        <small>{{ settingsSummary }}</small>
      </summary>
      <slot name="settings" />
    </details>

    <button
      type="button"
      class="secondary-button"
      data-test="mobile-sequence-open-advanced"
      @click="advancedOpen = true"
    >
      Advanced sequence controls
      <span v-if="advancedCount > 0">· {{ advancedCount }} on</span>
    </button>

    <p
      v-if="blockingReason"
      class="mobile-sequence-error"
      role="alert"
      data-test="mobile-sequence-error"
    >
      {{ blockingReason }}
    </p>
    <p
      v-if="submitError"
      class="mobile-sequence-error"
      role="alert"
      data-test="mobile-sequence-submit-error"
    >
      {{ submitError }}
    </p>

    <button
      class="primary-button mobile-sequence-generate"
      type="button"
      data-test="mobile-generate-sequence"
      :disabled="locked || !!blockingReason"
      @click="submit"
    >
      {{ submitting ? "Starting…" : "Generate sequence" }}
    </button>

    <MobileSeamSheet
      :open="!!seamClip"
      :transition="seamClip?.transition ?? 'smooth'"
      :fade-frames="seamClip?.fadeFrames ?? 8"
      :motion-tail="motionTail"
      :fps="fps"
      :fade-frames-max="fadeFramesMax"
      :from-label="clipLabel(Math.max(0, seamIndex - 1))"
      :to-label="clipLabel(Math.max(0, seamIndex))"
      @update:transition="setSeamTransition"
      @update:fade-frames="setSeamFade"
      @close="openSeamId = null"
    />
    <MobileAdvancedSheet
      :open="advancedOpen"
      :count="advancedCount"
      @close="advancedOpen = false"
      @reset="
        draft.openingImage = null;
        draft.enableAudio = false;
        draft.clips.forEach((clip) => (clip.negativePrompt = ''));
      "
    >
      <details class="mobile-native-disclosure" open data-test="mobile-sequence-advanced-opening">
        <summary>
          <span>Opening sequence image</span>
          <small>{{ draft.openingImage?.filename ?? "Original starting frame" }}</small>
        </summary>
        <img
          v-if="draft.openingImage?.base64"
          :src="
            base64ToDataUrl(
              draft.openingImage.base64 ?? '',
              sourceImageMime(draft.openingImage.filename),
            )
          "
          :alt="draft.openingImage.filename"
          data-test="mobile-sequence-source-preview"
        />
        <button
          type="button"
          class="secondary-button"
          data-test="mobile-sequence-source-pick"
          :disabled="locked || !target"
          @click="imagePickerOpen = true"
        >
          {{ draft.openingImage ? "Replace opening image" : "Attach opening image" }}
        </button>
        <button
          v-if="draft.openingImage"
          type="button"
          class="secondary-button"
          data-test="mobile-sequence-source-clear"
          :disabled="locked"
          @click="draft.openingImage = null"
        >
          Remove
        </button>
      </details>
      <label v-if="activeClip" class="field" data-test="mobile-sequence-advanced-negative">
        <span>Clip {{ activeIndex + 1 }} negative prompt</span>
        <input v-model="activeClip.negativePrompt" class="control" placeholder="Optional" />
      </label>
      <label
        v-if="chainLimits?.supports_audio"
        class="mobile-sequence-check"
        data-test="mobile-sequence-audio"
      >
        <input v-model="draft.enableAudio" type="checkbox" :disabled="locked" />
        Generate audio
      </label>
    </MobileAdvancedSheet>
    <MobileImagePickerSheet
      :open="imagePickerOpen"
      :target="target"
      @pick="setOpeningImage"
      @close="imagePickerOpen = false"
    />
  </section>
</template>

<style scoped>
.mobile-sequence {
  display: grid;
  gap: 14px;
}

.mobile-sequence-clip {
  display: grid;
  gap: 12px;
  padding: 14px;
  border: 1px solid var(--edge);
  border-radius: 16px;
  background: var(--bench);
}

.mobile-sequence-clips {
  display: grid;
  gap: 0;
}

.mobile-sequence-clip-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.mobile-sequence-clip-head button {
  min-height: 44px;
  color: var(--stop);
}

.mobile-sequence-prompt {
  min-height: 92px;
  resize: vertical;
  font-size: 16px;
}

.mobile-sequence-source {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.mobile-sequence-source img {
  width: 56px;
  height: 56px;
  object-fit: cover;
  border: 1px solid var(--edge);
  border-radius: 10px;
}

/* The seam owns the vertical run between two cards: short connector lines
   above and below keep the pill reading as a joint, not a floating chip. */
.mobile-sequence-seam {
  display: grid;
  justify-items: center;
  gap: 2px;
  padding: 2px 0;
}

.mobile-sequence-seam-line {
  width: 1px;
  height: 10px;
  background: var(--edge);
}

.mobile-sequence-add {
  width: 100%;
  min-height: 46px;
}

.mobile-sequence-generate {
  min-height: 46px;
}

.mobile-sequence-duration {
  margin: 0;
  color: var(--ink-3);
  font-family: var(--font-utility);
  font-size: var(--text-data);
}

.mobile-sequence-check {
  display: flex;
  min-height: 44px;
  align-items: center;
  gap: 10px;
}

.mobile-sequence-error {
  margin: 0;
  color: var(--stop);
  font-size: 14px;
}
</style>
