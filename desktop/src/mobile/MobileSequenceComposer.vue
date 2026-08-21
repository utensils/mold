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
import { computed, ref, watch } from "vue";
import SeamPill from "@ui/components/SeamPill.vue";
import {
  defaultClipFrames,
  formatFrameDuration,
  friendlySequenceError,
  sequenceClipFrameCap,
  sequenceDuration,
  sequenceFrameOptions,
  sequenceFrameStep,
  sequenceMotionTailFrames,
  sequenceValidation,
  transitionLabel,
  type SequenceStage,
  type SequenceTransition,
} from "@studio/lib/sequence";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  buildChainRequest,
  sequenceOpeningImageError,
  type SequenceSharedParams,
} from "@studio/lib/sequenceForm";
import { promptOptional } from "@studio/lib/promptRequirement";
import { cameraMotionMode } from "@studio/lib/cameraMotion";
import { parseSourceImageCapability } from "@studio/lib/sourceImageCapability";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import type { ApiTarget } from "../lib/api/client";
import type { Ltx2CameraControlInfo, ModelEntry } from "../lib/api/types";
import type { GenerateForm } from "../lib/generateForm";
import { generationCapabilitiesForFamily } from "../lib/capabilities";
import MobileSequenceOpeningImage from "./MobileSequenceOpeningImage.vue";
import MobileAdvancedSheet from "./MobileAdvancedSheet.vue";
import MobileSeamSheet from "./MobileSeamSheet.vue";
import { validateChain } from "@studio/api/chains";
import type { ChainValidationResponse } from "@studio/lib/api/chainTypes";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    selectedModel: ModelEntry | null;
    chainLimits: ChainLimits | null;
    target: ApiTarget | null;
    /** Live shared generation parameters owned by MobileApp's form. */
    shared: SequenceSharedParams;
    /** The generate form's frame rate — shown, never stored here. */
    fps: number;
    submitting?: boolean;
    /** A create/amend failure from the host, raw. */
    error?: string;
    /** A durable job is already starting on this host. */
    busy?: boolean;
    /** Collapsed caption for the shared-parameter disclosure. */
    settingsSummary?: string;
    cameraControls?: Ltx2CameraControlInfo[];
    cameraControlsLoaded?: boolean;
    upscalers?: ModelEntry[];
    cameraUnsupportedReason?: string | null;
  }>(),
  {
    target: null,
    submitting: false,
    error: "",
    busy: false,
    settingsSummary: "",
    cameraControls: () => [],
    cameraControlsLoaded: false,
    upscalers: () => [],
    cameraUnsupportedReason: null,
  },
);

const emit = defineEmits<{ submit: []; cancel: [] }>();

function submitOrCancel(): void {
  if (props.submitting) emit("cancel");
  else emit("submit");
}

const draft = useSequenceDraftStore();
const guidanceCaps = computed(() =>
  generationCapabilitiesForFamily(
    props.form.family,
    props.form.model,
    props.form.pipeline,
    props.selectedModel?.guidance_capabilities,
  ),
);

const motionTail = computed(() => sequenceMotionTailFrames(props.selectedModel));
const maxStages = computed(() => props.chainLimits?.max_stages ?? 16);
const fadeFramesMax = computed(() => props.chainLimits?.fade_frames_max ?? 32);
const newClipFrames = computed(() =>
  defaultClipFrames(props.selectedModel, props.chainLimits, motionTail.value),
);
const locked = computed(() => props.submitting || props.busy);
const advancedOpen = ref(false);
const activeClip = computed(
  () => draft.clips.find((clip) => clip.id === draft.activeClipId) ?? draft.clips[0] ?? null,
);
const activeIndex = computed(() =>
  activeClip.value ? draft.clips.findIndex((clip) => clip.id === activeClip.value?.id) : -1,
);
// The opening image left the Advanced sheet for the primary stack, so it is
// no longer one of the sheet's badge terms.
const advancedCount = computed(
  () =>
    Number(
      guidanceCaps.value.supportsNegativePrompt && Boolean(activeClip.value?.negativePrompt.trim()),
    ) + Number(Boolean(activeClip.value?.cameraControl)),
);
/** A checkpoint that reads no source image gets no opening-image well. An
 *  older server advertises nothing, which stays "unknown" and keeps it. */
const showOpeningImage = computed(
  () =>
    parseSourceImageCapability(
      props.selectedModel?.source_image ?? props.form.sourceImageCapability,
    ) !== "unsupported",
);

function setCameraMode(mode: string) {
  const clip = activeClip.value;
  if (!clip) return;
  if (mode === "custom") {
    if (cameraMotionMode(clip.cameraControl) !== "custom") clip.cameraControl = "";
  } else {
    clip.cameraControl = mode || null;
  }
}

/** Durations are the family's own grid (`8n+1`, `4n+1` for wan — #783) up to
 *  the cap, strictly above the motion tail; an off-grid loaded value stays
 *  visible rather than silently re-snapping. */
/** The model's own clip size bounds every picker — even against an older
 *  host that still advertises the family's single-request budget. */
const clipFrameCap = computed(() =>
  sequenceClipFrameCap(
    {
      name: props.selectedModel?.name ?? props.form.model,
      family: props.selectedModel?.family ?? props.form.family,
    },
    props.chainLimits,
  ),
);
function frameOptionsFor(frames: number): number[] {
  const options = sequenceFrameOptions(
    clipFrameCap.value,
    motionTail.value,
    props.selectedModel?.family ?? props.form.family,
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
// The opening image conditions clip 1, and every later clip inherits the
// previous clip's motion tail — the same handoff extend uses — so a
// promptless-capable family can render the whole sequence undescribed.
const clipPromptOptional = computed(() =>
  promptOptional({
    family: props.selectedModel?.family ?? null,
    sourceImage: draft.openingImage,
  }),
);
const validation = computed(() =>
  sequenceValidation(stages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: props.chainLimits?.max_total_frames ?? Number.MAX_SAFE_INTEGER,
    maxFramesPerClip: clipFrameCap.value,
    frameStep: sequenceFrameStep(props.selectedModel?.family),
    frameOffset: 1,
    motionTailFrames: motionTail.value,
    promptOptional: clipPromptOptional.value,
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
const validating = ref(false);
const validationPlan = ref<ChainValidationResponse | null>(null);
const validationError = ref("");
const validationSourceRevision = ref(0);
watch(
  () => [
    draft.openingImage?.base64 ?? null,
    ...draft.clips.map((clip) => clip.sourceImage?.base64 ?? null),
  ],
  () => {
    validationSourceRevision.value += 1;
  },
);
const validationInputSignature = computed(() =>
  JSON.stringify({
    shared: props.shared,
    motionTail: motionTail.value,
    enableAudio: draft.enableAudio,
    sourceRevision: validationSourceRevision.value,
    openingSourceFilename: draft.openingImage?.filename ?? null,
    clips: draft.clips.map((clip) => ({
      id: clip.id,
      prompt: clip.prompt,
      frames: clip.frames,
      transition: clip.transition,
      fadeFrames: clip.fadeFrames,
      negativePrompt: clip.negativePrompt,
      sourceFilename: clip.sourceImage?.filename ?? null,
      cameraControl: clip.cameraControl,
    })),
    target: props.target,
  }),
);
watch(validationInputSignature, () => {
  validationPlan.value = null;
  validationError.value = "";
});

async function validatePlan(): Promise<void> {
  const target = props.target;
  if (!target || blockingReason.value || locked.value || validating.value) return;
  const signature = validationInputSignature.value;
  validating.value = true;
  validationPlan.value = null;
  validationError.value = "";
  try {
    const request = buildChainRequest(props.shared, draft.clips, {
      motionTailFrames: motionTail.value,
      enableAudio: draft.enableAudio,
      openingImage: draft.openingImage,
    });
    const plan = await validateChain(request, target);
    if (validationInputSignature.value === signature) validationPlan.value = plan;
  } catch (error) {
    if (validationInputSignature.value === signature) {
      validationError.value = error instanceof Error ? error.message : String(error);
    }
  } finally {
    validating.value = false;
  }
}

const validationDuration = (plan: ChainValidationResponse) =>
  `${(plan.estimated_duration_ms / 1_000).toFixed(1)}s`;
const formatBytes = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(1)} GiB`;

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

    <button
      type="button"
      class="secondary-button mobile-sequence-validate"
      data-test="mobile-sequence-validate"
      :disabled="locked || validating || !!blockingReason || !target"
      @click="validatePlan"
    >
      {{ validating ? "Validating…" : "Validate plan" }}
    </button>
    <section
      v-if="validationPlan"
      class="mobile-sequence-plan"
      data-test="mobile-sequence-validation-plan"
      aria-live="polite"
    >
      <strong>
        Validated · {{ validationPlan.stage_count }} clips ·
        {{ validationPlan.estimated_total_frames }}f · {{ validationDuration(validationPlan) }}
      </strong>
      <span v-for="(stage, index) in validationPlan.stages" :key="index">
        Clip {{ index + 1 }} · {{ stage.frames }}f in / {{ stage.output_frames }}f out ·
        {{ transitionLabel(stage.transition, validationPlan.motion_tail_frames) }}
        <template v-if="stage.has_source_image">
          · {{ index === 0 ? "Opening image" : "Source image" }}
        </template>
        <template v-if="stage.has_negative_prompt"> · Negative prompt</template>
      </span>
      <span v-if="validationPlan.vram_estimate">
        VRAM {{ formatBytes(validationPlan.vram_estimate.worst_case_bytes) }} ·
        {{ validationPlan.vram_estimate.fits ? "fits" : "does not fit" }}
      </span>
      <span
        v-for="warning in validationPlan.warnings"
        :key="warning"
        class="mobile-sequence-warning"
      >
        {{ warning }}
      </span>
    </section>
    <p
      v-if="validationError"
      class="mobile-sequence-error"
      data-test="mobile-sequence-validation-error"
      role="alert"
    >
      {{ validationError }}
    </p>

    <!-- Source media rides in the primary stack for both outputs — one-shot
         keeps `mobile-source-disclosure` in exactly this seat, immediately
         above the shared parameters. -->
    <details
      v-if="showOpeningImage"
      class="mobile-native-disclosure"
      data-test="mobile-sequence-source-disclosure"
      :open="!!draft.openingImage"
    >
      <summary>
        <span>Opening sequence image</span>
        <small>{{ draft.openingImage?.filename ?? "Original starting frame" }}</small>
      </summary>
      <MobileSequenceOpeningImage
        :form="form"
        :upscalers="upscalers"
        :target="target"
        :locked="locked"
      />
    </details>

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

    <label
      v-if="chainLimits?.supports_audio"
      class="mobile-sequence-check"
      data-test="mobile-sequence-audio"
    >
      <input v-model="draft.enableAudio" type="checkbox" :disabled="locked" />
      Generate audio
    </label>

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
      :disabled="(locked || !!blockingReason) && !submitting"
      @click="submitOrCancel"
    >
      {{ submitting ? "Cancel · Preparing sequence…" : "Generate sequence" }}
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
        draft.clips.forEach((clip) => {
          clip.negativePrompt = '';
          clip.cameraControl = null;
        })
      "
    >
      <label v-if="activeClip" class="field" data-test="mobile-sequence-advanced-negative">
        <span>Clip {{ activeIndex + 1 }} negative prompt</span>
        <input
          v-model="activeClip.negativePrompt"
          class="control"
          placeholder="Optional"
          :disabled="!guidanceCaps.supportsNegativePrompt"
        />
        <small
          v-if="!guidanceCaps.supportsNegativePrompt"
          data-test="mobile-sequence-negative-unavailable-hint"
        >
          Saved for reuse, but this distilled recipe does not use negative-prompt guidance.
        </small>
      </label>
      <label
        v-if="activeClip && selectedModel?.family === 'ltx2'"
        class="field"
        data-test="mobile-sequence-advanced-camera"
      >
        <span>Clip {{ activeIndex + 1 }} camera motion</span>
        <select
          class="control"
          data-test="mobile-sequence-camera-motion"
          aria-label="Active clip camera motion"
          :disabled="locked"
          :value="cameraMotionMode(activeClip.cameraControl)"
          @change="setCameraMode(($event.target as HTMLSelectElement).value)"
        >
          <option value="">None</option>
          <option v-for="control in cameraControls" :key="control.id" :value="control.id">
            {{ control.label }}{{ control.installed ? "" : " · downloads on first use" }}
          </option>
          <option value="custom">Custom LoRA path…</option>
        </select>
      </label>
      <label
        v-if="
          activeClip &&
          selectedModel?.family === 'ltx2' &&
          cameraMotionMode(activeClip.cameraControl) === 'custom'
        "
        class="field"
      >
        <span>Camera LoRA path</span>
        <input
          v-model="activeClip.cameraControl"
          class="control"
          data-test="mobile-sequence-camera-motion-custom"
          aria-label="Active clip camera motion LoRA path"
          placeholder="/path/to/lora.safetensors"
          :disabled="locked"
        />
      </label>
      <p
        v-if="
          activeClip &&
          selectedModel?.family === 'ltx2' &&
          cameraControlsLoaded &&
          cameraControls.length === 0
        "
        class="mobile-sequence-camera-note"
        data-test="mobile-sequence-camera-motion-19b-hint"
      >
        {{
          cameraUnsupportedReason ??
          "Built-in camera motions are available for LTX-2 19B only. This model accepts a custom LoRA path."
        }}
      </p>
    </MobileAdvancedSheet>
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
.mobile-sequence-validate {
  min-height: 44px;
}
.mobile-sequence-plan {
  display: grid;
  gap: 6px;
  border: 1px solid color-mix(in srgb, var(--halide) 35%, var(--edge));
  border-radius: 12px;
  background: color-mix(in srgb, var(--halide) 8%, transparent);
  padding: 12px;
  font-family: var(--f-mono);
  font-size: 12px;
  line-height: 1.45;
}
.mobile-sequence-warning {
  color: var(--warning);
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

.mobile-sequence-camera-note {
  margin: 0;
  color: var(--ink-3);
  font-size: 12px;
  line-height: 1.45;
}

.mobile-sequence-error {
  margin: 0;
  color: var(--stop);
  font-size: 14px;
}
</style>
