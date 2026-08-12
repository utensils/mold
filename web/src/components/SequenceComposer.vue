<script setup lang="ts">
/*
 * Sequence composer (mockup 1c/3a) — web's sequence bench inside the
 * composer card chrome. Clips live in the shared sequence draft store; the
 * shared generation params (model/shape/detail/seed/fps) are PROPS projected
 * from the live generate form, so the inspector genuinely drives the
 * sequence — the stale private-copy bug cannot come back.
 */
import { computed, onMounted, ref, watch } from "vue";
import ClipRail from "@ui/components/ClipRail.vue";
import Popover from "@ui/components/Popover.vue";
import SeamEditor from "@ui/components/SeamEditor.vue";
import Icon from "@ui/components/Icon.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  buildChainRequest,
  chainScriptToClips,
  clipsToChainScript,
  sequenceOpeningImageError,
  stageInvalidation,
  type SequenceSharedParams,
} from "@studio/lib/sequenceForm";
import {
  defaultClipFrames,
  formatFrameDuration,
  sequenceDuration,
  sequenceFrameOptions,
  sequenceMotionTailFrames,
  sequenceValidation,
  transitionLabel,
  type SequenceStage,
} from "@studio/lib/sequence";
import { promptOptional } from "@studio/lib/promptRequirement";
import { parseChainScript, serializeChainScript } from "@studio/lib/chainToml";
import {
  fetchChainLimits,
  validateChain,
  type ChainLimits,
  type ChainValidationResponse,
  type StreamTarget,
} from "../api";
import { requestConfirm, toast } from "../lib/toasts";
import type { ClipRailMedia } from "@ui/components/types";

const props = withDefaults(
  defineProps<{
    model: string;
    family: string;
    /**
     * Selected model's advertised `source_image` contract. Wan's seam carries
     * context only for an image-conditioned checkpoint (#783), so the family
     * alone cannot name the transition; absent reads as "unknown" and takes
     * the conservative independent-clip path.
     */
    sourceImage?: string | null;
    /** LIVE shared params from the generate form (submit + TOML export). */
    shared: SequenceSharedParams;
    /** `/api/models` default_frames for the selected model, when known. */
    modelDefaultFrames?: number | null;
    /** Host the sequence will run on — chain-limits are per host. */
    target?: StreamTarget;
    /** Edit sessions: shape/detail/seed changed since the job was loaded. */
    chainLevelDirty?: boolean;
    /** Immutable durable-stage media, index-aligned with the submitted job. */
    stageMediaByClipId?: Readonly<
      Record<string, ClipRailMedia | undefined>
    > | null;
    playingClipId?: string | null;
  }>(),
  {
    sourceImage: null,
    modelDefaultFrames: null,
    chainLevelDirty: false,
    stageMediaByClipId: null,
    playingClipId: null,
  },
);

const emit = defineEmits<{
  submit: [];
  "duplicate-as-new": [];
  "discard-edit": [];
  "expand-clip": [clipId: string, prompt: string];
  "import-shared": [shared: Partial<SequenceSharedParams>];
  "play-clip": [clipId: string];
}>();

const draft = useSequenceDraftStore();

const limits = ref<ChainLimits | null>(null);
const limitsLoaded = ref(false);
const openSeamId = ref<string | null>(null);
const fileToolsOpen = ref(false);
const validating = ref(false);
const validationPlan = ref<ChainValidationResponse | null>(null);
const validationError = ref("");

/** Confirmed full clear: back to two fresh clips, staying in Sequence. */
async function clearSequence() {
  const edit = draft.editing
    ? " Ends the edit session without changing the finished job."
    : "";
  const accepted = await requestConfirm({
    title: "Clear sequence?",
    body: `Removes all ${draft.clips.length} clips and their prompts.${edit} Model and shared settings stay.`,
    confirmLabel: "Clear sequence",
    danger: true,
  });
  if (!accepted) return;
  openSeamId.value = null;
  draft.clearSequence(defaultFrames.value);
  toast("info", "Sequence cleared");
}
const importFileInput = ref<HTMLInputElement | null>(null);

const motionTail = computed(() =>
  sequenceMotionTailFrames({
    name: props.model,
    family: props.family,
    source_image: props.sourceImage,
  }),
);
const defaultFrames = computed(() =>
  defaultClipFrames(
    { default_frames: props.modelDefaultFrames },
    limits.value,
    motionTail.value,
  ),
);

async function loadLimits() {
  limitsLoaded.value = false;
  limits.value = await fetchChainLimits(props.model, props.target).catch(
    () => null,
  );
  limitsLoaded.value = true;
  // A non-AV model must not carry a stale audio request onto the wire.
  if (!limits.value?.supports_audio) draft.enableAudio = false;
}

onMounted(() => {
  draft.hydrate();
  draft.ensureClips(defaultFrames.value);
  void loadLimits();
});

watch(
  () => [props.model, props.family, props.target?.baseUrl],
  () => void loadLimits(),
);

// ── Active clip ───────────────────────────────────────────────────────
const activeIndex = computed(() => {
  const idx = draft.clips.findIndex((c) => c.id === draft.activeClipId);
  return idx >= 0 ? idx : 0;
});
const activeClip = computed(() => draft.clips[activeIndex.value] ?? null);

const clipCaption = computed(() => {
  const clip = activeClip.value;
  if (!clip) return "";
  const idx = activeIndex.value;
  const from =
    idx === 0
      ? "Opening clip"
      : `${transitionLabel(clip.transition, motionTail.value)} from clip ${idx}`;
  return `${formatFrameDuration(clip.frames, props.shared.fps)} · ${from}`;
});

// ── Rail wiring ───────────────────────────────────────────────────────
function applyOrder(ids: string[]) {
  ids.forEach((id, index) => draft.moveClip(id, index));
}

function resizeClip(id: string, frames: number) {
  const clip = draft.clips.find((candidate) => candidate.id === id);
  if (clip) clip.frames = frames;
}

const openSeamClip = computed(
  () => draft.clips.find((c) => c.id === openSeamId.value) ?? null,
);
const openSeamIndex = computed(() =>
  draft.clips.findIndex((c) => c.id === openSeamId.value),
);

function onSeamClick(id: string) {
  openSeamId.value = openSeamId.value === id ? null : id;
}

const editingPlans = computed(() => {
  const session = draft.editing;
  if (!session) return null;
  return stageInvalidation(session.baseline, draft.clips, {
    chainLevelDirty: props.chainLevelDirty,
    completedStages: session.completedStages,
  }).perClip;
});
function onPlayClip(clipId: string) {
  emit("play-clip", clipId);
}

// ── Validation / duration ─────────────────────────────────────────────
const stages = computed<SequenceStage[]>(() =>
  draft.clips.map((clip) => ({
    prompt: clip.prompt,
    frames: clip.frames,
    transition: clip.transition,
    fade_frames: clip.fadeFrames,
  })),
);
const maxStages = computed(() => limits.value?.max_stages ?? 16);
const clipPromptOptional = computed(() =>
  promptOptional({ family: props.family, sourceImage: draft.openingImage }),
);
const validationErrors = computed(() =>
  sequenceValidation(stages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: limits.value?.max_total_frames ?? 1552,
    motionTailFrames: motionTail.value,
    // The opening image conditions clip 1, and every later clip inherits the
    // previous clip's motion tail — the same handoff extend uses — so a
    // promptless-capable family can render the whole sequence undescribed.
    promptOptional: clipPromptOptional.value,
  }),
);
const duration = computed(() =>
  sequenceDuration(stages.value, props.shared.fps, motionTail.value),
);
const fitNote = computed(() => {
  const n = draft.clips.length;
  return `${n} ${n === 1 ? "clip" : "clips"} · ${formatFrameDuration(duration.value.frames, props.shared.fps)} @ ${props.shared.fps}fps`;
});

const frameOptions = computed(() => {
  const options = sequenceFrameOptions(
    limits.value?.frames_per_clip_cap ?? 97,
    motionTail.value,
    // Wan's VAE compresses time by 4, so its clips sit on `4k+1`; offering
    // the LTX grid hid its own 53-frame routing default (#783).
    props.family,
  );
  // A loaded value off the current model's grid must stay visible rather than
  // leave the select rendering blank — a reused or edited sequence carries a
  // clip length authored against another family's ladder. Same guard the
  // desktop and iPhone composers keep.
  const current = activeClip.value?.frames;
  if (current != null && !options.includes(current)) options.push(current);
  return options.sort((a, b) => a - b);
});

const sequenceUnsupported = computed(
  () => limits.value?.supports_sequence === false,
);
const chainUnavailable = computed(
  () => limitsLoaded.value && limits.value === null,
);
const canGenerate = computed(
  () =>
    limits.value !== null &&
    !sequenceUnsupported.value &&
    sequenceOpeningImageError(draft.openingImage, draft.mediaRestoring) ===
      null &&
    validationErrors.value.length === 0,
);

// Track source payload replacement separately so prompt edits do not repeatedly
// stringify multi-megabyte base64 values. This watcher depends only on the
// payload references, so ordinary clip edits leave it dormant.
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
    model: props.model,
    family: props.family,
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
    })),
    target: props.target,
  }),
);

watch(validationInputSignature, () => {
  validationPlan.value = null;
  validationError.value = "";
});

async function validatePlan() {
  if (!canGenerate.value || validating.value) return;
  const inputSignature = validationInputSignature.value;
  validating.value = true;
  validationPlan.value = null;
  validationError.value = "";
  try {
    const request = buildChainRequest(props.shared, draft.clips, {
      motionTailFrames: motionTail.value,
      enableAudio: draft.enableAudio,
      openingImage: draft.openingImage,
    });
    const plan = await validateChain(request, props.target);
    if (validationInputSignature.value === inputSignature) {
      validationPlan.value = plan;
    }
  } catch (error) {
    if (validationInputSignature.value === inputSignature) {
      validationError.value =
        error instanceof Error ? error.message : String(error);
    }
  } finally {
    validating.value = false;
  }
}

function validationDuration(plan: ChainValidationResponse): string {
  return `${(plan.estimated_duration_ms / 1_000).toFixed(1)}s`;
}

function formatBytes(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
}

function trySubmit() {
  if (!canGenerate.value) return;
  emit("submit");
}

function onPromptKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    trySubmit();
  }
}

// ── File tools ────────────────────────────────────────────────────────
function exportScript(): string {
  return serializeChainScript(
    clipsToChainScript(props.shared, draft.clips, {
      motionTailFrames: motionTail.value,
      enableAudio: draft.enableAudio,
      openingImage: draft.openingImage,
    }),
  );
}

function copyToml() {
  void navigator.clipboard.writeText(exportScript());
  fileToolsOpen.value = false;
}

function downloadToml() {
  const blob = new Blob([exportScript()], { type: "application/toml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "sequence.toml";
  a.click();
  URL.revokeObjectURL(url);
  fileToolsOpen.value = false;
}

function importTomlText(text: string) {
  try {
    const script = parseChainScript(text);
    const loaded = chainScriptToClips(script);
    draft.clips.splice(0, draft.clips.length, ...loaded.clips);
    draft.openingImage = loaded.openingImage;
    draft.enableAudio = loaded.enableAudio;
    draft.activeClipId = draft.clips[0]?.id ?? null;
    emit("import-shared", loaded.shared);
  } catch (error) {
    toast("error", error instanceof Error ? error.message : String(error));
  }
  fileToolsOpen.value = false;
}

function handleImportChange(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) {
    void file.text().then((text) => importTomlText(text));
  }
  input.value = "";
}

defineExpose({ importTomlText });
</script>

<template>
  <section
    class="flex flex-col gap-3 rounded-card-lg border border-edge bg-bench p-4 shadow-[inset_0_1px_0_var(--card-hi)]"
    data-test="sequence-composer"
  >
    <div class="flex items-center gap-2">
      <div>
        <span class="font-display text-sm font-semibold text-rebate"
          >Sequence</span
        >
        <p class="mt-0.5 text-xs text-ink-3">
          Tell the story one clip at a time.
        </p>
      </div>
      <div class="ml-auto flex items-center gap-2">
        <button
          type="button"
          class="sq-tool"
          data-test="sequence-clear"
          @click="clearSequence"
        >
          Clear sequence
        </button>
        <Popover
          :open="fileToolsOpen"
          placement="bottom-end"
          label="Sequence file tools"
          @update:open="fileToolsOpen = $event"
        >
          <template #trigger>
            <button
              type="button"
              class="sq-tool"
              data-test="sequence-file-tools"
              @click="fileToolsOpen = !fileToolsOpen"
            >
              <Icon name="settings" :size="13" />
              File tools
            </button>
          </template>
          <div class="flex flex-col gap-1">
            <button
              type="button"
              class="sq-menu-item"
              data-test="sequence-import-toml"
              @click="importFileInput?.click()"
            >
              <Icon name="upload" :size="13" />
              Import .toml
            </button>
            <button
              type="button"
              class="sq-menu-item"
              data-test="sequence-export-toml"
              @click="downloadToml"
            >
              <Icon name="download" :size="13" />
              Export .toml
            </button>
            <button
              type="button"
              class="sq-menu-item"
              data-test="sequence-copy-toml"
              @click="copyToml"
            >
              <Icon name="copy" :size="13" />
              Copy TOML
            </button>
          </div>
        </Popover>
        <input
          ref="importFileInput"
          type="file"
          accept=".toml"
          class="hidden"
          @change="handleImportChange"
        />
      </div>
    </div>

    <div
      v-if="draft.editing"
      class="flex flex-wrap items-center gap-2 rounded-control border border-halide/40 bg-halide/10 px-3 py-2 text-xs text-halide"
      data-test="sequence-edit-banner"
    >
      <span class="min-w-0 flex-1"
        >Editing sequence
        <span class="font-mono">{{ draft.editing.jobId }}</span> — completed
        clips stay cached where possible.</span
      >
      <button
        type="button"
        class="sq-tool"
        data-test="sequence-duplicate"
        @click="emit('duplicate-as-new')"
      >
        Duplicate as new
      </button>
      <button
        type="button"
        class="sq-tool"
        data-test="sequence-discard"
        @click="emit('discard-edit')"
      >
        Discard edit
      </button>
    </div>

    <div class="sq-filmstrip-wrap">
      <Popover
        :open="openSeamId !== null"
        placement="bottom-start"
        label="Transition"
        @update:open="(v: boolean) => (openSeamId = v ? openSeamId : null)"
      >
        <template #trigger>
          <ClipRail
            class="w-full"
            :clips="draft.clips"
            :active-id="draft.activeClipId"
            :motion-tail="motionTail"
            :max-stages="maxStages"
            :open-seam-id="openSeamId"
            :plans="editingPlans"
            :fps="shared.fps"
            :media-by-clip-id="stageMediaByClipId"
            :playing-id="playingClipId"
            :frame-options="frameOptions"
            @select="draft.activeClipId = $event"
            @add="draft.addClip(defaultFrames)"
            @remove="draft.removeClip($event)"
            @reorder="applyOrder"
            @resize="resizeClip"
            @seam-click="onSeamClick"
            @play="onPlayClip"
          >
            <template #thumb="{ clip }">
              <img
                v-if="
                  clip.id === draft.clips[0]?.id && draft.openingImage?.base64
                "
                :src="`data:image/png;base64,${draft.openingImage.base64}`"
                alt=""
                class="h-full w-full rounded object-cover"
              />
            </template>
          </ClipRail>
        </template>
        <SeamEditor
          v-if="openSeamClip"
          :transition="openSeamClip.transition"
          :fade-frames="openSeamClip.fadeFrames"
          :motion-tail="motionTail"
          :fps="shared.fps"
          :fade-frames-max="limits?.fade_frames_max ?? 32"
          :from-label="
            openSeamIndex === 1 ? 'opening' : `clip ${openSeamIndex}`
          "
          :to-label="`clip ${openSeamIndex + 1}`"
          :show-apply-all-hint="true"
          @update:transition="draft.setTransition(openSeamClip.id, $event)"
          @update:fade-frames="
            draft.setTransition(
              openSeamClip.id,
              openSeamClip.transition,
              $event,
            )
          "
          @apply-all="draft.applyTransitionToAllSeams($event)"
        />
      </Popover>
    </div>

    <div v-if="activeClip" class="flex flex-col gap-2">
      <div class="flex items-baseline justify-between gap-2">
        <span
          class="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-3"
          data-test="clip-caption"
          >CLIP {{ activeIndex + 1 }} OF {{ draft.clips.length }}</span
        >
        <span class="font-mono text-[10px] text-ink-3">{{ clipCaption }}</span>
      </div>
      <textarea
        :value="activeClip.prompt"
        data-test="clip-prompt"
        rows="3"
        class="w-full resize-y rounded-control border border-ce bg-bath px-3 py-2 text-sm text-rebate outline-none focus:border-safelight"
        :placeholder="
          activeIndex === 0
            ? 'Describe the opening clip'
            : `Describe clip ${activeIndex + 1}`
        "
        @input="
          activeClip.prompt = ($event.target as HTMLTextAreaElement).value
        "
        @keydown="onPromptKeydown"
      />
      <div class="flex flex-wrap items-center gap-2">
        <label class="flex items-center gap-2 text-xs text-ink-2">
          Duration
          <select
            :value="activeClip.frames"
            data-test="clip-frames"
            class="rounded-control border border-ce bg-bath px-2 py-1.5 font-mono text-xs text-rebate"
            @change="
              activeClip.frames = Number(
                ($event.target as HTMLSelectElement).value,
              )
            "
          >
            <option
              v-for="frames in frameOptions"
              :key="frames"
              :value="frames"
            >
              {{ formatFrameDuration(frames, shared.fps) }}
            </option>
          </select>
        </label>
        <button
          type="button"
          class="sq-tool"
          data-test="clip-expand"
          @click="emit('expand-clip', activeClip.id, activeClip.prompt)"
        >
          <Icon name="star" :size="13" />
          Expand
        </button>
      </div>
    </div>

    <p
      v-if="validationErrors.length"
      data-test="sequence-validation"
      class="rounded-control border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning"
      role="alert"
    >
      {{ validationErrors[0] }}
    </p>

    <p class="font-mono text-[11px] text-ink-3" data-test="sequence-fit-note">
      {{ fitNote }}
    </p>

    <section
      v-if="validationPlan"
      class="rounded-control border border-halide/30 bg-halide/10 px-3 py-2 text-xs text-ink-2"
      data-test="sequence-validation-plan"
      aria-live="polite"
    >
      <p class="font-semibold text-rebate">
        Validated · {{ validationPlan.stage_count }} clips ·
        {{ validationPlan.estimated_total_frames }}f ·
        {{ validationDuration(validationPlan) }}
      </p>
      <ol class="mt-2 space-y-1 font-mono text-[11px]">
        <li
          v-for="(stage, index) in validationPlan.stages"
          :key="index"
          class="flex flex-wrap gap-x-2"
        >
          <span class="text-rebate">Clip {{ index + 1 }}</span>
          <span>{{ stage.frames }}f input</span>
          <span>{{ stage.output_frames }}f output</span>
          <span>{{
            transitionLabel(stage.transition, validationPlan.motion_tail_frames)
          }}</span>
          <span v-if="stage.has_source_image">Opening image</span>
          <span v-if="stage.has_negative_prompt">Negative prompt</span>
        </li>
      </ol>
      <p v-if="validationPlan.vram_estimate" class="mt-2 font-mono text-[11px]">
        VRAM {{ formatBytes(validationPlan.vram_estimate.worst_case_bytes) }} ·
        {{ validationPlan.vram_estimate.fits ? "fits" : "does not fit" }}
      </p>
      <ul
        v-if="validationPlan.warnings.length"
        class="mt-2 list-disc space-y-1 pl-4 text-warning"
      >
        <li v-for="warning in validationPlan.warnings" :key="warning">
          {{ warning }}
        </li>
      </ul>
    </section>

    <p
      v-if="validationError"
      class="rounded-control border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning"
      data-test="sequence-validation-error"
      role="alert"
    >
      {{ validationError }}
    </p>

    <p
      v-if="!limitsLoaded && limits === null"
      class="text-center font-mono text-[11px] text-ink-3"
      data-test="chain-limits-pending"
    >
      checking sequence limits…
    </p>
    <p
      v-else-if="chainUnavailable"
      class="text-center font-mono text-[11px] text-warning"
      data-test="chain-unavailable"
    >
      this model can't chain sequences.
    </p>
    <p
      v-else-if="sequenceUnsupported"
      class="rounded border border-warning/30 bg-warning/10 px-3 py-2 text-center font-mono text-[11px] leading-relaxed text-warning"
      data-test="chain-pipeline-unsupported"
      role="alert"
    >
      {{
        limits?.sequence_unsupported_reason ??
        "This model's pipeline can't render sequences."
      }}
    </p>

    <div class="grid grid-cols-2 gap-2">
      <button
        type="button"
        class="rounded-control-lg border border-ce py-2.5 text-sm font-semibold text-ink-2 transition hover:border-halide hover:text-rebate disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="!canGenerate || validating"
        data-test="sequence-validate"
        @click="validatePlan"
      >
        {{ validating ? "Validating…" : "Validate plan" }}
      </button>
      <button
        class="rounded-control-lg py-2.5 text-sm font-semibold transition"
        :class="
          canGenerate
            ? 'bg-safelight text-on-accent hover:brightness-110'
            : 'cursor-not-allowed border border-edge bg-bath text-ink-3'
        "
        :disabled="!canGenerate"
        data-test="sequence-generate"
        @click="trySubmit"
      >
        {{ draft.editing ? "Update sequence" : "Generate sequence" }}
      </button>
    </div>
  </section>
</template>

<style scoped>
.sq-tool {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  padding: 6px 12px;
  border-radius: var(--radius-control);
  font-family: var(--f-mono);
  font-size: 11px;
  cursor: pointer;
  transition:
    border-color var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.sq-filmstrip-wrap {
  display: flex;
  width: 100%;
  min-width: 0;
}

.sq-filmstrip-wrap :deep(.ms-popover),
.sq-filmstrip-wrap :deep(.ms-popover__trigger) {
  display: flex;
  width: 100%;
  min-width: 0;
}
.sq-tool:hover {
  border-color: var(--safelight);
  color: var(--rebate);
}

.sq-menu-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  color: var(--ink-2);
  padding: 7px 10px;
  border-radius: var(--radius-control-sm);
  font-family: var(--f-mono);
  font-size: 11px;
  cursor: pointer;
  text-align: left;
}
.sq-menu-item:hover {
  background: var(--surface);
  color: var(--rebate);
}
</style>
