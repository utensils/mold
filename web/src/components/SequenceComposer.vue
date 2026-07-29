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
  chainScriptToClips,
  clipsToChainScript,
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
import { parseChainScript, serializeChainScript } from "@studio/lib/chainToml";
import { fetchChainLimits, type ChainLimits, type StreamTarget } from "../api";
import { toast } from "../lib/toasts";
import ImagePickerModal from "./ImagePickerModal.vue";
import type { SourceImageState } from "../types";
import type { ClipRailMedia } from "@ui/components/types";

const props = withDefaults(
  defineProps<{
    model: string;
    family: string;
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
const pickerOpen = ref(false);
const importFileInput = ref<HTMLInputElement | null>(null);

const motionTail = computed(() =>
  sequenceMotionTailFrames({ name: props.model, family: props.family }),
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
const validationErrors = computed(() =>
  sequenceValidation(stages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: limits.value?.max_total_frames ?? 1552,
    motionTailFrames: motionTail.value,
  }),
);
const duration = computed(() =>
  sequenceDuration(stages.value, props.shared.fps, motionTail.value),
);
const fitNote = computed(() => {
  const n = draft.clips.length;
  return `${n} ${n === 1 ? "clip" : "clips"} · ${formatFrameDuration(duration.value.frames, props.shared.fps)} @ ${props.shared.fps}fps`;
});

const frameOptions = computed(() =>
  sequenceFrameOptions(
    limits.value?.frames_per_clip_cap ?? 97,
    motionTail.value,
  ),
);

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
    validationErrors.value.length === 0,
);

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

// ── Opening image (clip 1 only — later clips condition on the prior
//    clip's motion tail) ───────────────────────────────────────────────
function onPickImage(images: SourceImageState[]) {
  const first = images[0];
  const clip = draft.clips[0];
  pickerOpen.value = false;
  if (!first || !clip) return;
  clip.sourceImage = { filename: first.filename, base64: first.base64 };
}

function clearOpeningImage() {
  const clip = draft.clips[0];
  if (clip) clip.sourceImage = null;
}

// ── File tools ────────────────────────────────────────────────────────
function exportScript(): string {
  return serializeChainScript(
    clipsToChainScript(props.shared, draft.clips, {
      motionTailFrames: motionTail.value,
      enableAudio: draft.enableAudio,
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
      <div class="ml-auto">
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
                v-if="clip.sourceImage?.base64"
                :src="`data:image/png;base64,${clip.sourceImage.base64}`"
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
        <template v-if="activeIndex === 0">
          <button
            type="button"
            class="sq-tool"
            data-test="opening-image-attach"
            @click="pickerOpen = true"
          >
            <Icon name="image" :size="13" />
            {{ activeClip.sourceImage?.filename ?? "Opening image" }}
          </button>
          <button
            v-if="activeClip.sourceImage"
            type="button"
            class="sq-tool"
            data-test="opening-image-clear"
            aria-label="Clear opening image"
            @click="clearOpeningImage"
          >
            <Icon name="close" :size="13" />
          </button>
        </template>
      </div>
      <details class="text-xs text-ink-3">
        <summary class="cursor-pointer select-none">
          Negative prompt<span v-if="activeClip.negativePrompt.trim()">
            · set</span
          >
        </summary>
        <textarea
          :value="activeClip.negativePrompt"
          data-test="clip-negative"
          rows="2"
          class="mt-2 w-full resize-y rounded-control border border-ce bg-bath px-3 py-2 text-sm text-rebate outline-none focus:border-safelight"
          placeholder="What to avoid in this clip"
          @input="
            activeClip.negativePrompt = (
              $event.target as HTMLTextAreaElement
            ).value
          "
        />
      </details>
    </div>

    <label
      v-if="limits?.supports_audio"
      class="flex cursor-pointer items-center gap-2 px-1 text-xs text-ink-2"
      title="Generate per-stage audio and mux it into the stitched MP4."
    >
      <input
        type="checkbox"
        data-test="sequence-enable-audio"
        class="h-4 w-4 rounded border-ce bg-bench text-safelight focus:ring-safelight"
        :checked="draft.enableAudio"
        @change="
          draft.enableAudio = ($event.target as HTMLInputElement).checked
        "
      />
      Generate audio
    </label>

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

    <button
      class="w-full rounded-control-lg py-2.5 text-sm font-semibold transition"
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

    <ImagePickerModal
      :open="pickerOpen"
      title="Opening image"
      @pick="onPickImage"
      @close="pickerOpen = false"
    />
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
