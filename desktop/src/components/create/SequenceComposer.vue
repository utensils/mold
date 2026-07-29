<script setup lang="ts">
/*
 * Sequence bench (mockup 1c): replaces the single-print ComposerCard when
 * Output = Sequence. Clip rail with seam pills, an active-clip prompt
 * editor, and a footer with file tools, validation, the fit note, and the
 * primary Generate/Update button. Clips live in the shared sequence draft
 * store; shared params stay in the generate form and are read at submit.
 */
import { computed, ref } from "vue";
import ClipRail from "@ui/components/ClipRail.vue";
import Popover from "@ui/components/Popover.vue";
import SeamEditor from "@ui/components/SeamEditor.vue";
import SwitchToggle from "@ui/components/SwitchToggle.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
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
import {
  chainScriptToClips,
  clipsToChainScript,
  stageInvalidation,
} from "@studio/lib/sequenceForm";
import { parseChainScript, serializeChainScript } from "@studio/lib/chainToml";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import { sequenceParams } from "../../lib/sequenceParams";
import type { GenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import type { ClipRailMedia } from "@ui/components/types";

const props = withDefaults(
  defineProps<{
    form: GenerateForm;
    selectedModel?: ModelEntry | null;
    chainLimits?: ChainLimits | null;
    installedModels?: ModelEntry[];
    /** True while the parent is POSTing the create/amend. */
    submitting?: boolean;
    /** Edit sessions: chain-level params differ from the loaded job's. */
    chainLevelDirty?: boolean;
    stageMediaByClipId?: Readonly<Record<string, ClipRailMedia | undefined>> | null;
    playingClipId?: string | null;
  }>(),
  {
    selectedModel: null,
    chainLimits: null,
    installedModels: () => [],
    submitting: false,
    chainLevelDirty: false,
    stageMediaByClipId: null,
    playingClipId: null,
  },
);

const emit = defineEmits<{
  /** Generate sequence / Update sequence (create vs amend is the parent's call). */
  submit: [];
  /** Edit session: submit the current clips as a NEW job instead of amending. */
  duplicate: [];
  "play-clip": [clipId: string];
}>();

const draft = useSequenceDraftStore();
const hosts = useHostsStore();
const toasts = useToastStore();

const motionTail = computed(() => sequenceMotionTailFrames(props.selectedModel));
const maxStages = computed(() => props.chainLimits?.max_stages ?? 16);
const newClipFrames = computed(() =>
  defaultClipFrames(props.selectedModel, props.chainLimits, motionTail.value),
);
const frameOptions = computed(() => {
  const options = sequenceFrameOptions(
    props.chainLimits?.frames_per_clip_cap ?? 97,
    motionTail.value,
  );
  // An off-grid loaded value must stay visible rather than mis-render.
  const current = activeClip.value?.frames;
  if (current != null && !options.includes(current)) options.push(current);
  return options.sort((a, b) => a - b);
});

// ── Active clip ──────────────────────────────────────────────────────────────
const activeIndex = computed(() => {
  const idx = draft.clips.findIndex((clip) => clip.id === draft.activeClipId);
  return idx >= 0 ? idx : 0;
});
const activeClip = computed(() => draft.clips[activeIndex.value] ?? null);
const activeMeta = computed(() => {
  const clip = activeClip.value;
  if (!clip) return "";
  const idx = activeIndex.value;
  const parts = [formatFrameDuration(clip.frames, props.form.fps)];
  if (idx > 0) {
    parts.push(`${transitionLabel(clip.transition, motionTail.value)} from clip ${idx}`);
  }
  return parts.join(" · ");
});
const negativeOpen = ref(false);

function onPromptKeydown(event: KeyboardEvent) {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    submit();
  }
}

function reorderClips(ids: string[]) {
  const byId = new Map(draft.clips.map((clip) => [clip.id, clip]));
  const next = ids.map((id) => byId.get(id)).filter((clip) => clip !== undefined);
  if (next.length === draft.clips.length) draft.clips.splice(0, draft.clips.length, ...next);
}

function resizeClip(id: string, frames: number) {
  const clip = draft.clips.find((candidate) => candidate.id === id);
  if (clip) clip.frames = frames;
}

// ── Seam editing (rail-anchored popover) ─────────────────────────────────────
const openSeamId = ref<string | null>(null);
const seamClip = computed(() => draft.clips.find((clip) => clip.id === openSeamId.value) ?? null);
const seamIndex = computed(() => draft.clips.findIndex((clip) => clip.id === openSeamId.value));
const clipLabel = (idx: number) => (idx === 0 ? "opening" : `clip ${idx + 1}`);

function onSeamClick(id: string) {
  openSeamId.value = openSeamId.value === id ? null : id;
}

function setSeamTransition(transition: "smooth" | "cut" | "fade") {
  if (openSeamId.value) draft.setTransition(openSeamId.value, transition);
}

function setSeamFade(frames: number) {
  const clip = seamClip.value;
  if (clip) draft.setTransition(clip.id, clip.transition, frames);
}

function applySeamToAll(transition: "smooth" | "cut" | "fade") {
  draft.applyTransitionToAllSeams(transition, seamClip.value?.fadeFrames);
  openSeamId.value = null;
}

// ── Opening image (clip 1) ───────────────────────────────────────────────────
const imageInput = ref<HTMLInputElement | null>(null);

async function onImageFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  const clip = draft.clips[0];
  if (!file || !clip) return;
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`Couldn't read ${file.name}`));
    reader.readAsDataURL(file);
  });
  clip.sourceImage = { filename: file.name, base64: dataUrl.replace(/^data:[^,]*,/, "") };
}

// ── Validation, duration, fit ────────────────────────────────────────────────
const stages = computed<SequenceStage[]>(() =>
  draft.clips.map((clip) => ({
    prompt: clip.prompt,
    frames: clip.frames,
    transition: clip.transition,
    fade_frames: clip.fadeFrames,
  })),
);
const validation = computed(() =>
  sequenceValidation(stages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: props.chainLimits?.max_total_frames ?? Number.MAX_SAFE_INTEGER,
    motionTailFrames: motionTail.value,
  }),
);
const duration = computed(() => sequenceDuration(stages.value, props.form.fps, motionTail.value));
const fitNote = computed(
  () =>
    `✓ fits · ${formatFrameDuration(duration.value.frames, props.form.fps)} @ ${props.form.fps}fps`,
);
const disabledReason = computed(() => {
  if (props.chainLimits && props.chainLimits.supports_sequence === false) {
    return (
      props.chainLimits.sequence_unsupported_reason ?? "This model can't render a clip sequence."
    );
  }
  if (!props.form.model) return "Pick a video model first.";
  return validation.value[0] ?? null;
});

// ── Edit sessions ────────────────────────────────────────────────────────────
const plan = computed(() => {
  const editing = draft.editing;
  if (!editing) return null;
  return stageInvalidation(editing.baseline, draft.clips, {
    chainLevelDirty: props.chainLevelDirty,
    completedStages: editing.completedStages,
  });
});
function onPlayClip(clipId: string) {
  emit("play-clip", clipId);
}
const editBanner = computed(() => {
  const editing = draft.editing;
  const current = plan.value;
  if (!editing || !current) return null;
  const cached = current.perClip.filter((p) => p === "cached").length;
  const rerender = current.perClip.length - cached;
  const host = hosts.all.find((h) => h.id === editing.hostId);
  return `Editing sequence ${editing.jobId.slice(0, 8)} on ${host?.label ?? editing.hostId} · ${cached} cached · ${rerender} will re-render`;
});

function discardEdit() {
  const editing = draft.editing;
  if (!editing) return;
  draft.clips.splice(0, draft.clips.length, ...editing.baseline.map((clip) => ({ ...clip })));
  draft.stopEditing();
}

function submit() {
  if (disabledReason.value || props.submitting) return;
  emit("submit");
}

// ── File tools ───────────────────────────────────────────────────────────────
const fileToolsOpen = ref(false);
const tomlInput = ref<HTMLInputElement | null>(null);

function currentScript() {
  return clipsToChainScript(sequenceParams(props.form, props.selectedModel), draft.clips, {
    motionTailFrames: motionTail.value,
    enableAudio: draft.enableAudio,
  });
}

function fileTool(action: () => void) {
  fileToolsOpen.value = false;
  action();
}

async function onTomlFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = ""; // allow re-picking the same file
  if (!file) return;
  try {
    const script = parseChainScript(await file.text());
    const loaded = chainScriptToClips(script);
    draft.clips.splice(0, draft.clips.length, ...loaded.clips);
    draft.activeClipId = draft.clips[0]?.id ?? null;
    draft.enableAudio = loaded.enableAudio;
    // Shared params flow to the LIVE generate form — the one source of truth.
    const shared = loaded.shared;
    if (shared.model) props.form.model = shared.model;
    if (shared.width != null) props.form.width = shared.width;
    if (shared.height != null) props.form.height = shared.height;
    if (shared.fps != null) props.form.fps = shared.fps;
    if (shared.steps != null) props.form.steps = shared.steps;
    if (shared.guidance != null) props.form.guidance = shared.guidance;
    if (shared.seed != null) props.form.seed = shared.seed;
    if (
      shared.model &&
      props.installedModels.length > 0 &&
      !props.installedModels.some((m) => m.name === shared.model)
    ) {
      toasts.push(`Pull ${shared.model} first`);
    }
    toasts.push(`Loaded ${file.name}`);
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}

function exportToml() {
  const blob = new Blob([serializeChainScript(currentScript())], { type: "application/toml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "chain.toml";
  a.click();
  URL.revokeObjectURL(url);
}

async function copyToml() {
  try {
    await navigator.clipboard.writeText(serializeChainScript(currentScript()));
    toasts.push("Copied chain.toml");
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}
</script>

<template>
  <div data-test="sequence-composer" class="ms-seqbench">
    <!-- Edit-session banner -->
    <div v-if="editBanner" data-test="edit-banner" class="ms-seqbench__banner">
      <span class="ms-seqbench__banner-text">{{ editBanner }}</span>
      <button
        type="button"
        data-test="edit-duplicate"
        class="ms-seqbench__banner-btn"
        @click="emit('duplicate')"
      >
        Duplicate as new
      </button>
      <button
        type="button"
        data-test="edit-discard"
        class="ms-seqbench__banner-btn"
        @click="discardEdit"
      >
        Discard edit
      </button>
    </div>

    <!-- Clip rail; the seam popover anchors to the rail itself -->
    <Popover
      :open="openSeamId !== null"
      placement="bottom-start"
      label="Transition editor"
      class="ms-seqbench__railwrap"
      @update:open="(open) => !open && (openSeamId = null)"
    >
      <template #trigger>
        <ClipRail
          class="ms-seqbench__rail"
          :clips="draft.clips"
          :active-id="draft.activeClipId"
          :motion-tail="motionTail"
          :max-stages="maxStages"
          :open-seam-id="openSeamId"
          :plans="plan?.perClip ?? null"
          :fps="form.fps"
          :media-by-clip-id="stageMediaByClipId"
          :playing-id="playingClipId"
          :frame-options="frameOptions"
          @select="draft.activeClipId = $event"
          @add="draft.addClip(newClipFrames)"
          @remove="draft.removeClip($event)"
          @reorder="reorderClips"
          @resize="resizeClip"
          @seam-click="onSeamClick"
          @play="onPlayClip"
        />
      </template>
      <SeamEditor
        v-if="seamClip"
        :transition="seamClip.transition"
        :fade-frames="seamClip.fadeFrames"
        :motion-tail="motionTail"
        :fps="form.fps"
        :fade-frames-max="chainLimits?.fade_frames_max ?? 32"
        :from-label="clipLabel(seamIndex - 1)"
        :to-label="clipLabel(seamIndex)"
        show-apply-all-hint
        @update:transition="setSeamTransition"
        @update:fade-frames="setSeamFade"
        @apply-all="applySeamToAll"
      />
    </Popover>

    <!-- Active clip editor -->
    <div v-if="activeClip" class="ms-seqbench__clip">
      <div class="ms-seqbench__cliphead">
        <span data-test="active-clip-caption" class="ms-seqbench__caption">
          CLIP {{ activeIndex + 1 }} OF {{ draft.clips.length }}
        </span>
        <span data-test="active-clip-meta" class="ms-seqbench__meta">{{ activeMeta }}</span>
        <div class="ms-seqbench__spacer" />
        <label class="ms-seqbench__frames">
          <span class="ms-seqbench__frames-label">Frames</span>
          <select
            v-model.number="activeClip.frames"
            data-test="clip-frames"
            class="ms-seqbench__select"
            aria-label="Clip frames"
          >
            <option v-for="frames in frameOptions" :key="frames" :value="frames">
              {{ formatFrameDuration(frames, form.fps) }}
            </option>
          </select>
        </label>
      </div>
      <textarea
        v-model="activeClip.prompt"
        data-test="clip-prompt"
        data-selectable
        rows="3"
        class="ms-seqbench__prompt"
        :placeholder="
          activeIndex === 0 ? 'Describe the opening clip…' : 'Describe what happens next…'
        "
        aria-label="Clip prompt"
        @keydown="onPromptKeydown"
      />
      <div class="ms-seqbench__cliptools">
        <template v-if="activeIndex === 0">
          <input
            ref="imageInput"
            type="file"
            accept="image/png,image/jpeg"
            class="hidden"
            @change="onImageFile"
          />
          <button
            type="button"
            data-test="opening-image-attach"
            class="ms-seqbench__tool"
            @click="imageInput?.click()"
          >
            {{
              activeClip.sourceImage
                ? `Image: ${activeClip.sourceImage.filename}`
                : "Attach opening image…"
            }}
          </button>
          <button
            v-if="activeClip.sourceImage"
            type="button"
            data-test="opening-image-clear"
            class="ms-seqbench__tool ms-seqbench__tool--danger"
            aria-label="Remove opening image"
            @click="activeClip.sourceImage = null"
          >
            ✕
          </button>
        </template>
        <button
          type="button"
          data-test="clip-negative-toggle"
          class="ms-seqbench__tool"
          :aria-expanded="negativeOpen"
          @click="negativeOpen = !negativeOpen"
        >
          {{ negativeOpen ? "Hide negative prompt" : "Negative prompt…" }}
        </button>
      </div>
      <textarea
        v-if="negativeOpen"
        v-model="activeClip.negativePrompt"
        data-test="clip-negative"
        data-selectable
        rows="2"
        class="ms-seqbench__prompt ms-seqbench__prompt--negative"
        placeholder="What to avoid in this clip…"
        aria-label="Clip negative prompt"
      />
    </div>

    <!-- Footer: file tools · audio · validation/fit · primary action -->
    <div class="ms-seqbench__footer">
      <input
        ref="tomlInput"
        type="file"
        accept=".toml,text/plain"
        class="hidden"
        @change="onTomlFile"
      />
      <Popover
        :open="fileToolsOpen"
        placement="top-start"
        label="File tools"
        @update:open="fileToolsOpen = $event"
      >
        <template #trigger>
          <button
            type="button"
            data-test="file-tools-toggle"
            class="ms-seqbench__tool"
            :aria-expanded="fileToolsOpen"
            aria-haspopup="menu"
            @click="fileToolsOpen = !fileToolsOpen"
          >
            File tools
          </button>
        </template>
        <div role="menu" class="ms-seqbench__menu" data-test="file-tools-menu">
          <button
            type="button"
            role="menuitem"
            class="ms-seqbench__menuitem"
            @click="fileTool(() => tomlInput?.click())"
          >
            Import .toml…
          </button>
          <button
            type="button"
            role="menuitem"
            class="ms-seqbench__menuitem"
            @click="fileTool(exportToml)"
          >
            Export .toml
          </button>
          <button
            type="button"
            role="menuitem"
            class="ms-seqbench__menuitem"
            @click="fileTool(() => void copyToml())"
          >
            Copy TOML
          </button>
        </div>
      </Popover>

      <label
        v-if="chainLimits?.supports_audio"
        class="ms-seqbench__audio"
        data-test="sequence-audio"
      >
        <SwitchToggle
          :model-value="draft.enableAudio"
          label="Generate audio"
          @update:model-value="draft.enableAudio = $event"
        />
        <span>Audio</span>
      </label>

      <span
        v-if="disabledReason"
        data-test="sequence-validation"
        role="alert"
        class="ms-seqbench__note ms-seqbench__note--blocked"
      >
        {{ disabledReason }}
      </span>
      <span v-else data-test="sequence-fit" class="ms-seqbench__note">{{ fitNote }}</span>

      <div class="ms-seqbench__spacer" />
      <button
        type="button"
        data-test="generate-sequence"
        class="ms-seqbench__generate"
        :disabled="disabledReason !== null || submitting"
        @click="submit"
      >
        {{ submitting ? "Starting…" : draft.editing ? "Update sequence" : "Generate sequence" }}
      </button>
    </div>
  </div>
</template>

<style scoped>
.ms-seqbench {
  display: flex;
  flex-direction: column;
  gap: 10px;
  border-top: 1px solid var(--edge);
  background: var(--bench);
  padding: 12px 22px 14px;
}

.ms-seqbench__banner {
  display: flex;
  align-items: center;
  gap: 10px;
  border: 1px solid color-mix(in srgb, var(--safelight) 45%, var(--ce));
  background: color-mix(in srgb, var(--safelight) 7%, transparent);
  border-radius: 9px;
  padding: 8px 12px;
}
.ms-seqbench__banner-text {
  flex: 1;
  min-width: 0;
  font-size: 12px;
  color: var(--ink-2);
}
.ms-seqbench__banner-btn {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 7px;
  padding: 4px 10px;
  font-size: 11px;
  cursor: pointer;
}
.ms-seqbench__banner-btn:hover {
  color: var(--rebate);
}

.ms-seqbench__railwrap {
  display: flex;
  width: 100%;
}
.ms-seqbench__railwrap :deep(.ms-popover__trigger) {
  display: flex;
  width: 100%;
  min-width: 0;
}
.ms-seqbench__rail {
  flex: 1;
  min-width: 0;
  padding: 2px 0;
}

.ms-seqbench__clip {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.ms-seqbench__cliphead {
  display: flex;
  align-items: center;
  gap: 10px;
}
.ms-seqbench__caption {
  font-family: var(--f-mono);
  font-size: 9px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-3);
}
.ms-seqbench__meta {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-3);
}
.ms-seqbench__spacer {
  flex: 1;
}
.ms-seqbench__frames {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.ms-seqbench__frames-label {
  font-size: 11px;
  color: var(--ink-3);
}
.ms-seqbench__select {
  height: 28px;
  border: 1px solid var(--ce);
  border-radius: 6px;
  background: var(--bath);
  color: var(--rebate);
  font-family: var(--f-mono);
  font-size: 12px;
  padding: 0 6px;
}
.ms-seqbench__prompt {
  width: 100%;
  resize: none;
  border: 1px solid var(--ce);
  border-radius: 9px;
  background: var(--bath);
  color: var(--rebate);
  font-size: 13px;
  line-height: 1.5;
  padding: 10px 12px;
}
.ms-seqbench__prompt:focus {
  outline: none;
  border-color: var(--safelight);
}
.ms-seqbench__prompt--negative {
  font-size: 12px;
}
.ms-seqbench__cliptools {
  display: flex;
  align-items: center;
  gap: 8px;
}
.ms-seqbench__tool {
  border: 1px solid var(--ce);
  background: transparent;
  color: var(--ink-2);
  border-radius: 7px;
  padding: 5px 10px;
  font-size: 11px;
  cursor: pointer;
}
.ms-seqbench__tool:hover {
  color: var(--rebate);
}
.ms-seqbench__tool--danger:hover {
  color: var(--stop);
  border-color: var(--stop);
}

.ms-seqbench__footer {
  display: flex;
  align-items: center;
  gap: 12px;
}
.ms-seqbench__menu {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 150px;
}
.ms-seqbench__menuitem {
  border: 0;
  background: transparent;
  color: var(--ink-2);
  text-align: left;
  padding: 7px 8px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
}
.ms-seqbench__menuitem:hover {
  background: var(--surface);
  color: var(--rebate);
}
.ms-seqbench__audio {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--ink-2);
}
.ms-seqbench__note {
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--halide);
}
.ms-seqbench__note--blocked {
  color: var(--stop);
}
.ms-seqbench__generate {
  height: 32px;
  border: 0;
  border-radius: 9px;
  background: var(--safelight);
  color: var(--on-accent, #fff);
  font-size: 13px;
  font-weight: 600;
  padding: 0 16px;
  cursor: pointer;
}
.ms-seqbench__generate:hover:not(:disabled) {
  filter: brightness(1.05);
}
.ms-seqbench__generate:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.hidden {
  display: none;
}
</style>
