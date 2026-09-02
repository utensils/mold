<script setup lang="ts">
/*
 * Sequence bench (mockup 1c): replaces the single-print ComposerCard when
 * Output = Sequence. Clip rail with seam pills, an active-clip prompt
 * editor, and a footer with file tools, validation, the fit note, and the
 * primary Generate/Update button. Clips live in the shared sequence draft
 * store; shared params stay in the generate form and are read at submit.
 */
import { computed, ref, watch } from "vue";
import ClipRail from "@ui/components/ClipRail.vue";
import Popover from "@ui/components/Popover.vue";
import SeamEditor from "@ui/components/SeamEditor.vue";
import ConfirmDialog from "../shell/ConfirmDialog.vue";
import { useSequenceDraftStore } from "@studio/stores/sequenceDraft";
import {
  defaultClipFrames,
  formatFrameDuration,
  sequenceDuration,
  sequenceClipFrameCap,
  sequenceFrameOptions,
  sequenceFrameStep,
  sequenceMotionTailFrames,
  sequenceValidation,
  transitionLabel,
  type SequenceStage,
} from "@studio/lib/sequence";
import {
  buildChainRequest,
  chainScriptToClips,
  clipsToChainScript,
  sequenceOpeningImageError,
  stageInvalidation,
} from "@studio/lib/sequenceForm";
import { promptOptional } from "@studio/lib/promptRequirement";
import { promptRecipeFromForm } from "../../lib/promptRecipe";
import ActionBlocker from "@ui/components/ActionBlocker.vue";
import { parseChainScript, serializeChainScript } from "@studio/lib/chainToml";
import type { ChainLimits } from "@studio/lib/api/chainTypes";
import { sequenceParams } from "../../lib/sequenceParams";
import type { GenerateForm } from "../../lib/generateForm";
import type { ModelEntry } from "../../lib/api/types";
import { useHostsStore } from "../../stores/hosts";
import { useToastStore } from "../../stores/toasts";
import { useContextMenuStore, type MenuEntry } from "../../stores/contextMenu";
import { clipContextEntries, railContextEntries } from "@studio/lib/sequenceContextMenu";
import type { ClipRailMedia } from "@ui/components/types";
import { validateChain } from "@studio/api/chains";
import type { ApiTarget } from "@studio/api/client";
import type { ChainValidationResponse } from "@studio/lib/api/chainTypes";

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
    /** Exact authenticated host that will render this sequence. */
    target?: ApiTarget | null;
  }>(),
  {
    selectedModel: null,
    chainLimits: null,
    installedModels: () => [],
    submitting: false,
    chainLevelDirty: false,
    stageMediaByClipId: null,
    playingClipId: null,
    target: null,
  },
);

const emit = defineEmits<{
  /** Generate sequence / Update sequence (create vs amend is the parent's call). */
  submit: [];
  /** Stop source preparation / placement before the sequence is queued. */
  cancel: [];
  /** Edit session: submit the current clips as a NEW job instead of amending. */
  duplicate: [];
  "play-clip": [clipId: string];
}>();

function submitOrCancel() {
  if (props.submitting) emit("cancel");
  else emit("submit");
}

const draft = useSequenceDraftStore();
const hosts = useHostsStore();
const toasts = useToastStore();
const contextMenu = useContextMenuStore();

const motionTail = computed(() => sequenceMotionTailFrames(props.selectedModel));
// Sequence stage images existed before the additive per-model contract, so an
// absent/unknown field remains compatible. Only explicit `unsupported` parks
// them while retaining the shared draft for a later model switch.
const supportsSourceImages = computed(
  () => (props.selectedModel?.source_image ?? props.form.sourceImageCapability) !== "unsupported",
);
const requestOpeningImage = computed(() =>
  supportsSourceImages.value ? draft.openingImage : null,
);
const requestClips = computed(() =>
  supportsSourceImages.value
    ? draft.clips
    : draft.clips.map((clip) => ({ ...clip, sourceImage: null })),
);
const maxStages = computed(() => props.chainLimits?.max_stages ?? 16);
const newClipFrames = computed(() =>
  defaultClipFrames(props.selectedModel, props.chainLimits, motionTail.value),
);
/** The model's own clip size bounds the picker and the validator alike —
 *  even against an older host that still advertises the family's
 *  single-request budget (481 LTX-2 frames at 24 fps). */
const clipFrameCap = computed(() =>
  sequenceClipFrameCap(
    {
      name: props.selectedModel?.name ?? props.form.model,
      family: props.selectedModel?.family ?? props.form.family,
    },
    props.chainLimits,
  ),
);
const frameOptions = computed(() => {
  const options = sequenceFrameOptions(
    clipFrameCap.value,
    motionTail.value,
    // Wan's VAE compresses time by 4, so its clips sit on `4k+1`; offering
    // the LTX grid hid its own 53-frame routing default (#783).
    props.selectedModel?.family ?? props.form.family,
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

// ── Validation, duration, fit ────────────────────────────────────────────────
const stages = computed<SequenceStage[]>(() =>
  draft.clips.map((clip) => ({
    prompt: clip.prompt,
    frames: clip.frames,
    transition: clip.transition,
    fade_frames: clip.fadeFrames,
  })),
);
// The opening image conditions clip 1, and every later clip inherits the
// previous clip's motion tail — the same handoff extend uses — so a
// promptless-capable family can render the whole sequence undescribed.
const clipPromptOptional = computed(() =>
  promptOptional({
    // The recipe is the authority; the family fields are the older host's
    // fallback. Keeping the rail on the same input the composer uses is what
    // stops the two disagreeing about a blank clip prompt.
    recipe: promptRecipeFromForm(props.form),
    family: props.form.family,
    sourceImage: requestOpeningImage.value,
  }),
);
const validation = computed(() =>
  sequenceValidation(stages.value, {
    maxStages: maxStages.value,
    maxTotalFrames: props.chainLimits?.max_total_frames ?? Number.MAX_SAFE_INTEGER,
    maxFramesPerClip: clipFrameCap.value,
    frameStep: sequenceFrameStep(props.selectedModel?.family ?? props.form.family),
    frameOffset: 1,
    motionTailFrames: motionTail.value,
    promptOptional: clipPromptOptional.value,
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
  const openingError = sequenceOpeningImageError(requestOpeningImage.value, draft.mediaRestoring);
  if (openingError) return openingError;
  return validation.value[0] ?? null;
});

const validating = ref(false);
const validationPlan = ref<ChainValidationResponse | null>(null);
const validationError = ref("");
const validationSourceRevision = ref(0);
watch(
  () => [
    requestOpeningImage.value?.base64 ?? null,
    ...requestClips.value.map((clip) => clip.sourceImage?.base64 ?? null),
  ],
  () => {
    validationSourceRevision.value += 1;
  },
);
const validationInputSignature = computed(() =>
  JSON.stringify({
    shared: sequenceParams(props.form, props.selectedModel),
    motionTail: motionTail.value,
    enableAudio: draft.enableAudio,
    sourceRevision: validationSourceRevision.value,
    openingSourceFilename: requestOpeningImage.value?.filename ?? null,
    clips: requestClips.value.map((clip) => ({
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

async function validatePlan() {
  const target = props.target;
  if (disabledReason.value || props.submitting || validating.value || !target) return;
  const signature = validationInputSignature.value;
  validating.value = true;
  validationPlan.value = null;
  validationError.value = "";
  try {
    const request = buildChainRequest(
      sequenceParams(props.form, props.selectedModel),
      requestClips.value,
      {
        motionTailFrames: motionTail.value,
        enableAudio: draft.enableAudio,
        openingImage: requestOpeningImage.value,
      },
    );
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
  draft.openingImage = editing.baselineOpeningImage ? { ...editing.baselineOpeningImage } : null;
  draft.enableAudio = editing.baselineEnableAudio ?? false;
  draft.stopEditing();
}

function submit() {
  if (disabledReason.value || props.submitting) return;
  emit("submit");
}

// ── Clear sequence ───────────────────────────────────────────────────────────
const clearConfirmOpen = ref(false);
const clearMessage = computed(() => {
  const edit = draft.editing ? " Ends the edit session without changing the finished job." : "";
  return `Removes all ${draft.clips.length} clips and their prompts.${edit} Model and shared settings stay.`;
});

function clearSequence() {
  clearConfirmOpen.value = false;
  openSeamId.value = null;
  draft.clearSequence(newClipFrames.value);
  toasts.push("Sequence cleared");
}

// ── File tools ───────────────────────────────────────────────────────────────
const fileToolsOpen = ref(false);
const tomlInput = ref<HTMLInputElement | null>(null);

function currentScript() {
  return clipsToChainScript(sequenceParams(props.form, props.selectedModel), requestClips.value, {
    motionTailFrames: motionTail.value,
    enableAudio: draft.enableAudio,
    openingImage: requestOpeningImage.value,
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
  await importTomlText(await file.text(), file.name);
}

async function importTomlText(text: string, filename = "sequence.toml") {
  try {
    const script = parseChainScript(text);
    const loaded = chainScriptToClips(script);
    draft.clips.splice(0, draft.clips.length, ...loaded.clips);
    draft.openingImage = loaded.openingImage;
    if (loaded.openingImage) {
      // Imported opening-image bytes are already the script's prepared source;
      // never inherit a stale client-only upscale policy from the prior draft.
      props.form.sourceFit = { mode: "crop-fill" };
    }
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
    if (shared.strength != null) props.form.strength = shared.strength;
    if (shared.seed != null) props.form.seed = shared.seed;
    if (
      shared.model &&
      props.installedModels.length > 0 &&
      !props.installedModels.some((m) => m.name === shared.model)
    ) {
      toasts.push(`Pull ${shared.model} first`);
    }
    toasts.push(`Loaded ${filename}`);
  } catch (err) {
    toasts.push(err instanceof Error ? err.message : String(err), "error");
  }
}

defineExpose({ importTomlText });

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

// ── Context menus ────────────────────────────────────────────────────────────
// One right-click handler for the whole bench, discriminated by target:
// text fields keep their own menu, a seam pill keeps its transition editor
// (SeamPill turns `contextmenu` into a `click`), a clip pill gets the clip
// menu, and everything else gets the bench menu. Entries come from the
// shared builder so web renders exactly the same actions.
const CONTEXT_MENU_TEXT_TARGETS = "textarea, input, select, [contenteditable], [data-selectable]";

function clipMenuEntries(clipId: string, index: number): MenuEntry[] {
  return clipContextEntries(
    {
      index,
      count: draft.clips.length,
      maxStages: maxStages.value,
      canPlay: Boolean(props.stageMediaByClipId?.[clipId]),
      locked: props.submitting,
    },
    {
      play: () => onPlayClip(clipId),
      duplicate: () => void draft.duplicateClip(clipId),
      insertBefore: () => void draft.insertClip(index, newClipFrames.value),
      insertAfter: () => void draft.insertClip(index + 1, newClipFrames.value),
      moveTo: (to) => draft.moveClip(clipId, to),
      remove: () => draft.removeClip(clipId),
    },
  );
}

function railMenuEntries(): MenuEntry[] {
  return railContextEntries(
    {
      count: draft.clips.length,
      maxStages: maxStages.value,
      locked: props.submitting,
      canValidate:
        disabledReason.value === null &&
        !props.submitting &&
        !validating.value &&
        props.target !== null,
    },
    {
      addClip: () => draft.addClip(newClipFrames.value),
      validate: () => void validatePlan(),
      importToml: () => tomlInput.value?.click(),
      exportToml: () => exportToml(),
      copyToml: () => void copyToml(),
      clear: () => (clearConfirmOpen.value = true),
    },
  );
}

function onBenchContextMenu(event: MouseEvent) {
  const node = event.target as HTMLElement | null;
  if (!node || typeof node.closest !== "function") return;
  if (node.closest(CONTEXT_MENU_TEXT_TARGETS)) return;
  if (node.closest(".ms-seam")) return;
  const clipId = node.closest("[data-clip-id]")?.getAttribute("data-clip-id") ?? null;
  if (clipId) {
    const index = draft.clips.findIndex((clip) => clip.id === clipId);
    if (index < 0) return;
    draft.activeClipId = clipId;
    contextMenu.open(event, clipMenuEntries(clipId, index));
    return;
  }
  contextMenu.open(event, railMenuEntries());
}
</script>

<template>
  <div data-test="sequence-composer" class="ms-seqbench" @contextmenu="onBenchContextMenu">
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
      @contextmenu="onBenchContextMenu"
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
        class="ms-seqbench__prompt ms-seqbench__prompt--main"
        :placeholder="
          activeIndex === 0 ? 'Describe the opening clip…' : 'Describe what happens next…'
        "
        aria-label="Clip prompt"
        @keydown="onPromptKeydown"
      />
    </div>

    <!-- Footer: file tools · audio · validation/fit · primary action -->
    <section
      v-if="validationPlan"
      class="ms-seqbench__plan"
      data-test="sequence-validation-plan"
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
      <span v-for="warning in validationPlan.warnings" :key="warning" class="ms-seqbench__warning">
        {{ warning }}
      </span>
    </section>
    <p
      v-if="validationError"
      class="ms-seqbench__validation-error"
      data-test="sequence-validation-error"
      role="alert"
    >
      {{ validationError }}
    </p>

    <div class="ms-seqbench__footer" data-test="sequence-composer-footer">
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

      <button
        type="button"
        data-test="sequence-validate"
        class="ms-seqbench__tool"
        :disabled="disabledReason !== null || submitting || validating || !target"
        @click="validatePlan"
      >
        {{ validating ? "Validating…" : "Validate plan" }}
      </button>

      <button
        type="button"
        data-test="sequence-clear"
        class="ms-seqbench__tool ms-seqbench__tool--danger"
        @click="clearConfirmOpen = true"
      >
        Clear sequence
      </button>

      <ActionBlocker
        v-if="disabledReason"
        data-test="sequence-validation"
        class="ms-seqbench__blocker"
        compact
        :reason="disabledReason"
        title="Before you generate"
      />
      <span v-else data-test="sequence-fit" class="ms-seqbench__note">{{ fitNote }}</span>

      <div class="ms-seqbench__spacer" />
      <button
        type="button"
        data-test="generate-sequence"
        class="ms-seqbench__generate"
        :disabled="disabledReason !== null && !submitting"
        @click="submitOrCancel"
      >
        {{
          submitting
            ? "Cancel · Preparing sequence…"
            : draft.editing
              ? "Update sequence"
              : "Generate sequence"
        }}
      </button>
    </div>

    <ConfirmDialog
      :open="clearConfirmOpen"
      title="Clear sequence?"
      :message="clearMessage"
      confirm-label="Clear sequence"
      danger
      @confirm="clearSequence"
      @cancel="clearConfirmOpen = false"
    />
  </div>
</template>

<style scoped>
.ms-seqbench {
  display: flex;
  flex-direction: column;
  gap: 10px;
  /*
   * The parent panel mounts this `flex: 1 1 0%`. Without an explicit
   * min-height the bench is floored at its own min-content — which counts
   * the rail's 204px preferred basis, not its 104px floor — so the panel
   * grew a scrollbar before the filmstrip's shrink weight ever engaged.
   * Zero lets the bench take exactly its protected parent shell's space and
   * flex the rail down for real. GenerateView gives that shell a 300px floor,
   * so Activity yields before the internal floors below or the Generate
   * button can clip.
   */
  min-height: 0;
  border-top: 1px solid var(--edge);
  background: var(--bench);
  padding: 12px 22px 14px;
}

.ms-seqbench__banner {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  gap: 10px;
  border: 1px solid color-mix(in srgb, var(--safelight) 45%, var(--ce));
  background: color-mix(in srgb, var(--safelight) 7%, transparent);
  border-radius: 9px;
  padding: 8px 12px;
}
.ms-seqbench__plan,
.ms-seqbench__validation-error {
  display: grid;
  flex-shrink: 0;
  gap: 3px;
  max-height: 112px;
  overflow: auto;
  border: 1px solid color-mix(in srgb, var(--halide) 35%, var(--ce));
  border-radius: 8px;
  background: color-mix(in srgb, var(--halide) 8%, transparent);
  padding: 7px 10px;
  font-family: var(--f-mono);
  font-size: 10px;
  color: var(--ink-2);
}
.ms-seqbench__validation-error,
.ms-seqbench__warning {
  color: var(--warning);
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

/*
 * The filmstrip absorbs bench resizes first: an outsized shrink weight pulls
 * the rail from its preferred 204px basis down to a hard floor before any
 * other row gives, so a shorter bench compresses thumbnails (fluid cqh
 * geometry inside ClipRail) instead of growing a scrollbar. The preferred
 * height MUST be the flex basis, not a `height` — a specified height becomes
 * the wrapper's min-content contribution, which propagates up as the
 * column's minimum and re-creates the scrollbar this exists to prevent.
 */
.ms-seqbench__railwrap {
  display: flex;
  width: 100%;
  flex: 0 999 204px;
  min-height: 104px;
}
.ms-seqbench__railwrap :deep(.ms-popover__trigger) {
  display: flex;
  width: 100%;
  min-width: 0;
  height: 100%;
}
/* Descendant selector outranks ClipRail's own `height: 188px` regardless of
   stylesheet injection order. */
.ms-seqbench .ms-seqbench__rail {
  flex: 1;
  min-width: 0;
  height: 100%;
  padding: 2px 0;
}

.ms-seqbench__clip {
  display: flex;
  flex-direction: column;
  flex: 1;
  /* Head (28) + gaps (12) + tools (28) + the prompt's 48px floor: below
     this the editor's own rows would start clipping. */
  min-height: 116px;
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
.ms-seqbench__prompt--main {
  flex: 1;
  min-height: 48px;
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
  flex-shrink: 0;
  gap: 12px;
  margin-top: auto;
  padding-top: 2px;
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
.ms-seqbench__blocker {
  max-width: min(380px, 38vw);
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
