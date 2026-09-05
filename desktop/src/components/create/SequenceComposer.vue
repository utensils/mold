<script setup lang="ts">
/*
 * The clip timeline (README §04), between the canvas and the composer: a
 * transport, a ruler, the scenes lane with its seams and playhead, and the
 * secondary controls a scene needs that the composer has no room for. The
 * composer below stays mounted and carries the selected scene's words and the
 * Generate button, so this file owns no prompt and no primary action — it
 * reports its own refusal upward instead and lets one button answer for both
 * modes. Scenes live in the shared sequence draft store; shared params stay in
 * the generate form and are read at submit.
 */
import { computed, ref, watch } from "vue";
import Icon from "@ui/components/Icon.vue";
import Popover from "@ui/components/Popover.vue";
import SeamEditor from "@ui/components/SeamEditor.vue";
import SceneLane from "./SceneLane.vue";
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
import { DESKTOP_SEQUENCE_WORDING } from "../../lib/sequenceWording";
import {
  buildChainRequest,
  chainScriptToClips,
  clipsToChainScript,
  sequenceOpeningImageError,
  stageInvalidation,
} from "@studio/lib/sequenceForm";
import { promptOptional } from "@studio/lib/promptRequirement";
import { promptRecipeFromForm } from "../../lib/promptRecipe";
import {
  SEQUENCE_NEEDS_STYLE,
  sceneLabel,
  type SequenceConfirmation,
} from "../../lib/sequenceTimeline";
import { formatGB } from "../../lib/format";
import { rulerTicks } from "../../lib/rulerTicks";
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
    /** How far into the playing scene its own player has run. */
    elapsedSeconds?: number;
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
    elapsedSeconds: 0,
    target: null,
  },
);

const emit = defineEmits<{
  /** Edit session: submit the current clips as a NEW job instead of amending. */
  duplicate: [];
  "play-clip": [clipId: string];
  /** What the composer's Generate must refuse for, or null when it may go. */
  "update:blockedReason": [reason: string | null];
  /** What the timeline is waiting to have confirmed, or null when nothing is. */
  "update:confirmation": [confirmation: SequenceConfirmation | null];
}>();

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
const canAdd = computed(() => draft.clips.length < maxStages.value);
/** "How does this work?" — the timeline explained in one paragraph. */
const helpOpen = ref(false);
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

// ── Active scene ─────────────────────────────────────────────────────────────
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
    parts.push(`${transitionLabel(clip.transition, motionTail.value)} from scene ${idx}`);
  }
  return parts.join(" · ");
});

/** The scene's own name, trimmed to fit one line of a dialog message. */
const SCENE_NAME_LIMIT = 42;
function sceneName(index: number): string {
  return sceneLabel(draft.clips[index]?.prompt ?? "", index, SCENE_NAME_LIMIT);
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

// ── Seam editing (lane-anchored popover) ─────────────────────────────────────
const openSeamId = ref<string | null>(null);
const seamClip = computed(() => draft.clips.find((clip) => clip.id === openSeamId.value) ?? null);
const seamIndex = computed(() => draft.clips.findIndex((clip) => clip.id === openSeamId.value));

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
    // fallback. Keeping the lane on the same input the composer uses is what
    // stops the two disagreeing about a blank scene's words.
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
    // Desktop says scene for every piece and clip for the whole.
    wording: DESKTOP_SEQUENCE_WORDING,
  }),
);
const duration = computed(() => sequenceDuration(stages.value, props.form.fps, motionTail.value));
const sceneCountLabel = computed(() =>
  draft.clips.length === 1 ? "1 scene" : `${draft.clips.length} scenes`,
);

/** `0:06` — the transport speaks in clock time, never in frames. */
function clockLabel(seconds: number): string {
  const whole = Math.max(0, Math.round(Number.isFinite(seconds) ? seconds : 0));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

const disabledReason = computed(() => {
  if (props.chainLimits && props.chainLimits.supports_sequence === false) {
    return props.chainLimits.sequence_unsupported_reason ?? "This style can't make a clip.";
  }
  if (!props.form.model) return SEQUENCE_NEEDS_STYLE;
  const openingError = sequenceOpeningImageError(requestOpeningImage.value, draft.mediaRestoring);
  if (openingError) return openingError;
  return validation.value[0] ?? null;
});
// The composer holds the one Generate button for both modes, so the timeline's
// own refusal has to reach it.
watch(disabledReason, (reason) => emit("update:blockedReason", reason), { immediate: true });

// ── Transport, ruler, playhead ───────────────────────────────────────────────
const playing = computed(() => props.playingClipId !== null);
const renderedClipIds = computed(() =>
  draft.clips
    .filter((clip) => props.stageMediaByClipId?.[clip.id]?.hasMedia)
    .map((clip) => clip.id),
);
/** Where the needle sits: everything before the playing scene, plus its own
 *  elapsed time. Nothing playing parks it at the start. */
const playheadSeconds = computed(() => {
  const index = draft.clips.findIndex((clip) => clip.id === props.playingClipId);
  if (index < 0) return 0;
  return (
    sequenceDuration(stages.value.slice(0, index), props.form.fps, motionTail.value).seconds +
    props.elapsedSeconds
  );
});
const playheadPercent = computed(() => {
  const total = duration.value.seconds;
  if (total <= 0) return 0;
  return Math.min(100, Math.max(0, (playheadSeconds.value / total) * 100));
});

function togglePlayback() {
  const current = props.playingClipId ?? renderedClipIds.value[0];
  if (current) emit("play-clip", current);
}

/** Ruler ticks: the coarsest round interval that still marks the clip out in
 *  a handful of steps, so the row never crowds at any clip length. The
 *  closing mark is pinned to the right edge rather than left:100%, where its
 *  label would paint past the strip and be cut by the bench's overflow. */
const ticks = computed(() => rulerTicks(duration.value.seconds, clockLabel));

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

/** The one-line truth under the lane: what the host says when it has been
 *  asked, and the draft's own arithmetic until then. */
const readout = computed(() => {
  const plan = validationPlan.value;
  if (!plan) {
    return `${sceneCountLabel.value} · ${duration.value.frames} frames · ${clockLabel(
      duration.value.seconds,
    )} at ${props.form.fps}fps`;
  }
  const parts = [
    plan.stage_count === 1 ? "1 scene" : `${plan.stage_count} scenes`,
    `${plan.estimated_total_frames} frames`,
    `${(plan.estimated_duration_ms / 1_000).toFixed(1)}s to render`,
  ];
  if (plan.vram_estimate) {
    parts.push(
      plan.vram_estimate.fits
        ? `${formatGB(plan.vram_estimate.worst_case_bytes)} of graphics memory`
        : `${formatGB(plan.vram_estimate.worst_case_bytes)} — more than this machine has`,
    );
  }
  return parts.join(" · ");
});

function stageLine(stage: ChainValidationResponse["stages"][number], index: number): string {
  const parts = [
    `Scene ${index + 1}`,
    `${stage.frames} → ${stage.output_frames} frames`,
    transitionLabel(stage.transition, validationPlan.value?.motion_tail_frames ?? 0),
  ];
  if (stage.has_source_image) parts.push(index === 0 ? "from your opening photo" : "from a photo");
  if (stage.has_negative_prompt) parts.push("with words to steer away from");
  return parts.join(" · ");
}

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
  return `Editing clip ${editing.jobId.slice(0, 8)} on ${host?.label ?? editing.hostId} · ${cached} cached · ${rerender} will re-render`;
});

function discardEdit() {
  const editing = draft.editing;
  if (!editing) return;
  draft.clips.splice(0, draft.clips.length, ...editing.baseline.map((clip) => ({ ...clip })));
  draft.openingImage = editing.baselineOpeningImage ? { ...editing.baselineOpeningImage } : null;
  draft.enableAudio = editing.baselineEnableAudio ?? false;
  draft.stopEditing();
}

// ── Removing and clearing ────────────────────────────────────────────────────
const removeCandidateId = ref<string | null>(null);
const clearConfirmOpen = ref(false);

function askRemoveScene(id: string) {
  if (draft.clips.length <= 2) return;
  removeCandidateId.value = id;
}

/** Delete on a scene the floor keeps. The lane has already swallowed the key,
 *  so the refusal has to say itself — silence read as a dead keyboard. */
function sceneRemoveBlocked() {
  toasts.push("A clip keeps at least two scenes — clear the clip to start over.");
}

function removeScene() {
  if (removeCandidateId.value) draft.removeClip(removeCandidateId.value);
  removeCandidateId.value = null;
}

function clearSequence() {
  clearConfirmOpen.value = false;
  openSeamId.value = null;
  draft.clearSequence(newClipFrames.value, props.chainLimits?.supports_audio);
  toasts.push("Clip cleared");
}

/** The timeline decides what it needs confirmed; the workbench renders it. */
const confirmation = computed<SequenceConfirmation | null>(() => {
  const index = draft.clips.findIndex((clip) => clip.id === removeCandidateId.value);
  if (index >= 0) {
    return {
      title: "Remove this scene?",
      message: `Removes “${sceneName(index)}” and its words. The scenes around it join up.`,
      confirmLabel: "Remove the scene",
      confirm: removeScene,
      cancel: () => (removeCandidateId.value = null),
    };
  }
  if (!clearConfirmOpen.value) return null;
  const edit = draft.editing ? " Ends the edit session without changing the finished job." : "";
  return {
    title: "Clear the clip?",
    message: `Removes all ${draft.clips.length} scenes and their words.${edit} The style and shared settings stay.`,
    confirmLabel: "Clear the clip",
    confirm: clearSequence,
    cancel: () => (clearConfirmOpen.value = false),
  };
});
watch(confirmation, (pending) => emit("update:confirmation", pending), { immediate: true });

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
// One right-click handler for the whole timeline, discriminated by target:
// text fields keep their own menu, a seam pill keeps its transition editor
// (SeamPill turns `contextmenu` into a `click`), a scene block gets the scene
// menu, and everything else gets the timeline menu. Entries come from the
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
      remove: () => askRemoveScene(clipId),
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
  <div data-test="sequence-composer" class="ms-timeline" @contextmenu="onBenchContextMenu">
    <!-- Edit-session banner -->
    <div v-if="editBanner" data-test="edit-banner" class="ms-timeline__banner">
      <span class="ms-timeline__banner-text">{{ editBanner }}</span>
      <button
        type="button"
        data-test="edit-duplicate"
        class="ms-toolbar-button"
        @click="emit('duplicate')"
      >
        Duplicate as new
      </button>
      <button type="button" data-test="edit-discard" class="ms-toolbar-button" @click="discardEdit">
        Discard edit
      </button>
    </div>

    <!-- Transport: play the whole clip, where it has got to, how it is built -->
    <div class="ms-timeline__transport">
      <button
        type="button"
        data-test="timeline-play"
        class="ms-timeline__play"
        :disabled="renderedClipIds.length === 0"
        :title="
          renderedClipIds.length === 0
            ? 'Nothing to play yet — generate the clip first'
            : playing
              ? 'Stop and go back to the live render'
              : 'Play the whole clip'
        "
        :aria-label="playing ? 'Stop playing' : 'Play the whole clip'"
        @click="togglePlayback"
      >
        <Icon :name="playing ? 'pause' : 'play'" :size="12" :stroke-width="2.5" />
      </button>
      <span class="font-mono text-xs text-fg" data-test="sequence-length">
        {{ clockLabel(playheadSeconds) }}
        <span class="text-fg-dim">/ {{ clockLabel(duration.seconds) }}</span>
      </span>
      <span class="ms-timeline__rule" aria-hidden="true" />
      <span class="text-xs text-fg-2">{{ sceneCountLabel }}, played end to end</span>
      <span class="ms-timeline__spacer" />
      <button
        type="button"
        data-test="timeline-help"
        class="cursor-pointer text-micro font-medium text-accent"
        :aria-expanded="helpOpen"
        @click="helpOpen = !helpOpen"
      >
        How does this work?
      </button>
      <button
        type="button"
        data-test="add-scene"
        class="ms-toolbar-button h-6 text-micro"
        :disabled="!canAdd"
        @click="draft.addClip(newClipFrames)"
      >
        <Icon name="plus" :size="12" :stroke-width="2" />
        Add a scene
      </button>
    </div>
    <div v-if="helpOpen" class="ms-timeline__help" data-test="timeline-help-text">
      <span class="font-mono text-xs text-accent">•</span>
      <span class="text-micro leading-body text-fg-2" style="text-wrap: pretty">
        Each block is one scene, written in your own words, and it is as wide as the time it plays.
        Drag the selected block's right edge to make that scene longer, drag the block itself to
        reorder, and click the marker between two blocks to say how they should meet. The whole
        thing is made as one continuous clip.
      </span>
    </div>

    <!-- Ruler, left-padded to clear the lane's own label -->
    <div class="ms-timeline__ruler" aria-hidden="true">
      <div class="ms-timeline__ticks">
        <span
          v-for="tick in ticks"
          :key="tick.at"
          class="ms-timeline__tick"
          :class="{ 'ms-timeline__tick--end': tick.atEnd }"
          :style="tick.style"
        >
          <span class="ms-timeline__tick-label">{{ tick.label }}</span>
          <span class="ms-timeline__tick-mark" />
        </span>
      </div>
    </div>

    <!-- Scenes lane; the seam popover anchors to the lane row itself -->
    <Popover
      :open="openSeamId !== null"
      placement="bottom-start"
      label="Transition editor"
      class="ms-timeline__lanewrap"
      @update:open="(open) => !open && (openSeamId = null)"
    >
      <template #trigger>
        <div class="ms-timeline__lane-label">
          <span class="text-xs font-semibold text-fg">Scenes</span>
          <span class="text-micro text-fg-dim">drag to trim</span>
        </div>
        <div class="ms-timeline__lane-area">
          <SceneLane
            :clips="draft.clips"
            :active-id="draft.activeClipId"
            :motion-tail="motionTail"
            :fps="form.fps"
            :open-seam-id="openSeamId"
            :plans="plan?.perClip ?? null"
            :media-by-clip-id="stageMediaByClipId"
            :playing-id="playingClipId"
            :frame-options="frameOptions"
            :disabled="submitting"
            @select="draft.activeClipId = $event"
            @remove="askRemoveScene"
            @remove-blocked="sceneRemoveBlocked"
            @reorder="reorderClips"
            @resize="resizeClip"
            @seam-click="onSeamClick"
          />
          <span
            v-if="renderedClipIds.length > 0"
            class="ms-timeline__playhead"
            data-test="timeline-playhead"
            aria-hidden="true"
            :style="{ left: `${playheadPercent}%` }"
          >
            <span class="ms-timeline__playhead-handle" />
          </span>
        </div>
      </template>
      <SeamEditor
        v-if="seamClip"
        :transition="seamClip.transition"
        :fade-frames="seamClip.fadeFrames"
        :motion-tail="motionTail"
        :fps="form.fps"
        :fade-frames-max="chainLimits?.fade_frames_max ?? 32"
        :from-label="sceneName(seamIndex - 1)"
        :to-label="sceneName(seamIndex)"
        show-apply-all-hint
        @update:transition="setSeamTransition"
        @update:fade-frames="setSeamFade"
        @apply-all="applySeamToAll"
      />
    </Popover>

    <!-- The selected scene's own controls, and what the clip adds up to -->
    <div v-if="activeClip" class="ms-timeline__scene">
      <span data-test="active-clip-caption" class="ms-group-label uppercase">
        Scene {{ activeIndex + 1 }} of {{ draft.clips.length }}
      </span>
      <span data-test="active-clip-meta" class="font-mono text-micro text-fg-dim">{{
        activeMeta
      }}</span>
      <span class="ms-timeline__spacer" />
      <label class="ms-timeline__frames">
        <span class="text-xs text-fg-dim">Length</span>
        <select
          v-model.number="activeClip.frames"
          data-test="clip-frames"
          class="ms-timeline__select"
          aria-label="How long this scene runs"
        >
          <option v-for="frames in frameOptions" :key="frames" :value="frames">
            {{ formatFrameDuration(frames, form.fps) }}
          </option>
        </select>
      </label>
    </div>

    <div class="ms-timeline__foot">
      <p class="ms-timeline__readout" data-test="sequence-fit" aria-live="polite">{{ readout }}</p>
      <span class="ms-timeline__spacer" />
      <input
        ref="tomlInput"
        type="file"
        accept=".toml,text/plain"
        class="ms-timeline__file"
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
            class="ms-toolbar-button"
            :aria-expanded="fileToolsOpen"
            aria-haspopup="menu"
            @click="fileToolsOpen = !fileToolsOpen"
          >
            File tools
          </button>
        </template>
        <div role="menu" class="ms-timeline__menu" data-test="file-tools-menu">
          <button
            type="button"
            role="menuitem"
            class="ms-timeline__menuitem"
            @click="fileTool(() => tomlInput?.click())"
          >
            Import .toml…
          </button>
          <button
            type="button"
            role="menuitem"
            class="ms-timeline__menuitem"
            @click="fileTool(exportToml)"
          >
            Export .toml
          </button>
          <button
            type="button"
            role="menuitem"
            class="ms-timeline__menuitem"
            @click="fileTool(() => void copyToml())"
          >
            Copy TOML
          </button>
        </div>
      </Popover>

      <button
        type="button"
        data-test="sequence-validate"
        class="ms-toolbar-button"
        :disabled="disabledReason !== null || submitting || validating || !target"
        @click="validatePlan"
      >
        {{ validating ? "Checking…" : "Check the plan" }}
      </button>

      <button
        type="button"
        data-test="sequence-clear"
        class="ms-toolbar-button ms-toolbar-button--danger-hover"
        @click="clearConfirmOpen = true"
      >
        Clear the clip
      </button>
    </div>

    <!-- The host's own plan, scene by scene, or its refusal -->
    <section
      v-if="validationPlan"
      class="ms-timeline__plan"
      data-test="sequence-validation-plan"
      aria-live="polite"
    >
      <span v-for="(stage, index) in validationPlan.stages" :key="index">
        {{ stageLine(stage, index) }}
      </span>
      <span v-for="warning in validationPlan.warnings" :key="warning" class="ms-timeline__warning">
        {{ warning }}
      </span>
    </section>
    <p
      v-if="validationError"
      class="ms-timeline__plan ms-timeline__plan--error"
      data-test="sequence-validation-error"
      role="alert"
    >
      {{ validationError }}
    </p>
  </div>
</template>

<style scoped>
/* The clip timeline (README §04) between the canvas and the composer, on the
   deep surface: transport, ruler, the scenes lane, and what the clip adds up
   to. The composer below owns the words and the Generate button. */
.ms-timeline {
  display: flex;
  flex-direction: column;
  gap: 8px;
  /*
   * The parent panel mounts this `flex: 1 1 0%`. Without an explicit
   * min-height the timeline is floored at its own min-content, so the panel
   * grew a scrollbar before the lane's shrink weight ever engaged. Zero lets
   * it take exactly its protected parent shell's space and flex the lane down
   * for real.
   *
   * Width floors at auto too: a scene title is nowrap with an ellipsis, and a
   * flex item's min-content contribution ignores its overflow, so the root
   * would be as wide as every prompt laid end to end and the bench would cut
   * the transport's and the footer's right edge off.
   */
  min-height: 0;
  min-width: 0;
  border-top: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-bg-deep);
  padding: 0 12px 11px;
  /* Room above the lane for the seam chips and the playhead's handle. */
  --scene-seam-space: 24px;
}

.ms-timeline__banner {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  gap: 10px;
  margin-top: 8px;
  border: var(--mold-bw) solid var(--mold-blue);
  background: var(--mold-accent-tint);
  border-radius: var(--mold-radius-2);
  padding: 6px 10px;
}
.ms-timeline__banner-text {
  flex: 1;
  min-width: 0;
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-2);
}

.ms-timeline__transport {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  gap: 10px;
  height: 36px;
  margin: 0 -12px;
  padding: 0 12px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
}
.ms-timeline__play {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 26px;
  height: 26px;
  border: 0;
  border-radius: var(--mold-radius-2);
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  cursor: pointer;
}
.ms-timeline__play:disabled {
  background: var(--mold-surface-2);
  color: var(--mold-text-faint);
  cursor: default;
}
.ms-timeline__rule {
  width: var(--mold-bw);
  height: 16px;
  background: var(--mold-border);
}
.ms-timeline__spacer {
  flex: 1;
  min-width: 12px;
}
.ms-timeline__help {
  display: flex;
  flex-shrink: 0;
  gap: 10px;
  margin: -8px -12px 0;
  padding: 10px 12px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
  background: var(--mold-surface);
}

/* The ruler's own left padding lines its zero up with the lane's first block,
   past the lane label — 74px of label plus the row's 8px gap. */
.ms-timeline__ruler {
  display: flex;
  flex-shrink: 0;
  align-items: flex-end;
  height: 18px;
  margin: -8px -12px 0;
  padding: 0 12px 0 94px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
}
.ms-timeline__ticks {
  position: relative;
  flex: 1;
  height: 100%;
}
.ms-timeline__tick {
  position: absolute;
  bottom: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
}
/* The closing mark grows INWARD from the right edge: pinned at left:100% its
   label starts at the edge and is cut off by the bench's overflow. */
.ms-timeline__tick--end {
  align-items: flex-end;
}
.ms-timeline__tick-label {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  line-height: 1;
  color: var(--mold-text-dim);
}
.ms-timeline__tick-mark {
  display: block;
  width: var(--mold-bw);
  height: 4px;
  background: var(--mold-surface-3);
}

/*
 * The lane absorbs a resize first: an outsized shrink weight pulls it from its
 * preferred basis down to a hard floor before any other row gives. The
 * preferred height MUST be the flex basis, not a `height` — a specified height
 * becomes the wrapper's min-content contribution, which propagates up as the
 * column's minimum and re-creates the scrollbar this exists to prevent.
 */
.ms-timeline__lanewrap {
  display: flex;
  width: 100%;
  flex: 0 999 96px;
  min-height: 62px;
  /* The seam chips ride above the lane, so the row reserves their height. */
  margin-top: var(--scene-seam-space);
}
.ms-timeline__lanewrap :deep(.ms-popover__trigger) {
  display: flex;
  align-items: stretch;
  gap: 8px;
  width: 100%;
  min-width: 0;
  height: 100%;
}
.ms-timeline__lane-label {
  display: flex;
  width: 74px;
  flex-shrink: 0;
  flex-direction: column;
  justify-content: center;
  gap: 2px;
  white-space: nowrap;
}
.ms-timeline__lane-area {
  position: relative;
  display: flex;
  flex: 1;
  min-width: 0;
}

.ms-timeline__playhead {
  position: absolute;
  top: calc(-1 * var(--scene-seam-space));
  bottom: 0;
  width: 2px;
  background: var(--mold-blue);
  pointer-events: none;
}
.ms-timeline__playhead-handle {
  position: absolute;
  top: 0;
  left: -4px;
  display: block;
  width: 10px;
  height: 7px;
  border-radius: var(--mold-radius-1);
  background: var(--mold-blue);
}

.ms-timeline__scene {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  gap: 10px;
}
.ms-timeline__frames {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.ms-timeline__select {
  height: var(--mold-ctl-md);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-bg);
  color: var(--mold-text-2);
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  padding: 0 6px;
}

.ms-timeline__foot {
  display: flex;
  align-items: center;
  flex-shrink: 0;
  gap: 8px;
  margin-top: auto;
}
.ms-timeline__readout {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-2);
}
.ms-timeline__file {
  display: none;
}
.ms-timeline__menu {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 150px;
}
.ms-timeline__menuitem {
  border: 0;
  background: transparent;
  color: var(--mold-text-2);
  text-align: left;
  padding: 6px 8px;
  border-radius: var(--mold-radius-1);
  font-size: var(--mold-fs-xs);
  cursor: pointer;
}
.ms-timeline__menuitem:hover {
  background: var(--mold-surface-2);
  color: var(--mold-text);
}

.ms-timeline__plan {
  display: grid;
  flex-shrink: 0;
  gap: 3px;
  max-height: 92px;
  overflow: auto;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
  padding: 7px 10px;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  color: var(--mold-text-2);
}
.ms-timeline__plan--error,
.ms-timeline__warning {
  color: var(--mold-warning);
}
.ms-timeline__plan--error {
  border-color: var(--mold-warning);
}
</style>
