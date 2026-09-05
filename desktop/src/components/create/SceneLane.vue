<script setup lang="ts">
/*
 * The scenes lane (README §04): the whole clip laid out across the timeline's
 * width. Every block is `flex: <seconds> 1 0`, so a block is as wide as the
 * time it plays and the lane always fits — there is nothing to scroll. A
 * rendered scene's poster sits behind its block, the block's title is the
 * scene's own words, and the seam that joins it to the scene before floats
 * above the join. Only the selected block carries the trim grip, which snaps
 * to the model's own frame grid.
 */
import { computed, onBeforeUnmount, ref, type ComponentPublicInstance } from "vue";
import { VueDraggable } from "vue-draggable-plus";
import SeamGlyph from "@ui/components/SeamGlyph.vue";
import { transitionLabel } from "@ui/lib/seam";
import { formatFrameDuration } from "@ui/lib/duration";
import type { ClipRailMedia, RailClip } from "@ui/components/types";
import { sceneLabel } from "../../lib/sequenceTimeline";

const props = withDefaults(
  defineProps<{
    clips: RailClip[];
    activeId: string | null;
    /** Motion-tail frames of the active model; a smooth seam carries them. */
    motionTail: number;
    fps: number;
    /** Seam whose editor is open, for the pill's active ring. */
    openSeamId?: string | null;
    /** Per-scene render plan for edit sessions (index-aligned). */
    plans?: readonly ("cached" | "rerender" | "new")[] | null;
    mediaByClipId?: Readonly<Record<string, ClipRailMedia | undefined>> | null;
    playingId?: string | null;
    /** The model's own valid frame counts — the grid the grip snaps to. */
    frameOptions?: readonly number[] | null;
    disabled?: boolean;
  }>(),
  {
    openSeamId: null,
    plans: null,
    mediaByClipId: null,
    playingId: null,
    frameOptions: null,
    disabled: false,
  },
);

const emit = defineEmits<{
  select: [id: string];
  remove: [id: string];
  /** Delete on a focused block the two-scene floor refuses to remove. */
  "remove-blocked": [];
  reorder: [ids: string[]];
  "seam-click": [id: string];
  resize: [id: string, frames: number];
}>();

const laneRef = ref<ComponentPublicInstance | null>(null);
const removable = computed(() => !props.disabled && props.clips.length > 2);
const gridOptions = computed(() => [...(props.frameOptions ?? [])].sort((a, b) => a - b));

/** What a scene adds to the finished clip: a smooth or fading seam hands the
 *  frames it carries to the scene before, so they are counted once. */
function playedFrames(clip: RailClip, index: number): number {
  if (index === 0) return clip.frames;
  if (clip.transition === "smooth") return Math.max(0, clip.frames - props.motionTail);
  if (clip.transition === "fade") return Math.max(0, clip.frames - clip.fadeFrames);
  return clip.frames;
}

const totalFrames = computed(() =>
  props.clips.reduce((total, clip, index) => total + playedFrames(clip, index), 0),
);

function blockStyle(clip: RailClip, index: number) {
  const fps = props.fps > 0 ? props.fps : 1;
  return { flexGrow: `${playedFrames(clip, index) / fps}`, flexShrink: "1", flexBasis: "0px" };
}

const sceneTitle = (clip: RailClip, index: number) => sceneLabel(clip.prompt, index);

/** The block's caption is the time the block IS — the same number its width
 *  comes from. Labelled with `clip.frames` it claimed a length the block did
 *  not draw, because a smooth or fading seam hands its tail to the scene
 *  before. The authored length is still one hover away, and the Length picker
 *  above still edits it. */
function playedLabel(clip: RailClip, index: number): string {
  return formatFrameDuration(playedFrames(clip, index), props.fps);
}

function lengthTitle(clip: RailClip, index: number): string | undefined {
  const played = playedFrames(clip, index);
  if (played === clip.frames) return undefined;
  return `Plays ${formatFrameDuration(played, props.fps)} of ${formatFrameDuration(
    clip.frames,
    props.fps,
  )} — the seam hands ${clip.frames - played} frames to the scene before`;
}

const planWord: Record<"cached" | "rerender" | "new", string> = {
  cached: "kept",
  rerender: "re-made",
  new: "new",
};

/** The seam chip's own words, and what assistive tech hears. */
function seamText(clip: RailClip): string {
  return transitionLabel(clip.transition, props.motionTail);
}
function seamFrames(clip: RailClip): string | null {
  return clip.transition === "fade" ? `${clip.fadeFrames}f` : null;
}

// ── Trim grip ────────────────────────────────────────────────────────────────
// The lane is the ruler, so one pixel is worth the same number of frames
// everywhere on it. Both are read once at grab time: re-reading a lane that is
// re-laying itself out under the drag would chase its own tail.
let gripId: string | null = null;
let gripStartX = 0;
let gripStartFrames = 0;
let gripFramesPerPixel = 0;

function stopResize() {
  document.removeEventListener("pointermove", onResizeMove);
  document.removeEventListener("pointerup", stopResize);
  gripId = null;
}

function onResizeMove(event: MouseEvent) {
  const options = gridOptions.value;
  if (!gripId || options.length < 2) return;
  const target = gripStartFrames + (event.clientX - gripStartX) * gripFramesPerPixel;
  const frames = options.reduce((nearest, option) =>
    Math.abs(option - target) < Math.abs(nearest - target) ? option : nearest,
  );
  emit("resize", gripId, frames);
}

function startResize(event: PointerEvent, clip: RailClip) {
  const width = (laneRef.value?.$el as HTMLElement | undefined)?.getBoundingClientRect().width ?? 0;
  if (width <= 0 || totalFrames.value <= 0) return;
  event.preventDefault();
  event.stopPropagation();
  gripId = clip.id;
  gripStartX = event.clientX;
  gripStartFrames = clip.frames;
  gripFramesPerPixel = totalFrames.value / width;
  document.addEventListener("pointermove", onResizeMove);
  document.addEventListener("pointerup", stopResize);
}

function stepLength(clip: RailClip, direction: -1 | 1) {
  const options = gridOptions.value;
  if (options.length < 2) return;
  const at = options.reduce(
    (nearest, option, index) =>
      Math.abs(option - clip.frames) < Math.abs(options[nearest]! - clip.frames) ? index : nearest,
    0,
  );
  const frames = options[Math.min(options.length - 1, Math.max(0, at + direction))];
  if (frames !== undefined && frames !== clip.frames) emit("resize", clip.id, frames);
}

onBeforeUnmount(stopResize);

// ── Keyboard ─────────────────────────────────────────────────────────────────
function onBlockKeydown(event: KeyboardEvent, clip: RailClip, index: number) {
  if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
    if (event.shiftKey) {
      event.preventDefault();
      stepLength(clip, event.key === "ArrowLeft" ? -1 : 1);
      return;
    }
    const next = props.clips[index + (event.key === "ArrowLeft" ? -1 : 1)];
    if (!next) return;
    event.preventDefault();
    emit("select", next.id);
    // Focus moves with the selection: left on the old block, the next arrow
    // would count from the old index and Delete would remove the old scene.
    const block = (event.currentTarget as HTMLElement | null)?.parentElement?.querySelector(
      `[data-clip-id="${CSS.escape(next.id)}"]`,
    );
    if (block instanceof HTMLElement) block.focus();
  } else if (event.key === "Enter" || event.key === " ") {
    if (index === 0) return;
    event.preventDefault();
    emit("seam-click", clip.id);
  } else if (event.key === "Delete" || event.key === "Backspace") {
    // ALWAYS consumed on a focused block, floor or no floor: the webview's own
    // default for a bare Backspace is to go back, so leaving the key
    // unanswered navigated out of New image instead of refusing the removal.
    event.preventDefault();
    if (!removable.value) {
      emit("remove-blocked");
      return;
    }
    emit("remove", clip.id);
  }
}

// VueDraggable mutates its own list; the lane emits the order and lets the
// draft store apply it, so the lane never owns scene state.
const dragModel = computed({
  get: () => [...props.clips],
  set: (next: RailClip[]) =>
    emit(
      "reorder",
      next.map((clip) => clip.id),
    ),
});
</script>

<template>
  <VueDraggable
    ref="laneRef"
    v-model="dragModel"
    class="ms-lane"
    role="list"
    aria-label="Scenes lane"
    data-test="scene-lane"
    :disabled="disabled"
    handle=".ms-lane__body"
    filter=".ms-lane__grip,.ms-seam,.ms-seam *"
    :force-fallback="true"
    :fallback-tolerance="3"
  >
    <div
      v-for="(clip, index) in clips"
      :key="clip.id"
      class="ms-lane__scene"
      data-test="scene-block"
      role="listitem"
      tabindex="0"
      :data-clip-id="clip.id"
      :data-selected="clip.id === activeId ? 'true' : undefined"
      :data-playing="clip.id === playingId ? 'true' : undefined"
      :data-status="mediaByClipId?.[clip.id]?.status"
      :style="blockStyle(clip, index)"
      :aria-current="clip.id === activeId ? 'true' : undefined"
      @click="emit('select', clip.id)"
      @keydown="onBlockKeydown($event, clip, index)"
    >
      <!-- `ms-seam` is the timeline's right-click hook: a seam keeps its own
           editor instead of opening the scene menu. -->
      <button
        v-if="index > 0"
        type="button"
        class="ms-seam ms-lane__seam"
        data-test="scene-seam"
        :data-on="openSeamId === clip.id ? 'true' : undefined"
        :disabled="disabled"
        :aria-label="`How ${sceneTitle(clips[index - 1]!, index - 1)} meets ${sceneTitle(
          clip,
          index,
        )}: ${seamText(clip)}`"
        @click.stop="emit('seam-click', clip.id)"
        @contextmenu.prevent.stop="emit('seam-click', clip.id)"
        @keydown.stop
      >
        <SeamGlyph class="ms-lane__seam-glyph" :transition="clip.transition" />
        <span class="ms-lane__seam-label">{{ seamText(clip) }}</span>
        <span v-if="seamFrames(clip)" class="ms-lane__seam-frames">{{ seamFrames(clip) }}</span>
      </button>
      <div class="ms-lane__body">
        <span
          v-if="mediaByClipId?.[clip.id]?.posterUrl"
          class="ms-lane__poster"
          aria-hidden="true"
          :style="{ backgroundImage: `url(${mediaByClipId[clip.id]!.posterUrl})` }"
        />
        <span class="ms-lane__veil" aria-hidden="true" />
        <span class="ms-lane__title" data-test="scene-title">{{ sceneTitle(clip, index) }}</span>
        <span class="ms-lane__foot">
          <span class="ms-lane__length" :title="lengthTitle(clip, index)">{{
            playedLabel(clip, index)
          }}</span>
          <span v-if="plans?.[index]" class="ms-lane__plan">{{ planWord[plans[index]!] }}</span>
        </span>
        <span
          v-if="mediaByClipId?.[clip.id]?.progressPercent"
          class="ms-lane__progress"
          aria-hidden="true"
          :style="{ width: `${mediaByClipId[clip.id]!.progressPercent}%` }"
        />
      </div>
      <span
        v-if="clip.id === activeId && gridOptions.length > 1"
        class="ms-lane__grip"
        data-test="scene-grip"
        role="separator"
        aria-orientation="vertical"
        :aria-label="`Trim ${sceneTitle(clip, index)}; shift with the arrow keys`"
        title="Drag to change how long this scene runs"
        @pointerdown="startResize($event, clip)"
      >
        <span aria-hidden="true" />
      </span>
    </div>
  </VueDraggable>
</template>

<style scoped>
.ms-lane {
  display: flex;
  align-items: stretch;
  flex: 1;
  min-width: 0;
}

.ms-lane__scene {
  position: relative;
  display: flex;
  min-width: 0;
  cursor: pointer;
  outline: none;
}
.ms-lane__scene + .ms-lane__scene {
  margin-left: var(--mold-bw);
}

.ms-lane__body {
  position: relative;
  display: flex;
  flex: 1;
  min-width: 0;
  flex-direction: column;
  justify-content: space-between;
  gap: 4px;
  overflow: hidden;
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  background: var(--mold-surface);
  padding: 6px 8px;
  transition: border-color var(--mold-dur-quick) var(--mold-ease-out);
}
.ms-lane__scene:hover .ms-lane__body {
  border-color: var(--mold-border-focus);
}
.ms-lane__scene[data-selected="true"] .ms-lane__body,
.ms-lane__scene[data-playing="true"] .ms-lane__body {
  border-color: var(--mold-blue);
}
.ms-lane__scene:focus-visible .ms-lane__body {
  outline: 2px solid var(--mold-blue);
  outline-offset: 1px;
}
.ms-lane__scene[data-status="error"] .ms-lane__body {
  border-color: var(--mold-error);
}

.ms-lane__poster {
  position: absolute;
  inset: 0;
  opacity: 0.34;
  background-position: center;
  background-size: cover;
}
/* Keeps the scene's own words legible over its poster. */
.ms-lane__veil {
  position: absolute;
  inset: 0 0 auto;
  height: 60%;
  background: linear-gradient(to bottom, var(--mold-surface), transparent);
  pointer-events: none;
}

.ms-lane__title {
  position: relative;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--mold-fs-micro);
  font-weight: 600;
  color: var(--mold-text);
}
.ms-lane__foot {
  position: relative;
  display: flex;
  align-items: baseline;
  gap: 6px;
  min-width: 0;
  overflow: hidden;
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  white-space: nowrap;
}
.ms-lane__length {
  color: var(--mold-text-2);
}
.ms-lane__plan {
  color: var(--mold-text-dim);
}
.ms-lane__progress {
  position: absolute;
  left: 0;
  bottom: 0;
  height: 2px;
  background: var(--mold-blue);
}

/* The seam that joins this scene to the one before it: a chip riding the join
   itself, so the lane below stays a continuous strip of time. */
.ms-lane__seam {
  position: absolute;
  left: 0;
  top: -22px;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 4px;
  height: 19px;
  padding: 0 7px;
  transform: translateX(-50%);
  border: 0;
  border-radius: var(--mold-radius-2);
  background: var(--mold-blue);
  color: var(--mold-on-accent);
  line-height: 1;
  white-space: nowrap;
  cursor: pointer;
}
.ms-lane__seam[data-on="true"] {
  box-shadow: 0 0 0 2px var(--mold-bg-deep);
  filter: brightness(1.12);
}
.ms-lane__seam:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.ms-lane__seam:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
.ms-lane__seam-glyph {
  display: block;
  width: 10px;
  height: 10px;
  flex-shrink: 0;
  fill: currentColor;
}
.ms-lane__seam-label {
  font-size: var(--mold-fs-micro);
  font-weight: 600;
  line-height: 1;
}
.ms-lane__seam-frames {
  font-family: var(--mold-font-mono);
  font-size: var(--mold-fs-micro);
  line-height: 1;
  opacity: 0.75;
}

.ms-lane__grip {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  z-index: 1;
  display: flex;
  width: 9px;
  align-items: center;
  justify-content: center;
  border-radius: 0 var(--mold-radius-2) var(--mold-radius-2) 0;
  background: var(--mold-blue);
  cursor: ew-resize;
  touch-action: none;
}
.ms-lane__grip span {
  display: block;
  width: var(--mold-bw);
  height: 14px;
  background: var(--mold-bg-deep);
}
</style>
