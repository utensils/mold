<script setup lang="ts">
import { computed, onBeforeUnmount } from "vue";
import { formatFrameDuration } from "../lib/duration";
import type { ClipRailMedia } from "./types";

const props = withDefaults(
  defineProps<{
    label: string;
    frames: number;
    fps?: number;
    active?: boolean;
    playing?: boolean;
    removable?: boolean;
    /** Render plan badge for edit sessions: cached tick or re-render mark. */
    plan?: "cached" | "rerender" | "new" | null;
    /** Durable stage media; never inferred from the mutable draft clip. */
    media?: ClipRailMedia | null;
    /** Valid model frame counts used by the timeline trim handle. */
    frameOptions?: readonly number[] | null;
  }>(),
  {
    fps: 24,
    active: false,
    playing: false,
    removable: false,
    plan: null,
    media: null,
    frameOptions: null,
  },
);

const emit = defineEmits<{
  select: [];
  play: [];
  remove: [];
  resize: [frames: number];
}>();

let resizeStartX = 0;
let resizeStartFrames = 0;

/** A stable timeline zoom: equal playback time gets equal visual width. */
const TIMELINE_PIXELS_PER_SECOND = 48;
const MIN_CLIP_WIDTH = 148;

const sortedFrameOptions = computed(() =>
  [...(props.frameOptions ?? [])].sort((a, b) => a - b),
);
const clipWidth = computed(() => {
  const fps = Math.max(1, props.fps);
  return Math.max(
    MIN_CLIP_WIDTH,
    Math.round((props.frames / fps) * TIMELINE_PIXELS_PER_SECOND),
  );
});

function stopResize() {
  document.removeEventListener("pointermove", onResizeMove);
  document.removeEventListener("pointerup", stopResize);
}

function onResizeMove(event: PointerEvent) {
  const options = sortedFrameOptions.value;
  if (!options.length) return;
  const pixelsPerFrame = TIMELINE_PIXELS_PER_SECOND / Math.max(1, props.fps);
  const target =
    resizeStartFrames + (event.clientX - resizeStartX) / pixelsPerFrame;
  const frames = options.reduce((nearest, option) =>
    Math.abs(option - target) < Math.abs(nearest - target) ? option : nearest,
  );
  if (frames !== props.frames) emit("resize", frames);
}

function onResizeKeydown(event: KeyboardEvent) {
  const options = sortedFrameOptions.value;
  if (options.length < 2) return;
  const currentIndex = options.reduce(
    (nearest, option, index) =>
      Math.abs(option - props.frames) <
      Math.abs(options[nearest]! - props.frames)
        ? index
        : nearest,
    0,
  );
  let nextIndex = currentIndex;
  if (event.key === "ArrowLeft") nextIndex = Math.max(0, currentIndex - 1);
  else if (event.key === "ArrowRight")
    nextIndex = Math.min(options.length - 1, currentIndex + 1);
  else if (event.key === "Home") nextIndex = 0;
  else if (event.key === "End") nextIndex = options.length - 1;
  else return;
  event.preventDefault();
  const frames = options[nextIndex];
  if (frames !== undefined && frames !== props.frames) emit("resize", frames);
}

function startResize(event: PointerEvent) {
  event.preventDefault();
  event.stopPropagation();
  resizeStartX = event.clientX;
  resizeStartFrames = props.frames;
  document.addEventListener("pointermove", onResizeMove);
  document.addEventListener("pointerup", stopResize);
}

onBeforeUnmount(stopResize);

function onRemove(event: MouseEvent) {
  event.stopPropagation();
  emit("remove");
}

function onPlay(event: MouseEvent) {
  event.stopPropagation();
  emit("play");
}

const durationLabel = computed(() =>
  formatFrameDuration(props.frames, props.fps),
);
const boundedProgress = computed(() =>
  Math.min(100, Math.max(0, props.media?.progressPercent ?? 0)),
);
const statusLabel = computed(() => {
  if (props.media?.status === "running") {
    return `Rendering ${Math.round(boundedProgress.value)}%`;
  }
  if (props.media?.status === "error") return "Render failed";
  if (props.media?.hasMedia && props.media.cacheReady) return "Cached";
  if (props.media?.hasMedia) return "Ready";
  if (props.plan === "rerender") return "Re-render";
  if (props.plan === "cached") return "Cached";
  if (props.plan === "new") return "New";
  if (props.media?.status === "pending") return "Waiting";
  return null;
});
</script>

<template>
  <span
    class="ms-clip"
    :data-on="active ? 'true' : undefined"
    :data-playing="playing ? 'true' : undefined"
    :data-status="media?.status ?? undefined"
    :data-plan="plan ?? undefined"
    :data-frames="frames"
    :style="{ '--clip-width': `${clipWidth}px` }"
  >
    <button
      type="button"
      class="ms-clip__body"
      :aria-label="`${label} · ${durationLabel}${statusLabel ? ` · ${statusLabel}` : ''}`"
      :aria-current="active ? 'true' : undefined"
      :title="
        media?.status === 'error'
          ? (media.error ?? 'This clip failed to render')
          : undefined
      "
      @click="emit('select')"
    >
      <span class="ms-clip__thumb">
        <img v-if="media?.posterUrl" :src="media.posterUrl" alt="" />
        <slot v-else name="thumb" />
        <span class="ms-clip__number" aria-hidden="true">{{
          label === "Opening clip" ? 1 : label.replace("Clip ", "")
        }}</span>
        <span
          v-if="media?.status === 'running'"
          class="ms-clip__rendering"
          aria-hidden="true"
        >
          <span class="ms-clip__spinner" />
          {{ Math.round(boundedProgress) }}%
        </span>
        <span
          v-else-if="media?.status === 'error'"
          class="ms-clip__error-mark"
          aria-hidden="true"
          >!</span
        >
        <span
          v-if="media?.status === 'running'"
          class="ms-clip__progress"
          aria-hidden="true"
        >
          <span :style="{ width: `${boundedProgress}%` }" />
        </span>
      </span>
      <span class="ms-clip__footer">
        <span class="ms-clip__label">{{ label }}</span>
        <span class="ms-clip__frames">{{ durationLabel }}</span>
      </span>
      <span
        v-if="statusLabel"
        class="ms-clip__status"
        :data-kind="media?.status ?? plan ?? undefined"
        aria-hidden="true"
      >
        <span v-if="statusLabel === 'Cached'">✓</span>
        <span v-else-if="statusLabel === 'Re-render'">↻</span>
        {{ statusLabel }}
      </span>
    </button>
    <button
      v-if="media?.hasMedia"
      type="button"
      class="ms-clip__play"
      :aria-label="`${playing ? 'Pause' : 'Play'} ${label}, ${durationLabel}`"
      :aria-pressed="playing"
      @click="onPlay"
    >
      <svg
        v-if="!playing"
        viewBox="0 0 24 24"
        width="16"
        height="16"
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M8 5.5v13l10-6.5z" />
      </svg>
      <svg
        v-else
        viewBox="0 0 24 24"
        width="16"
        height="16"
        fill="currentColor"
        aria-hidden="true"
      >
        <path d="M7 5h4v14H7zm6 0h4v14h-4z" />
      </svg>
    </button>
    <button
      v-if="removable"
      type="button"
      class="ms-clip__remove"
      :aria-label="`Remove ${label}`"
      @click="onRemove"
    >
      ✕
    </button>
    <button
      v-if="sortedFrameOptions.length > 1"
      type="button"
      class="ms-clip__resize"
      :aria-label="`Resize ${label}; currently ${durationLabel}`"
      title="Drag to change scene length"
      @pointerdown="startResize"
      @keydown="onResizeKeydown"
    >
      <span aria-hidden="true" />
    </button>
  </span>
</template>

<style scoped>
.ms-clip {
  position: relative;
  display: block;
  flex: 0 0 auto;
  width: var(--clip-width);
}

.ms-clip__resize {
  position: absolute;
  z-index: 5;
  top: 30px;
  right: -6px;
  bottom: 24px;
  width: 13px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: col-resize;
  touch-action: none;
}

.ms-clip__resize span {
  display: block;
  width: 3px;
  height: 34px;
  margin: auto;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.32);
  transition:
    height var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-clip__resize:hover span,
.ms-clip__resize:focus-visible span {
  height: 48px;
  background: var(--mold-blue);
}

.ms-clip__resize:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 1px;
}

.ms-clip__body {
  display: grid;
  width: 100%;
  height: var(--filmstrip-scene-height, 140px);
  padding: 0;
  overflow: hidden;
  text-align: left;
  background: color-mix(in srgb, var(--mold-media-bed) 92%, black);
  border: 1px solid color-mix(in srgb, var(--mold-text) 22%, transparent);
  border-radius: 9px;
  color: color-mix(in srgb, var(--mold-bg-deep) 90%, white);
  font-family: var(--mold-font-sans);
  font-size: 12px;
  cursor: pointer;
  transition:
    border-color var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out),
    transform var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-clip__body:hover {
  border-color: color-mix(in srgb, var(--mold-blue) 60%, transparent);
  transform: translateY(-1px);
}

.ms-clip[data-on="true"] .ms-clip__body {
  border-color: var(--mold-blue);
  color: white;
  box-shadow:
    0 0 0 1px var(--mold-blue),
    0 0 22px color-mix(in srgb, var(--mold-blue) 18%, transparent);
}

.ms-clip__body:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-clip__thumb {
  position: relative;
  display: block;
  width: 100%;
  height: var(--filmstrip-thumb-height, 104px);
  background: linear-gradient(
    145deg,
    color-mix(in srgb, var(--mold-sapphire) 14%, #101113),
    #050506
  );
  overflow: hidden;
}

.ms-clip__thumb :deep(img),
.ms-clip__thumb :deep(video) {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.ms-clip__thumb::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(180deg, transparent 48%, rgba(0, 0, 0, 0.48));
}

.ms-clip__number {
  position: absolute;
  z-index: 1;
  top: 8px;
  left: 8px;
  min-width: 23px;
  height: 23px;
  display: grid;
  place-items: center;
  padding: 0 5px;
  border: 1px solid rgba(255, 255, 255, 0.4);
  border-radius: 5px;
  background: rgba(4, 4, 5, 0.72);
  color: white;
  font-family: var(--mold-font-mono);
  font-size: 11px;
}

.ms-clip__rendering,
.ms-clip__error-mark {
  position: absolute;
  z-index: 2;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  background: rgba(5, 5, 6, 0.62);
  color: white;
  font-family: var(--mold-font-mono);
  font-size: 11px;
}

.ms-clip__error-mark {
  color: var(--mold-error);
  font-size: 20px;
  font-weight: 700;
}

.ms-clip__spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.26);
  border-top-color: var(--mold-blue);
  border-radius: 50%;
  animation: ms-clip-spin 0.8s linear infinite;
}

.ms-clip__progress {
  position: absolute;
  z-index: 3;
  right: 0;
  bottom: 0;
  left: 0;
  height: 3px;
  background: rgba(255, 255, 255, 0.15);
}

.ms-clip__progress > span {
  display: block;
  height: 100%;
  background: var(--mold-blue);
}

.ms-clip__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-height: var(--filmstrip-footer-height, 36px);
  padding: 8px 9px;
}

.ms-clip__label {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ms-clip__frames {
  font-family: var(--mold-font-mono);
  font-size: 9px;
  color: rgba(255, 255, 255, 0.6);
  white-space: nowrap;
}

.ms-clip[data-on="true"] .ms-clip__frames {
  color: color-mix(in srgb, var(--mold-blue) 72%, white);
}

.ms-clip__status {
  position: absolute;
  z-index: 4;
  right: 7px;
  top: 7px;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  max-width: 110px;
  padding: 3px 6px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(5, 5, 6, 0.76);
  color: rgba(255, 255, 255, 0.78);
  font-family: var(--mold-font-mono);
  font-size: 8px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ms-clip__status[data-kind="ready"],
.ms-clip__status[data-kind="cached"] {
  color: var(--mold-success);
}

.ms-clip__status[data-kind="error"] {
  color: var(--mold-error);
}

.ms-clip__status[data-kind="rerender"] {
  color: var(--mold-warning);
}

.ms-clip__play {
  position: absolute;
  z-index: 5;
  top: 44px;
  left: 50%;
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  padding: 0;
  border: 1px solid rgba(255, 255, 255, 0.68);
  border-radius: 50%;
  background: rgba(5, 5, 6, 0.75);
  color: white;
  cursor: pointer;
  opacity: 0;
  transform: translate(-50%, 3px);
  transition:
    opacity var(--mold-dur-quick) var(--mold-ease-out),
    transform var(--mold-dur-quick) var(--mold-ease-out),
    background var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-clip:hover .ms-clip__play,
.ms-clip:focus-within .ms-clip__play,
.ms-clip[data-playing="true"] .ms-clip__play {
  opacity: 1;
  transform: translate(-50%, 0);
}

.ms-clip__play:hover {
  background: var(--mold-blue);
  color: #17120a;
}

.ms-clip__play:focus-visible {
  opacity: 1;
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-clip__remove {
  position: absolute;
  z-index: 6;
  top: -7px;
  right: -7px;
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  padding: 0;
  border: 0;
  border-radius: 50%;
  border: 1px solid color-mix(in srgb, var(--mold-error) 55%, transparent);
  background: color-mix(in srgb, var(--mold-media-bed) 90%, black);
  color: rgba(255, 255, 255, 0.75);
  font-size: 10px;
  cursor: pointer;
  opacity: 0;
  pointer-events: none;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out),
    opacity var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-clip:hover .ms-clip__remove,
.ms-clip:focus-within .ms-clip__remove {
  opacity: 1;
  pointer-events: auto;
}

.ms-clip__remove:hover {
  background: var(--mold-error);
  color: white;
}

.ms-clip__remove:focus-visible {
  opacity: 1;
  pointer-events: auto;
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

@keyframes ms-clip-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (prefers-reduced-motion: reduce) {
  .ms-clip__spinner {
    animation: none;
  }
}

@media (hover: none) {
  .ms-clip__play {
    opacity: 1;
    transform: translate(-50%, 0);
  }
}
</style>
