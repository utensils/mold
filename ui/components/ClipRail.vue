<script setup lang="ts" generic="C extends RailClip">
/*
 * Clip rail — the horizontal strip inside the composer bench (mockup 1c):
 * clip pills with seam pills between them, a dashed add-clip pill gated by
 * the stage cap, and drag-to-reorder. The seam pill between clip N−1 and N
 * edits clip N's incoming transition; clicking it emits `seam-click` so
 * the surface can anchor its popover (desktop/web) or sheet (iPhone).
 */
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
} from "vue";
import { VueDraggable } from "vue-draggable-plus";
import ClipPill from "./ClipPill.vue";
import SeamPill from "./SeamPill.vue";
import type { ClipRailMedia, RailClip } from "./types";

// Re-exported for convenience; the canonical home is ./types (see its
// header comment — SFC named exports break clean-sandbox builds).
export type { RailClip };

const props = withDefaults(
  defineProps<{
    clips: C[];
    activeId: string | null;
    /** Motion-tail frames of the active model (0 → "Join" seams). */
    motionTail: number;
    maxStages?: number;
    /** Seam whose editor is currently open (clip id), for the active ring. */
    openSeamId?: string | null;
    /** Per-clip render plan for edit sessions (index-aligned). */
    plans?: readonly ("cached" | "rerender" | "new")[] | null;
    /** Live model FPS, used for honest frame + seconds labels. */
    fps?: number;
    /** Durable stage state keyed by draft clip id. */
    mediaByClipId?: Readonly<Record<string, ClipRailMedia | undefined>> | null;
    /** Clip currently playing in the surface's main canvas. */
    playingId?: string | null;
    /** Valid frame counts for drag-to-trim scene sizing. */
    frameOptions?: readonly number[] | null;
    disabled?: boolean;
  }>(),
  {
    maxStages: 16,
    openSeamId: null,
    plans: null,
    fps: 24,
    mediaByClipId: null,
    playingId: null,
    frameOptions: null,
    disabled: false,
  },
);

const emit = defineEmits<{
  select: [id: string];
  add: [];
  remove: [id: string];
  reorder: [ids: string[]];
  "seam-click": [id: string];
  play: [id: string];
  resize: [id: string, frames: number];
}>();

const canAdd = computed(
  () => !props.disabled && props.clips.length < props.maxStages,
);
const removable = computed(() => !props.disabled && props.clips.length > 2);
const scroller = ref<HTMLElement | null>(null);
const canScrollBackward = ref(false);
const canScrollForward = ref(false);
let resizeObserver: ResizeObserver | null = null;

function updateScrollAffordances(): void {
  const element = scroller.value;
  if (!element) return;
  const max = Math.max(0, element.scrollWidth - element.clientWidth);
  canScrollBackward.value = element.scrollLeft > 2;
  canScrollForward.value = element.scrollLeft < max - 2;
}

function scrollTo(left: number, behavior: ScrollBehavior = "smooth"): void {
  const element = scroller.value;
  if (!element) return;
  element.scrollTo({ left: Math.max(0, left), behavior });
}

function scrollPage(direction: -1 | 1): void {
  const element = scroller.value;
  if (!element) return;
  const distance = Math.max(180, element.clientWidth * 0.72);
  scrollTo(element.scrollLeft + distance * direction);
}

/**
 * Keep the active scene fully visible. `scrollIntoView()` can also move the
 * page/panel vertically, so constrain this to the filmstrip's x axis.
 */
function revealActiveClip(behavior: ScrollBehavior = "smooth"): void {
  const element = scroller.value;
  if (!element || !props.activeId) return;
  const item = [
    ...element.querySelectorAll<HTMLElement>("[data-clip-id]"),
  ].find((candidate) => candidate.dataset.clipId === props.activeId);
  if (!item) return;
  const viewport = element.getBoundingClientRect();
  const bounds = item.getBoundingClientRect();
  const edgeInset = 46;
  if (bounds.left < viewport.left + edgeInset) {
    scrollTo(
      element.scrollLeft + bounds.left - viewport.left - edgeInset,
      behavior,
    );
  } else if (bounds.right > viewport.right - edgeInset) {
    scrollTo(
      element.scrollLeft + bounds.right - viewport.right + edgeInset,
      behavior,
    );
  }
}

watch(
  () =>
    [
      props.activeId,
      props.clips.map((clip) => `${clip.id}:${clip.frames}`).join(":"),
    ] as const,
  () =>
    void nextTick(() => {
      updateScrollAffordances();
      revealActiveClip();
    }),
);

onMounted(() => {
  void nextTick(() => {
    updateScrollAffordances();
    revealActiveClip("auto");
  });
  if (typeof ResizeObserver !== "undefined" && scroller.value) {
    resizeObserver = new ResizeObserver(() => {
      updateScrollAffordances();
      revealActiveClip("auto");
    });
    resizeObserver.observe(scroller.value);
  }
  window.addEventListener("resize", updateScrollAffordances);
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
  window.removeEventListener("resize", updateScrollAffordances);
});

function clipLabel(index: number): string {
  return index === 0 ? "Opening clip" : `Clip ${index + 1}`;
}

// VueDraggable mutates a proxy list; we emit the id order and let the
// store apply it so the rail never owns clip state.
const dragModel = computed({
  get: () => [...props.clips],
  set: (next: C[]) =>
    emit(
      "reorder",
      next.map((clip) => clip.id),
    ),
});
</script>

<template>
  <div
    class="ms-rail-frame"
    aria-label="Sequence filmstrip"
    :data-scroll-start="canScrollBackward ? 'true' : undefined"
    :data-scroll-end="canScrollForward ? 'true' : undefined"
  >
    <div class="ms-rail__perfs ms-rail__perfs--top" aria-hidden="true" />
    <div
      ref="scroller"
      class="ms-rail"
      tabindex="0"
      aria-label="Scrollable sequence clips"
      @scroll="updateScrollAffordances"
    >
      <VueDraggable
        v-model="dragModel"
        class="ms-rail__clips"
        role="list"
        aria-label="Sequence clips"
        :disabled="disabled"
        handle=".ms-clip__body"
        filter=".ms-clip__play,.ms-clip__remove,.ms-clip__resize,.ms-clip__resize *,.ms-seam,.ms-seam *"
        :force-fallback="true"
        :fallback-tolerance="3"
      >
        <template v-for="(clip, index) in clips" :key="clip.id">
          <div class="ms-rail__item" role="listitem" :data-clip-id="clip.id">
            <SeamPill
              v-if="index > 0"
              :transition="clip.transition"
              :motion-tail="motionTail"
              :fade-frames="clip.fadeFrames"
              :active="openSeamId === clip.id"
              :disabled="disabled"
              @click="emit('seam-click', clip.id)"
            />
            <ClipPill
              :label="clipLabel(index)"
              :frames="clip.frames"
              :fps="fps"
              :active="clip.id === activeId"
              :playing="clip.id === playingId"
              :removable="removable"
              :plan="plans?.[index] ?? null"
              :media="mediaByClipId?.[clip.id] ?? null"
              :frame-options="frameOptions"
              @select="emit('select', clip.id)"
              @play="emit('play', clip.id)"
              @remove="emit('remove', clip.id)"
              @resize="emit('resize', clip.id, $event)"
            >
              <template #thumb
                ><slot name="thumb" :clip="clip" :index="index"
              /></template>
            </ClipPill>
          </div>
        </template>
      </VueDraggable>
      <button
        v-if="canAdd"
        type="button"
        class="ms-rail__add"
        aria-label="Add clip"
        @click="emit('add')"
      >
        <svg
          viewBox="0 0 24 24"
          width="13"
          height="13"
          fill="none"
          stroke="currentColor"
          stroke-width="1.7"
          stroke-linecap="round"
          aria-hidden="true"
        >
          <path d="M12 5v14M5 12h14" />
        </svg>
        <span>Add clip</span>
      </button>
      <span class="ms-rail__end-space" aria-hidden="true" />
    </div>
    <span
      v-show="canScrollBackward"
      class="ms-rail__fade ms-rail__fade--start"
      aria-hidden="true"
    />
    <span
      v-show="canScrollForward"
      class="ms-rail__fade ms-rail__fade--end"
      aria-hidden="true"
    />
    <button
      v-show="canScrollBackward"
      type="button"
      class="ms-rail__scroll ms-rail__scroll--start"
      aria-label="Scroll filmstrip left"
      @click="scrollPage(-1)"
    >
      ‹
    </button>
    <button
      v-show="canScrollForward"
      type="button"
      class="ms-rail__scroll ms-rail__scroll--end"
      aria-label="Scroll filmstrip right"
      @click="scrollPage(1)"
    >
      ›
    </button>
    <div class="ms-rail__perfs ms-rail__perfs--bottom" aria-hidden="true" />
  </div>
</template>

<style scoped>
.ms-rail-frame {
  /*
   * Fluid filmstrip geometry: the frame is its own size container, so every
   * internal vertical measure (paddings, scene, thumb) derives from the
   * frame's actual height. Surfaces only ever set the frame height — the
   * desktop bench resizer shrinks it via flex — and the strip re-proportions
   * continuously instead of stepping through fixed density tiers or growing
   * scrollbars.
   */
  container-type: size;
  --filmstrip-pad-top: clamp(12px, 10cqh, 19px);
  --filmstrip-pad-bottom: clamp(14px, 12cqh, 23px);
  --filmstrip-footer-height: clamp(26px, 20cqh, 36px);
  --filmstrip-scene-height: calc(
    100cqh - var(--filmstrip-pad-top) - var(--filmstrip-pad-bottom)
  );
  --filmstrip-thumb-height: calc(
    var(--filmstrip-scene-height) - var(--filmstrip-footer-height)
  );
  --filmstrip-seam-width: 46px;
  position: relative;
  height: 188px;
  min-width: 0;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--rebate) 16%, transparent);
  border-radius: 11px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.04), transparent 20%),
    color-mix(in srgb, var(--print) 94%, black);
  box-shadow:
    inset 0 1px rgba(255, 255, 255, 0.06),
    0 8px 24px color-mix(in srgb, var(--print) 18%, transparent);
}

.ms-rail {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: stretch;
  gap: 10px;
  min-width: 0;
  padding: var(--filmstrip-pad-top) 0 var(--filmstrip-pad-bottom) 13px;
  overflow-x: auto;
  overflow-y: hidden;
  scroll-padding-inline: 48px;
  scroll-snap-type: x proximity;
  scrollbar-width: thin;
  scrollbar-color: color-mix(in srgb, var(--ink-3) 55%, transparent) transparent;
}

.ms-rail::-webkit-scrollbar {
  height: 6px;
}

.ms-rail::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: color-mix(in srgb, var(--ink-3) 55%, transparent);
}

.ms-rail__clips {
  display: flex;
  align-items: stretch;
  gap: 10px;
  flex: 0 0 auto;
}

.ms-rail__item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  flex: 0 0 auto;
  scroll-snap-align: start;
  scroll-snap-stop: normal;
}

.ms-rail__perfs {
  position: absolute;
  z-index: 8;
  right: 8px;
  left: 8px;
  height: clamp(5px, 4cqh, 8px);
  pointer-events: none;
  background: repeating-linear-gradient(
    90deg,
    rgba(255, 255, 255, 0.15) 0 9px,
    transparent 9px 19px
  );
  mask: linear-gradient(#000 0 0);
  border-radius: 2px;
}

.ms-rail__perfs--top {
  top: clamp(4px, 3.5cqh, 7px);
}

.ms-rail__perfs--bottom {
  bottom: clamp(4px, 3.5cqh, 7px);
}

.ms-rail__add {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 9px;
  width: 78px;
  min-height: var(--filmstrip-scene-height);
  padding: 10px;
  flex: 0 0 78px;
  background: rgba(255, 255, 255, 0.035);
  border: 1px dashed rgba(255, 255, 255, 0.22);
  border-radius: 9px;
  color: rgba(255, 255, 255, 0.62);
  font-family: var(--f-body);
  font-size: 12px;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
  scroll-snap-align: end;
}

.ms-rail__add:hover {
  color: white;
  border-color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 8%, transparent);
}

.ms-rail__add:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-rail__end-space {
  width: 54px;
  flex: 0 0 54px;
}

.ms-rail__fade {
  position: absolute;
  z-index: 9;
  top: 14px;
  bottom: 14px;
  width: 58px;
  pointer-events: none;
}

.ms-rail__fade--start {
  left: 0;
  background: linear-gradient(90deg, var(--print), transparent);
}

.ms-rail__fade--end {
  right: 0;
  background: linear-gradient(270deg, var(--print), transparent);
}

.ms-rail__scroll {
  position: absolute;
  z-index: 10;
  top: 50%;
  width: 34px;
  height: 44px;
  display: grid;
  place-items: center;
  padding: 0;
  border: 1px solid color-mix(in srgb, white 22%, transparent);
  border-radius: var(--radius-pill);
  background: color-mix(in srgb, var(--print) 88%, transparent);
  color: white;
  font-family: var(--f-body);
  font-size: 25px;
  line-height: 1;
  cursor: pointer;
  transform: translateY(-50%);
  backdrop-filter: blur(8px);
}

.ms-rail__scroll--start {
  left: 8px;
}

.ms-rail__scroll--end {
  right: 8px;
}

.ms-rail__scroll:hover {
  border-color: var(--safelight);
  color: var(--safelight);
}

.ms-rail__scroll:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/*
 * Seams inside the filmstrip pick up the strip's light-on-dark ink and are
 * centered on the picture, not the combined picture + metadata card, so every
 * seam sits on one visual edit axis at any rail height.
 */
.ms-rail-frame :deep(.ms-seam) {
  width: var(--filmstrip-seam-width);
  min-width: 44px;
  margin-top: calc(var(--filmstrip-thumb-height) / 2 - 16px);
  color: rgba(255, 255, 255, 0.72);
}

.ms-rail-frame :deep(.ms-seam:hover:not(:disabled)) {
  color: white;
}

.ms-rail-frame :deep(.ms-seam__diagram) {
  border-color: color-mix(in srgb, white 34%, transparent);
  background: color-mix(in srgb, var(--print) 88%, white 12%);
  box-shadow: 0 2px 8px color-mix(in srgb, black 32%, transparent);
}

.ms-rail-frame :deep(.ms-seam:hover:not(:disabled) .ms-seam__diagram) {
  border-color: rgba(255, 255, 255, 0.7);
}

.ms-rail-frame :deep(.ms-seam[data-on="true"]),
.ms-rail-frame :deep(.ms-seam[data-transition="fade"]) {
  color: var(--safelight);
}

.ms-rail-frame :deep(.ms-seam[data-on="true"] .ms-seam__diagram),
.ms-rail-frame :deep(.ms-seam[data-transition="fade"] .ms-seam__diagram) {
  border-color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 14%, var(--print));
}

.ms-rail-frame :deep(.ms-seam__caption) {
  max-width: 100%;
  font-size: 8.5px;
}

@media (max-width: 639px) {
  .ms-rail-frame {
    --filmstrip-seam-width: 44px;
    height: 160px;
  }

  .ms-rail {
    padding-left: 10px;
  }
}
</style>
