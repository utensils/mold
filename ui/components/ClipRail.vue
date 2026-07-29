<script setup lang="ts" generic="C extends RailClip">
/*
 * Clip rail — the horizontal strip inside the composer bench (mockup 1c):
 * clip pills with seam pills between them, a dashed add-clip pill gated by
 * the stage cap, and drag-to-reorder. The seam pill between clip N−1 and N
 * edits clip N's incoming transition; clicking it emits `seam-click` so
 * the surface can anchor its popover (desktop/web) or sheet (iPhone).
 */
import { computed } from "vue";
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
    /** Motion-tail frames of the active model (0 → "Join clips" seams). */
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
  <div class="ms-rail" aria-label="Sequence filmstrip">
    <div class="ms-rail__perfs ms-rail__perfs--top" aria-hidden="true" />
    <VueDraggable
      v-model="dragModel"
      class="ms-rail__clips"
      role="list"
      aria-label="Sequence clips"
      :disabled="disabled"
      handle=".ms-clip__body"
      filter=".ms-clip__play,.ms-clip__remove,.ms-clip__resize,.ms-clip__resize *,.ms-seam,.ms-seam *"
    >
      <template v-for="(clip, index) in clips" :key="clip.id">
        <div class="ms-rail__item" role="listitem">
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
    <div class="ms-rail__perfs ms-rail__perfs--bottom" aria-hidden="true" />
  </div>
</template>

<style scoped>
.ms-rail {
  position: relative;
  display: flex;
  align-items: stretch;
  gap: 10px;
  min-height: 182px;
  padding: 23px 13px;
  overflow-x: auto;
  overflow-y: hidden;
  border: 1px solid color-mix(in srgb, var(--rebate) 16%, transparent);
  border-radius: 11px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.04), transparent 20%),
    color-mix(in srgb, var(--print) 94%, black);
  box-shadow:
    inset 0 1px rgba(255, 255, 255, 0.06),
    0 8px 24px color-mix(in srgb, var(--print) 18%, transparent);
  scrollbar-width: none;
}

.ms-rail::-webkit-scrollbar {
  display: none;
}

.ms-rail__clips {
  display: flex;
  align-items: stretch;
  gap: 10px;
  flex: 0 0 auto;
}

.ms-rail__item {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
}

.ms-rail__perfs {
  position: absolute;
  z-index: 8;
  right: 8px;
  left: 8px;
  height: 8px;
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
  top: 7px;
}

.ms-rail__perfs--bottom {
  bottom: 7px;
}

.ms-rail__add {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 9px;
  width: 78px;
  min-height: 138px;
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

.ms-rail :deep(.ms-seam) {
  width: 54px;
  height: auto;
  min-height: 76px;
  flex-direction: column;
  justify-content: center;
  gap: 7px;
  padding: 8px 4px;
  border-color: rgba(255, 255, 255, 0.2);
  background: rgba(255, 255, 255, 0.07);
  color: rgba(255, 255, 255, 0.72);
  white-space: normal;
}

.ms-rail :deep(.ms-seam__label) {
  max-width: 48px;
  overflow: hidden;
  text-align: center;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ms-rail :deep(.ms-seam__chevron) {
  display: none;
}

.ms-rail :deep(.ms-seam[data-on="true"]),
.ms-rail :deep(.ms-seam[data-transition="fade"]) {
  border-color: var(--safelight);
  color: var(--safelight);
  background: color-mix(in srgb, var(--safelight) 10%, transparent);
}

@media (max-width: 639px) {
  .ms-rail {
    min-height: 167px;
    padding-inline: 10px;
  }

  .ms-clip {
    width: 168px;
  }
}
</style>
