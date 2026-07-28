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
import type { RailClip } from "./types";

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
    disabled?: boolean;
  }>(),
  { maxStages: 16, openSeamId: null, plans: null, disabled: false },
);

const emit = defineEmits<{
  select: [id: string];
  add: [];
  remove: [id: string];
  reorder: [ids: string[]];
  "seam-click": [id: string];
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
  <div class="ms-rail" role="list" aria-label="Sequence clips">
    <VueDraggable
      v-model="dragModel"
      class="ms-rail__clips"
      :disabled="disabled"
      handle=".ms-clip__body"
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
            :active="clip.id === activeId"
            :removable="removable"
            :plan="plans?.[index] ?? null"
            @select="emit('select', clip.id)"
            @remove="emit('remove', clip.id)"
          >
            <template #thumb
              ><slot name="thumb" :clip="clip" :index="index"
            /></template>
          </ClipPill>
        </div>
      </template>
    </VueDraggable>
    <span class="ms-rail__divider" aria-hidden="true" />
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
      clip
    </button>
    <span class="ms-rail__spacer" />
    <span v-if="!disabled && clips.length > 1" class="ms-rail__hint"
      >drag to reorder</span
    >
  </div>
</template>

<style scoped>
.ms-rail {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow-x: auto;
}

.ms-rail__clips {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ms-rail__item {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 0 auto;
}

.ms-rail__divider {
  width: 1px;
  height: 20px;
  background: var(--edge);
  flex: 0 0 1px;
  margin: 0 2px;
}

.ms-rail__add {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  flex: 0 0 auto;
  background: transparent;
  border: 1px dashed var(--ce);
  border-radius: var(--radius-pill);
  color: var(--ink-3);
  font-family: var(--f-body);
  font-size: 12px;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}

.ms-rail__add:hover {
  color: var(--rebate);
  border-color: var(--ink-3);
}

.ms-rail__add:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-rail__spacer {
  flex: 1;
}

.ms-rail__hint {
  font-family: var(--f-mono);
  font-size: 9.5px;
  color: var(--ink-3);
  flex: 0 0 auto;
}
</style>
