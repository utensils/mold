<script setup lang="ts">
/*
 * Seam sheet — the iPhone presentation of the shared @ui SeamEditor, and
 * mobile's FIRST fade-length control (the old in-card transition radiogroup
 * had no way to express `fade_frames` at all).
 *
 * Deliberately NOT @ui/SheetPanel: that primitive is `position: absolute;
 * inset: 0`, so on the scrolling mobile content column it renders off-screen,
 * and its `full` variant drops the #header slot — the "Join · opening →
 * clip 2" context line would silently vanish. This follows
 * MobileAdvancedSheet instead: a fixed overlay whose body owns its own
 * scroll and every safe-area inset, with the head row rendered in that body.
 */
import type { SequenceTransition } from "@studio/lib/sequence";
import SeamEditor from "@ui/components/SeamEditor.vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    transition: SequenceTransition;
    fadeFrames: number;
    /** Motion-tail frames of the active model; 0 relabels smooth "Join". */
    motionTail: number;
    fps: number;
    /** `chain-limits.fade_frames_max`; 32 when the server hasn't said. */
    fadeFramesMax?: number;
    fromLabel: string;
    toLabel: string;
  }>(),
  { fadeFramesMax: 32 },
);

const emit = defineEmits<{
  "update:transition": [value: SequenceTransition];
  "update:fadeFrames": [value: number];
  close: [];
}>();
</script>

<template>
  <div
    class="mobile-seam-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-label="Clip transition"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-seam-sheet"
  >
    <button
      class="mobile-seam-sheet-backdrop"
      type="button"
      aria-label="Close clip transition"
      data-test="mobile-seam-sheet-backdrop"
      @click="emit('close')"
    />
    <div class="mobile-seam-sheet-panel">
      <span class="mobile-seam-sheet-grabber" aria-hidden="true" />
      <div class="mobile-seam-sheet-body">
        <p class="mobile-seam-sheet-head" data-test="mobile-seam-sheet-head">
          Join · {{ fromLabel }} → {{ toLabel }}
        </p>
        <SeamEditor
          large
          :transition="transition"
          :fade-frames="fadeFrames"
          :motion-tail="motionTail"
          :fps="fps"
          :fade-frames-max="props.fadeFramesMax"
          :show-apply-all-hint="false"
          @update:transition="emit('update:transition', $event)"
          @update:fade-frames="emit('update:fadeFrames', $event)"
        />
        <button
          class="mobile-seam-sheet-done"
          type="button"
          data-test="mobile-seam-sheet-done"
          @click="emit('close')"
        >
          Done
        </button>
      </div>
    </div>
  </div>
</template>
