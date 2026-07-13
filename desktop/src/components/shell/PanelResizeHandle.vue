<script setup lang="ts">
import { ref } from "vue";

/**
 * Thin vertical drag handle for resizable panels. Emits raw horizontal
 * deltas from the pointerdown origin — the parent owns the width math
 * (lib/panelResize.ts) and persistence. The parent also positions it
 * (typically `absolute inset-y-0` on a panel edge).
 */
defineProps<{ ariaLabel: string }>();

const emit = defineEmits<{
  resize: [dx: number];
  commit: [];
  reset: [];
}>();

const dragging = ref(false);
let startX = 0;

function onPointerDown(e: PointerEvent) {
  if (e.button !== 0) return;
  e.preventDefault();
  dragging.value = true;
  startX = e.clientX;
  const el = e.currentTarget as HTMLElement | null;
  try {
    el?.setPointerCapture?.(e.pointerId);
  } catch {
    // Synthetic events (tests) and exotic pointers may not support capture;
    // the drag still works while the pointer stays over the handle.
  }
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return;
  emit("resize", e.clientX - startX);
}

function onPointerEnd(e: PointerEvent) {
  if (!dragging.value) return;
  dragging.value = false;
  const el = e.currentTarget as HTMLElement | null;
  try {
    el?.releasePointerCapture?.(e.pointerId);
  } catch {
    // Capture may never have been taken; nothing to release.
  }
  emit("commit");
}
</script>

<template>
  <div
    role="separator"
    aria-orientation="vertical"
    :aria-label="ariaLabel"
    class="w-1 shrink-0 touch-none cursor-col-resize transition-colors duration-100 after:absolute after:inset-y-0 after:-inset-x-1 after:content-['']"
    :class="
      dragging
        ? 'bg-[color-mix(in_srgb,var(--safelight)_40%,transparent)]'
        : 'bg-transparent hover:bg-[color-mix(in_srgb,var(--safelight)_25%,transparent)]'
    "
    @pointerdown="onPointerDown"
    @pointermove="onPointerMove"
    @pointerup="onPointerEnd"
    @pointercancel="onPointerEnd"
    @dblclick="emit('reset')"
  />
</template>
