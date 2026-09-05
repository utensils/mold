<script setup lang="ts">
import { ref } from "vue";

/**
 * Thin vertical drag handle for resizable panels. Emits raw horizontal
 * deltas from the pointerdown origin — the parent owns the width math
 * (lib/panelResize.ts) and persistence.
 *
 * Contract: the CALLER must position the root (e.g. `absolute inset-y-0`
 * on a panel edge) — the widened `::after` hit strip anchors to that
 * positioned root; an unpositioned instance would anchor the strip to the
 * nearest positioned ancestor instead.
 */
// `label` (not `ariaLabel`): a prop named like the reserved aria attribute
// invites kebab/camel binding confusion at call sites.
defineProps<{ label: string }>();

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
  // Self-heal a lost capture: if every button is already up, the pointerup
  // never reached us — end the drag instead of emitting phantom resizes.
  if (e.buttons === 0) return onPointerEnd(e);
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

function onKeyDown(e: KeyboardEvent) {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  e.preventDefault();
  emit("resize", e.key === "ArrowLeft" ? -10 : 10);
  emit("commit");
}
</script>

<template>
  <div
    role="separator"
    tabindex="0"
    aria-orientation="vertical"
    :aria-label="label"
    class="w-1 shrink-0 touch-none cursor-col-resize transition-colors duration-100 before:absolute before:inset-y-0 before:left-1/2 before:w-px before:-translate-x-1/2 before:bg-border-control before:content-[''] after:absolute after:inset-y-0 after:-inset-x-1 after:content-[''] focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-accent"
    :class="dragging ? 'bg-accent' : 'bg-transparent hover:bg-accent'"
    @pointerdown="onPointerDown"
    @pointermove="onPointerMove"
    @pointerup="onPointerEnd"
    @pointercancel="onPointerEnd"
    @keydown="onKeyDown"
    @dblclick="emit('reset')"
  />
</template>
