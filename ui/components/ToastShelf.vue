<script setup lang="ts">
/*
 * Toast shelf — top-right presenter inside the app frame (spec G11/G12).
 * The host owns the toast list and timers; this component renders it and
 * reports dismiss(id) / action(id). Errors are alerts with a stop-tinted
 * border; success/info are polite status rows. Optional action button (undo).
 */
import { computed } from "vue";
import type { Toast } from "./types";

export type { Toast };

const props = defineProps<{
  toasts: readonly Toast[];
}>();

// Hosts append, so the newest toast is last; it slides in at the top of the
// shelf and pushes the older ones down.
const ordered = computed(() => [...props.toasts].reverse());

const emit = defineEmits<{ dismiss: [id: string]; action: [id: string] }>();

const GLYPHS: Record<Toast["kind"], string> = {
  info: "•",
  success: "✓",
  error: "✕",
};
</script>

<template>
  <div class="ms-toasts">
    <TransitionGroup name="ms-toast">
      <div
        v-for="toast in ordered"
        :key="toast.id"
        class="ms-toast"
        :class="`ms-toast--${toast.kind}`"
        :role="toast.kind === 'error' ? 'alert' : 'status'"
      >
        <span class="ms-toast__glyph" aria-hidden="true">
          {{ GLYPHS[toast.kind] }}
        </span>
        <span class="ms-toast__text">{{ toast.text }}</span>
        <button
          v-if="toast.actionLabel"
          type="button"
          class="ms-toast__action"
          @click="emit('action', toast.id)"
        >
          {{ toast.actionLabel }}
        </button>
        <button
          type="button"
          class="ms-toast__dismiss"
          aria-label="Dismiss"
          @click="emit('dismiss', toast.id)"
        >
          ✕
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
/* Sits just clear of the app bar (52px compact, 56px wide) so a toast never
 * covers the nav controls. */
.ms-toasts {
  position: absolute;
  right: 16px;
  top: 64px;
  z-index: 130;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}

@media (min-width: 640px) {
  .ms-toasts {
    top: 68px;
  }
}

/* Local entrance — the shared ms-fade-up rises from below, which reads wrong
 * for a shelf that hangs from the top edge. */
.ms-toast-enter-from {
  opacity: 0;
  transform: translateY(-8px);
}

.ms-toast-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

.ms-toast-enter-active,
.ms-toast-leave-active,
.ms-toast-move {
  transition:
    opacity var(--dur-slow) var(--ease),
    transform var(--dur-slow) var(--ease);
}

/* Out of flow while leaving so the survivors slide up under the move class
 * instead of snapping once the fade ends. */
.ms-toast-leave-active {
  position: absolute;
  left: 0;
  right: 0;
}

@media (prefers-reduced-motion: reduce) {
  .ms-toast-enter-active,
  .ms-toast-leave-active,
  .ms-toast-move {
    transition: none;
  }
}

.ms-toast {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: 380px;
  background: var(--bench);
  border: 1px solid var(--edge);
  border-radius: var(--radius-card);
  padding: 10px 14px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
}

.ms-toast--error {
  border-color: color-mix(in srgb, var(--stop) 50%, transparent);
}

.ms-toast__glyph {
  flex: 0 0 auto;
  font-size: 12px;
  line-height: 1;
}

.ms-toast--success .ms-toast__glyph {
  color: var(--success);
}

.ms-toast--info .ms-toast__glyph {
  color: var(--halide);
}

.ms-toast--error .ms-toast__glyph {
  color: var(--stop);
}

.ms-toast__text {
  flex: 1;
  min-width: 0;
  font-family: var(--f-body);
  font-size: 13px;
  color: var(--rebate);
}

.ms-toast__action {
  flex: 0 0 auto;
  border: 0;
  background: transparent;
  color: var(--safelight);
  font-family: var(--f-body);
  font-size: 12.5px;
  font-weight: 700;
  padding: 0;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}

.ms-toast__action:hover {
  color: var(--rebate);
}

.ms-toast__dismiss {
  flex: 0 0 auto;
  border: 0;
  background: transparent;
  color: var(--ink-3);
  font-size: 11px;
  line-height: 1;
  padding: 2px;
  cursor: pointer;
  transition: color var(--dur-quick) var(--ease);
}

.ms-toast__dismiss:hover {
  color: var(--rebate);
}

.ms-toast__action:focus-visible,
.ms-toast__dismiss:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}
</style>
