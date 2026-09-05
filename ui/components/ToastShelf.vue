<script setup lang="ts">
/*
 * Toast shelf — top-right presenter inside the app frame.
 * The host owns the toast list and timers; this component renders it and
 * reports dismiss(id) / action(id). Errors and warnings are alerts with a
 * tinted border — a warning here is the sticky "your machine is gone", not an
 * aside; success/info are polite status rows. Optional action button (undo).
 */
import { computed } from "vue";
import {
  severityIsUrgent,
  severityMark,
  severityTint,
} from "../lib/notificationSeverity";
import type { Toast } from "./types";

export type { Toast };

const props = defineProps<{
  toasts: readonly Toast[];
}>();

// Hosts append, so the newest toast is last; it slides in at the top of the
// shelf and pushes the older ones down.
const ordered = computed(() => [...props.toasts].reverse());

const emit = defineEmits<{ dismiss: [id: string]; action: [id: string] }>();

/* Glyphs and severity names come from the one shared table so this shelf, the
 * desktop shelf, and the bell can never drift apart. */
const mark = severityMark;
const urgent = severityIsUrgent;

/* Border tint is derived from the same color rather than restated in CSS —
 * a per-kind stylesheet rule is how the tables drifted in the first place. */
function borderStyle(kind: Toast["kind"]) {
  return kind === "info" ? {} : { borderColor: severityTint(kind, 50) };
}
</script>

<template>
  <div class="ms-toasts">
    <TransitionGroup name="ms-toast">
      <div
        v-for="toast in ordered"
        :key="toast.id"
        class="ms-toast"
        :class="`ms-toast--${toast.kind}`"
        :style="borderStyle(toast.kind)"
        :role="urgent(toast.kind) ? 'alert' : 'status'"
      >
        <span
          class="ms-toast__glyph"
          :style="{ color: mark(toast.kind).color }"
          aria-hidden="true"
        >
          {{ mark(toast.kind).glyph }}
        </span>
        <span class="ms-toast__tone">{{ mark(toast.kind).label }}</span>
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
    opacity var(--mold-dur-slow) var(--mold-ease-out),
    transform var(--mold-dur-slow) var(--mold-ease-out);
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
  background: var(--mold-bg);
  border: 1px solid var(--mold-border);
  border-radius: var(--mold-radius-2);
  padding: 10px 14px;
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
}

.ms-toast__tone {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
  border: 0;
}

.ms-toast__glyph {
  flex: 0 0 auto;
  font-size: 12px;
  line-height: 1;
}

.ms-toast__text {
  flex: 1;
  min-width: 0;
  font-family: var(--mold-font-sans);
  font-size: 13px;
  color: var(--mold-text);
}

.ms-toast__action {
  flex: 0 0 auto;
  border: 0;
  background: transparent;
  color: var(--mold-blue);
  font-family: var(--mold-font-sans);
  font-size: 12.5px;
  font-weight: 700;
  padding: 0;
  cursor: pointer;
  transition: color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-toast__action:hover {
  color: var(--mold-text);
}

.ms-toast__dismiss {
  flex: 0 0 auto;
  border: 0;
  background: transparent;
  color: var(--mold-text-dim);
  font-size: 11px;
  line-height: 1;
  padding: 2px;
  cursor: pointer;
  transition: color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-toast__dismiss:hover {
  color: var(--mold-text);
}

.ms-toast__action:focus-visible,
.ms-toast__dismiss:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}
</style>
