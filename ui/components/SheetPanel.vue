<script setup lang="ts">
/*
 * Phone-surface sheet (spec §05) — full-screen takeover ("full") or
 * bottom sheet with grab handle ("bottom"). Renders INSIDE its owning
 * frame: absolute overlay, never Teleport or position:fixed. Backdrop
 * click and Esc close; clicks inside the panel do not.
 */
import { ref } from "vue";
import { useRootFocusOnOpen } from "../lib/useRootFocusOnOpen";
import Icon from "./Icon.vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    variant?: "full" | "bottom";
    /** Header title; also the dialog's accessible name. */
    title?: string;
  }>(),
  { variant: "full" },
);

const emit = defineEmits<{ close: [] }>();

const root = ref<HTMLElement | null>(null);
useRootFocusOnOpen(root, () => props.open);
</script>

<template>
  <div
    v-if="open"
    ref="root"
    class="ms-sheet"
    :data-variant="variant"
    role="dialog"
    aria-modal="true"
    :aria-label="title"
    tabindex="-1"
    @click="emit('close')"
    @keydown.escape="emit('close')"
  >
    <div v-if="variant === 'full'" class="ms-sheet__panel-full" @click.stop>
      <div class="ms-sheet__header">
        <div class="ms-sheet__title">{{ title }}</div>
        <button
          type="button"
          class="ms-sheet__close"
          aria-label="Close"
          @click="emit('close')"
        >
          <Icon name="close" :size="15" />
        </button>
      </div>
      <div class="ms-sheet__body">
        <slot />
      </div>
    </div>
    <div v-else class="ms-sheet__panel-bottom" @click.stop>
      <div class="ms-sheet__handle" />
      <div v-if="title" class="ms-sheet__bottom-title">{{ title }}</div>
      <slot />
    </div>
  </div>
</template>

<style scoped>
@keyframes ms-fade-up {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

.ms-sheet {
  position: absolute;
  inset: 0;
  background: rgba(6, 5, 10, 0.72);
  backdrop-filter: blur(5px);
  display: flex;
  flex-direction: column;
  animation: ms-fade-up var(--dur-base) var(--ease);
}

.ms-sheet[data-variant="bottom"] {
  justify-content: flex-end;
}

.ms-sheet:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

/* ── Full takeover ─────────────────────────────────────────────────── */
.ms-sheet__panel-full {
  position: absolute;
  inset: 0;
  background: var(--bath);
  display: flex;
  flex-direction: column;
}

.ms-sheet__header {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 18px 12px;
  border-bottom: 1px solid var(--edge);
}

.ms-sheet__title {
  flex: 1;
  min-width: 0;
  font-family: var(--f-display);
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
}

.ms-sheet__close {
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  border-radius: var(--radius-pill);
  border: 0;
  background: color-mix(in srgb, var(--rebate) 9%, transparent);
  color: var(--ink-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition:
    background var(--dur-quick) var(--ease),
    color var(--dur-quick) var(--ease);
}

.ms-sheet__close:hover {
  background: color-mix(in srgb, var(--rebate) 14%, transparent);
  color: var(--rebate);
}

.ms-sheet__close:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-sheet__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 14px 18px 20px;
}

/* ── Bottom sheet ──────────────────────────────────────────────────── */
.ms-sheet__panel-bottom {
  background: var(--bench);
  border-top: 1px solid var(--edge);
  border-radius: 22px 22px 0 0;
  padding: 14px 16px 26px;
}

.ms-sheet__handle {
  width: 38px;
  height: 4px;
  border-radius: 3px;
  background: var(--ce);
  margin: 0 auto 14px;
}

.ms-sheet__bottom-title {
  font-family: var(--f-display);
  font-size: 16px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--rebate);
  margin-bottom: 10px;
}
</style>
