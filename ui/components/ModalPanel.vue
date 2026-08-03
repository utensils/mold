<script setup lang="ts">
/*
 * Centered task modal (spec §05) — e.g. connect-machine flow. Optional
 * stepped progress bars along the top. Renders INSIDE its owning frame:
 * absolute overlay, never Teleport or position:fixed. Backdrop click and
 * Esc close; clicks inside the panel do not.
 */
import { ref, useSlots } from "vue";
import { useRootFocusOnOpen } from "../lib/useRootFocusOnOpen";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** Panel width in px. */
    width?: number;
    /** Total number of progress bars; omit to hide the row. */
    steps?: number;
    /** Current 1-based step; this and earlier bars are tinted. */
    step?: number;
    /** Accessible name for the dialog. */
    label?: string;
  }>(),
  { width: 480, step: 1 },
);

const emit = defineEmits<{ close: [] }>();

const slots = useSlots();
const root = ref<HTMLElement | null>(null);
useRootFocusOnOpen(root, () => props.open);
</script>

<template>
  <div
    v-if="open"
    ref="root"
    class="ms-modal"
    role="dialog"
    aria-modal="true"
    :aria-label="label"
    tabindex="-1"
    @click="emit('close')"
    @keydown.escape="emit('close')"
  >
    <div class="ms-modal__panel" :style="{ width: `${width}px` }" @click.stop>
      <div v-if="steps" class="ms-modal__steps" aria-hidden="true">
        <span
          v-for="i in steps"
          :key="i"
          class="ms-modal__dot"
          :data-on="i <= step ? 'true' : undefined"
        />
      </div>
      <div class="ms-modal__body">
        <slot />
      </div>
      <div v-if="slots.footer" class="ms-modal__footer">
        <slot name="footer" />
      </div>
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

.ms-modal {
  position: absolute;
  inset: 0;
  background: rgba(6, 5, 10, 0.72);
  backdrop-filter: blur(5px);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
  animation: ms-fade-up var(--dur-base) var(--ease);
}

.ms-modal:focus-visible {
  outline: 2px solid var(--safelight);
  outline-offset: 2px;
}

.ms-modal__panel {
  max-width: 92%;
  box-sizing: border-box;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-card-lg);
  box-shadow: 0 40px 90px -20px rgba(0, 0, 0, 0.7);
  overflow: hidden;
}

.ms-modal__steps {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 18px 22px 0;
}

.ms-modal__dot {
  width: 22px;
  height: 3px;
  border-radius: 2px;
  background: var(--ce);
  transition: background var(--dur-base) var(--ease);
}

.ms-modal__dot[data-on="true"] {
  background: var(--safelight);
}

.ms-modal__body {
  padding: 16px 22px 22px;
}

.ms-modal__footer {
  border-top: 1px solid var(--edge);
  padding: 14px 22px;
  display: flex;
  align-items: center;
  gap: 10px;
}
</style>
