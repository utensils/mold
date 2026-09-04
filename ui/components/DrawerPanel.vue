<script setup lang="ts">
/*
 * Right-side drawer — Advanced controls, model detail. Renders
 * INSIDE its owning frame: absolute overlay against the nearest positioned
 * ancestor, never Teleport or position:fixed. Backdrop click and Esc close;
 * clicks inside the panel do not.
 */
import { ref, useSlots } from "vue";
import { useRootFocusOnOpen } from "../lib/useRootFocusOnOpen";
import Icon from "./Icon.vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** Panel width in px. */
    width?: number;
    /** Header label when no header slot is given; also the dialog's name. */
    title?: string;
  }>(),
  { width: 452 },
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
    class="ms-drawer ms-fade-up"
    role="dialog"
    aria-modal="true"
    :aria-label="title"
    tabindex="-1"
    @click="emit('close')"
    @keydown.escape="emit('close')"
  >
    <div class="ms-drawer__panel" :style="{ width: `${width}px` }" @click.stop>
      <slot name="leading" />
      <div class="ms-drawer__header">
        <slot name="header">
          <span v-if="title" class="ms-group-label ms-drawer__title">{{
            title
          }}</span>
        </slot>
        <div class="ms-drawer__spacer" />
        <button
          type="button"
          class="ms-drawer__close"
          aria-label="Close"
          @click="emit('close')"
        >
          <Icon name="close" :size="15" />
        </button>
      </div>
      <div class="ms-drawer__body">
        <slot />
      </div>
      <div v-if="slots.footer" class="ms-drawer__footer">
        <slot name="footer" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.ms-drawer {
  position: absolute;
  inset: 0;
  z-index: 40;
  background: var(--mold-scrim);
  backdrop-filter: blur(5px);
  display: flex;
  justify-content: flex-end;
}

.ms-drawer__panel {
  position: relative;
  max-width: 100%;
  height: 100%;
  box-sizing: border-box;
  background: var(--mold-bg);
  border-left: var(--mold-bw) solid var(--mold-border);
  box-shadow: -24px 0 70px rgba(0, 0, 0, 0.5);
  display: flex;
  flex-direction: column;
}

.ms-drawer__header {
  height: 52px;
  flex: 0 0 52px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 20px;
}

.ms-drawer__title {
  text-transform: uppercase;
}

.ms-drawer__spacer {
  flex: 1;
}

.ms-drawer__close {
  width: 30px;
  height: 30px;
  flex: 0 0 30px;
  border-radius: var(--mold-radius-2);
  border: 0;
  background: color-mix(in srgb, var(--mold-text) 9%, transparent);
  color: var(--mold-text-2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition:
    background var(--mold-dur-quick) var(--mold-ease-out),
    color var(--mold-dur-quick) var(--mold-ease-out);
}

.ms-drawer__close:hover {
  background: color-mix(in srgb, var(--mold-text) 14%, transparent);
  color: var(--mold-text);
}

.ms-drawer__close:focus-visible {
  outline: 2px solid var(--mold-blue);
  outline-offset: 2px;
}

.ms-drawer__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 22px 22px 26px;
}

.ms-drawer__footer {
  border-top: var(--mold-bw) solid var(--mold-border);
  padding: 14px 20px;
}
</style>
