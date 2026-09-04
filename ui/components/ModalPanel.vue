<script setup lang="ts">
/*
 * Centered task dialog (README §04: 480–560px, header / body / footer,
 * radius-3, a 72% crust scrim) — e.g. connect a machine. Optional stepped
 * progress bars along the top. Renders INSIDE its owning frame: absolute
 * overlay, never Teleport or position:fixed. Backdrop click and Esc close;
 * clicks inside the panel do not.
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
    label?: string | undefined;
    /** `alertdialog` for a decision the user cannot get past. */
    role?: "dialog" | "alertdialog";
    /** Header title; with it the header block renders, bordered below. */
    title?: string | undefined;
    /** One plain sentence under the title; the `description` slot overrides it. */
    description?: string | undefined;
  }>(),
  { width: 480, step: 1, role: "dialog" },
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
    class="ms-modal ms-fade-up"
    :role="role"
    aria-modal="true"
    :aria-label="label ?? title"
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
      <div v-if="title" class="ms-modal__head">
        <span class="ms-modal__title">{{ title }}</span>
        <span v-if="slots.description || description" class="ms-modal__desc">
          <slot name="description">{{ description }}</slot>
        </span>
      </div>
      <div v-if="slots.default" class="ms-modal__body">
        <slot />
      </div>
      <div v-if="slots.footer" class="ms-modal__footer">
        <slot name="footer" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.ms-modal {
  position: absolute;
  inset: 0;
  background: var(--mold-scrim);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.ms-modal__panel {
  max-width: 92%;
  box-sizing: border-box;
  background: var(--mold-surface);
  border: var(--mold-bw) solid var(--mold-border);
  border-radius: var(--mold-radius-3);
  box-shadow: var(--mold-shadow-md);
  overflow: hidden;
}

.ms-modal__head {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 16px;
  border-bottom: var(--mold-bw) solid var(--mold-border);
}

.ms-modal__title {
  font-size: var(--mold-fs-md);
  font-weight: 600;
  color: var(--mold-text);
}

.ms-modal__desc {
  font-size: var(--mold-fs-xs);
  color: var(--mold-text-dim);
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
  border-radius: var(--mold-radius-1);
  background: var(--mold-border-control);
  transition: background var(--mold-dur-base) var(--mold-ease-out);
}

.ms-modal__dot[data-on="true"] {
  background: var(--mold-blue);
}

.ms-modal__body {
  padding: 16px;
}

.ms-modal__footer {
  border-top: var(--mold-bw) solid var(--mold-border);
  padding: 14px 16px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 8px;
}
</style>
