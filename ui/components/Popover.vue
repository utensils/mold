<script setup lang="ts">
/*
 * Generic anchored popover — trigger + panel, outside-pointerdown and
 * Escape dismissal, focus return. Renders in place (relative wrapper,
 * absolute panel), NOT portaled, so unlike Drawer/Sheet/Modal it needs no
 * fixed-inset viewport host on scrolling pages. First shared extraction of
 * the pattern HostChip / File tools / Templates each hand-rolled.
 */
import { onBeforeUnmount, ref, watch } from "vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    /** Panel placement relative to the trigger. */
    placement?: "bottom-start" | "bottom-end" | "top-start" | "top-end";
    /** Accessible name for the panel dialog. */
    label?: string;
  }>(),
  { placement: "bottom-start" },
);

const emit = defineEmits<{ "update:open": [value: boolean] }>();

const root = ref<HTMLElement | null>(null);
const triggerEl = ref<HTMLElement | null>(null);

function close() {
  emit("update:open", false);
}

function onPointerDown(event: PointerEvent) {
  if (!props.open) return;
  if (
    root.value &&
    event.target instanceof Node &&
    !root.value.contains(event.target)
  ) {
    close();
  }
}

function onKeydown(event: KeyboardEvent) {
  if (!props.open) return;
  if (event.key === "Escape") {
    event.stopPropagation();
    close();
    // Focus returns to the trigger so keyboard flow doesn't dead-end.
    const trigger =
      triggerEl.value?.querySelector<HTMLElement>("button, [tabindex]");
    (trigger ?? triggerEl.value)?.focus?.();
  }
}

watch(
  () => props.open,
  (open) => {
    if (open) {
      document.addEventListener("pointerdown", onPointerDown, true);
      document.addEventListener("keydown", onKeydown, true);
    } else {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeydown, true);
    }
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onPointerDown, true);
  document.removeEventListener("keydown", onKeydown, true);
});
</script>

<template>
  <div ref="root" class="ms-popover">
    <div ref="triggerEl" class="ms-popover__trigger">
      <slot name="trigger" />
    </div>
    <div
      v-if="open"
      class="ms-popover__panel"
      :class="`ms-popover__panel--${placement}`"
      role="dialog"
      :aria-label="label"
    >
      <slot />
    </div>
  </div>
</template>

<style scoped>
.ms-popover {
  position: relative;
  display: inline-flex;
}

.ms-popover__trigger {
  display: inline-flex;
}

.ms-popover__panel {
  position: absolute;
  z-index: 30;
  min-width: 240px;
  padding: 8px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-card);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
}

.ms-popover__panel--bottom-start {
  top: calc(100% + 8px);
  left: 0;
}

.ms-popover__panel--bottom-end {
  top: calc(100% + 8px);
  right: 0;
}

.ms-popover__panel--top-start {
  bottom: calc(100% + 8px);
  left: 0;
}

.ms-popover__panel--top-end {
  bottom: calc(100% + 8px);
  right: 0;
}
</style>
