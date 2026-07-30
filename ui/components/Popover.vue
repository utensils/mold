<script setup lang="ts">
/*
 * Generic anchored popover — trigger + panel, outside-pointerdown and
 * Escape dismissal, focus return. The panel teleports to <body> and is
 * fixed-positioned from the trigger's viewport rect: rendered in place it
 * would extend a scrollable ancestor's overflow, so merely OPENING it could
 * grow a scrollbar (the Create bench did exactly that). Fixed boxes under
 * <body> contribute overflow to nothing and overlay everything. First
 * shared extraction of the pattern HostChip / File tools / Templates each
 * hand-rolled.
 */
import { nextTick, onBeforeUnmount, ref, watch } from "vue";

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
const panelEl = ref<HTMLElement | null>(null);
const panelStyle = ref<Record<string, string>>({});

const VIEWPORT_INSET = 8;
const GAP = 8;

function reposition() {
  const anchor = triggerEl.value;
  if (!anchor) return;
  const rect = anchor.getBoundingClientRect();
  const [side, align] = props.placement.split("-") as [
    "bottom" | "top",
    "start" | "end",
  ];
  panelStyle.value = {
    top: `${side === "bottom" ? rect.bottom + GAP : rect.top - GAP}px`,
    left: `${align === "start" ? rect.left : rect.right}px`,
    transform: `translate(${align === "end" ? "-100%" : "0"}, ${
      side === "top" ? "-100%" : "0"
    })`,
  };
  // Second pass once the panel has a size: slide it back inside the
  // viewport so a tall panel near an edge stays fully reachable.
  void nextTick(() => {
    const panel = panelEl.value;
    if (!panel) return;
    const box = panel.getBoundingClientRect();
    const overRight = box.right - (window.innerWidth - VIEWPORT_INSET);
    const overBottom = box.bottom - (window.innerHeight - VIEWPORT_INSET);
    const shiftX = Math.max(0, overRight) || Math.min(0, box.left - VIEWPORT_INSET);
    const shiftY = Math.max(0, overBottom) || Math.min(0, box.top - VIEWPORT_INSET);
    if (shiftX || shiftY) {
      panelStyle.value = {
        ...panelStyle.value,
        top: `${box.top - shiftY}px`,
        left: `${box.left - shiftX}px`,
        transform: "none",
      };
    }
  });
}

function close() {
  emit("update:open", false);
}

function onPointerDown(event: PointerEvent) {
  if (!props.open) return;
  if (!(event.target instanceof Node)) return;
  // The panel lives under <body>, not under root — check both.
  if (root.value?.contains(event.target)) return;
  if (panelEl.value?.contains(event.target)) return;
  close();
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
      // Capture-phase scroll keeps the fixed panel glued to its anchor
      // while any ancestor (bench, filmstrip) scrolls or resizes.
      window.addEventListener("scroll", reposition, true);
      window.addEventListener("resize", reposition);
      void nextTick(reposition);
    } else {
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("keydown", onKeydown, true);
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
    }
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", onPointerDown, true);
  document.removeEventListener("keydown", onKeydown, true);
  window.removeEventListener("scroll", reposition, true);
  window.removeEventListener("resize", reposition);
});
</script>

<template>
  <!-- The Teleport stays INSIDE the root div: a sibling root would make the
       template multi-root and silently drop consumers' class/style
       fallthrough (the bench's railwrap sizing rides on it). -->
  <div ref="root" class="ms-popover">
    <div ref="triggerEl" class="ms-popover__trigger">
      <slot name="trigger" />
    </div>
    <Teleport to="body">
      <div
        v-if="open"
        ref="panelEl"
        class="ms-popover__panel"
        :style="panelStyle"
        role="dialog"
        :aria-label="label"
      >
        <slot />
      </div>
    </Teleport>
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
  position: fixed;
  z-index: 30;
  min-width: 240px;
  padding: 8px;
  background: var(--bench);
  border: 1px solid var(--ce);
  border-radius: var(--radius-card);
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.45);
}
</style>
