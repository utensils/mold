<script setup lang="ts">
/*
 * Hover/focus tooltip — the styled replacement for native `title` attributes
 * (slow to appear, unstyled, and paired with the `cursor: help` question
 * mark). The tip teleports to <body> and is fixed-positioned from the
 * trigger's viewport rect for the same reason Popover is: rendered in place
 * it would extend a scrollable ancestor's overflow. Hover waits a short
 * delay; keyboard focus shows immediately so the copy is reachable without
 * a pointer, and the trigger is aria-described by the tip while it is open.
 */
import { computed, nextTick, onBeforeUnmount, ref } from "vue";

const props = withDefaults(
  defineProps<{
    /** The tip copy. An empty string renders the trigger alone. */
    text: string;
    /** Which side of the trigger the tip prefers. */
    placement?: "top" | "bottom";
    /** Hover delay; focus always shows immediately. */
    delayMs?: number;
  }>(),
  { placement: "top", delayMs: 350 },
);

let nextId = 0;
const tipId = `ms-tooltip-${++nextId}-${Math.random().toString(36).slice(2, 8)}`;

const root = ref<HTMLElement | null>(null);
const tip = ref<HTMLElement | null>(null);
const open = ref(false);
const tipStyle = ref<Record<string, string>>({});
let showTimer: ReturnType<typeof setTimeout> | null = null;

const VIEWPORT_INSET = 8;
const GAP = 6;

const describedBy = computed(() =>
  open.value && props.text ? tipId : undefined,
);

function reposition() {
  const anchor = root.value;
  if (!anchor) return;
  const rect = anchor.getBoundingClientRect();
  tipStyle.value = {
    top: `${props.placement === "top" ? rect.top - GAP : rect.bottom + GAP}px`,
    left: `${rect.left + rect.width / 2}px`,
    transform: `translate(-50%, ${props.placement === "top" ? "-100%" : "0"})`,
  };
  // Second pass once the tip has a size: flip to the other side of the
  // trigger when the preferred side leaves the viewport vertically, then
  // slide horizontally so copy near an edge stays fully readable.
  void nextTick(() => {
    const box = tip.value?.getBoundingClientRect();
    const anchorBox = root.value?.getBoundingClientRect();
    if (!box || !anchorBox) return;
    let side = props.placement;
    if (side === "top" && box.top < VIEWPORT_INSET) side = "bottom";
    else if (
      side === "bottom" &&
      box.bottom > window.innerHeight - VIEWPORT_INSET
    ) {
      side = "top";
    }
    const top =
      side === "top"
        ? anchorBox.top - GAP - box.height
        : anchorBox.bottom + GAP;
    let left = anchorBox.left + anchorBox.width / 2 - box.width / 2;
    left = Math.min(left, window.innerWidth - VIEWPORT_INSET - box.width);
    left = Math.max(left, VIEWPORT_INSET);
    tipStyle.value = { top: `${top}px`, left: `${left}px`, transform: "none" };
  });
}

function show() {
  if (!props.text || open.value) return;
  open.value = true;
  reposition();
  window.addEventListener("scroll", hide, true);
}

function hide() {
  if (showTimer) {
    clearTimeout(showTimer);
    showTimer = null;
  }
  if (!open.value) return;
  open.value = false;
  window.removeEventListener("scroll", hide, true);
}

function onEnter() {
  if (showTimer || open.value) return;
  showTimer = setTimeout(() => {
    showTimer = null;
    show();
  }, props.delayMs);
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === "Escape" && open.value) {
    event.stopPropagation();
    hide();
  }
}

onBeforeUnmount(hide);
</script>

<template>
  <span
    ref="root"
    class="ms-tooltip"
    :aria-describedby="describedBy"
    @mouseenter="onEnter"
    @mouseleave="hide"
    @focusin="show"
    @focusout="hide"
    @keydown="onKeydown"
  >
    <slot />
    <Teleport to="body">
      <span
        v-if="open && text"
        :id="tipId"
        ref="tip"
        class="ms-tooltip__tip"
        role="tooltip"
        :style="tipStyle"
      >
        {{ text }}
      </span>
    </Teleport>
  </span>
</template>

<style scoped>
.ms-tooltip {
  display: inline-flex;
  min-width: 0;
}

.ms-tooltip__tip {
  position: fixed;
  z-index: 40;
  max-width: min(320px, calc(100vw - 16px));
  padding: 5px 9px;
  background: var(--mold-bg);
  border: 1px solid var(--mold-border-control);
  border-radius: var(--mold-radius-2, 8px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  color: var(--mold-text);
  font-size: 12px;
  line-height: 1.45;
  pointer-events: none;
}
</style>
