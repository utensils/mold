<script setup lang="ts">
/*
 * More settings — the phone's advanced surface, now a real bottom sheet on the
 * shared `.mobile-sheet-*` chrome: a grabber, a scrim you can tap through to
 * dismiss, a bounded panel that leaves the composer visible behind it, and the
 * iOS header of a text control on each side with a centred title.
 *
 * It used to be a bare opaque overlay pinned to the top edge with a 44px
 * circular Done, which read as a screen that had replaced the composer rather
 * than a tray over it.
 *
 * The hosted controls stay mounted whether the sheet is open or not so their
 * validity wiring keeps reporting and the Generate flow can read every
 * advanced field; only visibility toggles.
 */
import { useMobileBack } from "./useMobileBack";
import { computed, ref, toRef, watch, nextTick, onBeforeUnmount } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";

const props = defineProps<{
  open: boolean;
  count: number;
}>();

const emit = defineEmits<{
  (event: "close"): void;
  (event: "reset"): void;
}>();
useMobileBack(toRef(props, "open"), () => emit("close"));
const panel = ref<HTMLElement | null>(null);
const body = ref<HTMLElement | null>(null);
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-more-settings");
let previousFocus: HTMLElement | null = null;
const dragOffset = ref(0);
const dragging = ref(false);
let dragTouchId: number | null = null;
let dragStartX = 0;
let dragStartY = 0;

const DISMISS_DISTANCE = 96;
const panelStyle = computed(() => ({
  transform: dragOffset.value > 0 ? `translateY(${dragOffset.value}px)` : undefined,
}));
const backdropStyle = computed(() => ({
  opacity: dragOffset.value > 0 ? Math.max(0.24, 1 - dragOffset.value / 320) : undefined,
}));

function resetDrag(): void {
  dragTouchId = null;
  dragOffset.value = 0;
  dragging.value = false;
}

watch(
  () => props.open,
  async (open) => {
    if (open) {
      previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      await nextTick();
      if (props.open) panel.value?.focus();
    } else {
      resetDrag();
      previousFocus?.focus();
      previousFocus = null;
    }
  },
  { immediate: true },
);

function beginDismiss(event: TouchEvent): void {
  if (
    event.touches.length !== 1 ||
    (body.value?.scrollTop ?? 0) > 0 ||
    (event.target instanceof Element &&
      Boolean(event.target.closest("input, textarea, select, button, a, [contenteditable='true']")))
  ) {
    resetDrag();
    return;
  }
  const touch = event.touches[0];
  if (!touch) return;
  dragTouchId = touch.identifier;
  dragStartX = touch.clientX;
  dragStartY = touch.clientY;
}

function moveDismiss(event: TouchEvent): void {
  if (dragTouchId === null || event.touches.length !== 1) return;
  const touch = [...event.touches].find((candidate) => candidate.identifier === dragTouchId);
  if (!touch) return;
  const deltaX = touch.clientX - dragStartX;
  const deltaY = touch.clientY - dragStartY;
  if (deltaY <= 0 || Math.abs(deltaX) >= deltaY) {
    dragOffset.value = 0;
    return;
  }
  dragging.value = true;
  dragOffset.value = Math.min(280, deltaY * 0.82);
  event.preventDefault();
}

function finishDismiss(): void {
  if (dragTouchId === null) return;
  const dismiss = dragOffset.value >= DISMISS_DISTANCE;
  resetDrag();
  if (dismiss) emit("close");
}

function onKeydown(event: KeyboardEvent): void {
  if (!props.open || !isTop()) return;
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopImmediatePropagation();
    emit("close");
  } else if (event.key === "Tab") {
    const controls = [
      ...(panel.value?.querySelectorAll<HTMLElement>(
        "button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex='0']",
      ) ?? []),
    ].filter((node) => !node.closest("[inert]") && node.getClientRects().length > 0);
    const first = controls[0];
    const last = controls.at(-1);
    if (
      !first ||
      (event.shiftKey &&
        (document.activeElement === first || document.activeElement === panel.value))
    ) {
      event.preventDefault();
      (last ?? panel.value)?.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
}
onBeforeUnmount(() => {
  resetDrag();
  previousFocus = null;
});
</script>

<template>
  <div
    class="mobile-sheet mobile-advanced-sheet"
    :inert="!open"
    aria-modal="true"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-label="More settings"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-advanced-sheet"
    @keydown="onKeydown"
  >
    <button
      class="mobile-sheet-scrim"
      type="button"
      data-sheet-close
      data-test="mobile-advanced-sheet-scrim"
      aria-label="Close more settings"
      :style="backdropStyle"
      @click="$emit('close')"
    />
    <div
      ref="panel"
      class="mobile-sheet-panel"
      :class="{ 'is-dragging': dragging }"
      :style="panelStyle"
      tabindex="-1"
      @touchstart="beginDismiss"
      @touchmove="moveDismiss"
      @touchend="finishDismiss"
      @touchcancel="resetDrag"
    >
      <span class="mobile-sheet-grabber" aria-hidden="true" />
      <header class="mobile-sheet-head mobile-advanced-sheet-head">
        <div class="mobile-sheet-head-slot">
          <button
            class="mobile-sheet-action mobile-advanced-sheet-reset"
            type="button"
            data-test="mobile-advanced-reset"
            @click="$emit('reset')"
          >
            Reset
          </button>
        </div>
        <div class="mobile-sheet-heading">
          <h2 class="mobile-sheet-title">More settings</h2>
        </div>
        <div class="mobile-sheet-head-slot">
          <span
            v-if="count > 0"
            class="mobile-advanced-sheet-badge"
            data-test="mobile-advanced-count"
          >
            {{ count }}
          </span>
          <button
            class="mobile-sheet-action is-strong mobile-advanced-sheet-close"
            type="button"
            data-sheet-close
            aria-label="Done with more settings"
            data-test="mobile-advanced-close"
            @click="$emit('close')"
          >
            Done
          </button>
        </div>
      </header>
      <div ref="body" class="mobile-advanced-sheet-body">
        <slot />
      </div>
    </div>
  </div>
</template>
