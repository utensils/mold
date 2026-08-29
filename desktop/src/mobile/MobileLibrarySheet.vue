<script setup lang="ts">
/*
 * Library bottom sheet — the iPhone home for the Library's editors (tag
 * editor, collection checklist, new/rename collection, the "More tags" list,
 * and the viewer's print info). Follows MobileSeamSheet / MobileAdvancedSheet
 * rather than @ui/SheetPanel: a fixed overlay whose body owns its scroll and
 * every safe-area inset, with the head row rendered in the body so it can
 * never vanish the way SheetPanel's `full` variant drops its #header slot.
 */
import { computed, onBeforeUnmount, ref, watch } from "vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title: string;
    /** Focus the first editor when the sheet opens. Disable for read-first sheets. */
    focusFirstControl?: boolean;
    /** Trailing label of the one closing control. */
    doneLabel?: string;
    testId?: string;
    /** Platform-specific minimum target for the closing control. */
    touchTargetSize?: number;
    /** Let a downward drag from the top dismiss this read-first sheet. */
    swipeToDismiss?: boolean;
  }>(),
  {
    focusFirstControl: true,
    doneLabel: "Done",
    testId: "mobile-library-sheet",
    touchTargetSize: 46,
    swipeToDismiss: false,
  },
);

const emit = defineEmits<{ close: [] }>();

const panel = ref<HTMLElement | null>(null);
const body = ref<HTMLElement | null>(null);
const dragOffset = ref(0);
const dragging = ref(false);
let restoreFocus: HTMLElement | null = null;
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
  (open) => {
    if (open) {
      restoreFocus = document.activeElement as HTMLElement | null;
      // Editing sheets may raise the keyboard immediately. Read-first sheets
      // focus the panel so the keyboard waits for an explicit field tap.
      queueMicrotask(() => {
        if (!props.focusFirstControl) {
          panel.value?.focus?.();
          return;
        }
        const first = panel.value?.querySelector<HTMLElement>(
          "input, textarea, select, button:not([data-sheet-close])",
        );
        (first ?? panel.value)?.focus?.();
      });
    } else {
      resetDrag();
      restoreFocus?.focus?.();
      restoreFocus = null;
    }
  },
);

onBeforeUnmount(() => {
  resetDrag();
  restoreFocus = null;
});

function beginDismiss(event: TouchEvent): void {
  if (
    !props.swipeToDismiss ||
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
  if (event.key === "Escape") {
    event.preventDefault();
    emit("close");
  }
}
</script>

<template>
  <div
    class="mobile-library-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-modal="true"
    :aria-label="title"
    :aria-hidden="open ? undefined : 'true'"
    :data-test="testId"
    :style="{ '--mobile-sheet-touch-target': `${touchTargetSize}px` }"
    @keydown="onKeydown"
  >
    <button
      class="mobile-library-sheet-backdrop"
      type="button"
      data-sheet-close
      :aria-label="`Close ${title}`"
      :data-test="`${testId}-backdrop`"
      :style="backdropStyle"
      @click="emit('close')"
    />
    <div
      ref="panel"
      class="mobile-library-sheet-panel"
      :class="{ 'is-dragging': dragging }"
      :style="panelStyle"
      tabindex="-1"
      @touchstart="beginDismiss"
      @touchmove="moveDismiss"
      @touchend="finishDismiss"
      @touchcancel="resetDrag"
    >
      <span class="mobile-library-sheet-grabber" aria-hidden="true" />
      <div ref="body" class="mobile-library-sheet-body">
        <p class="mobile-library-sheet-head" :data-test="`${testId}-head`">{{ title }}</p>
        <slot />
        <button
          class="mobile-library-sheet-done"
          type="button"
          data-sheet-close
          :data-test="`${testId}-done`"
          @click="emit('close')"
        >
          {{ doneLabel }}
        </button>
      </div>
    </div>
  </div>
</template>
