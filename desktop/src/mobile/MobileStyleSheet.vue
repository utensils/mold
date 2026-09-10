<script setup lang="ts">
/*
 * Style bottom sheet — the iPhone's host for the SHARED style list.
 *
 * The phone used to pick a style through a native `<select>`, which can say a
 * label and nothing else: no size, no on-GPU state, no description, no
 * type-to-filter, and no way to see that a restored print's style is simply
 * not on any machine yet. Desktop's popover and web's Create chip already open
 * `@studio/components/StyleMenu.vue`; this is the same list under a thumb.
 *
 * It follows MobileLibrarySheet / MobileAdvancedSheet rather than
 * @ui/SheetPanel: a fixed overlay whose body owns its scroll and safe-area
 * insets, Android Back through `useMobileBack`, an overlay-stack `isTop()`
 * gate so a sheet above it keeps Escape, and a downward drag that dismisses.
 *
 * The list is mounted only while the sheet is open — StyleMenu resolves its
 * keyboard cursor on mount, so a parked copy would open on a stale row.
 */
import { computed, nextTick, onBeforeUnmount, ref, toRef, watch } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";
import StyleMenu from "@studio/components/StyleMenu.vue";
import type { StyleMenuModel } from "@studio/lib/styleMenu";
import { useMobileBack } from "./useMobileBack";

const props = withDefaults(
  defineProps<{
    open: boolean;
    models: StyleMenuModel[];
    selected: StyleMenuModel | null;
    /** The form's style when no reachable machine has it. */
    missingModel?: string | null;
    /** What this sheet holds, in the section's own words. */
    kicker?: string | null;
    emptyLabel?: string | null;
    availabilityTag?: ((model: StyleMenuModel) => string | null) | null;
    disabledReason?: ((model: StyleMenuModel) => string | null) | null;
    title?: string;
  }>(),
  {
    missingModel: null,
    kicker: null,
    emptyLabel: null,
    availabilityTag: null,
    disabledReason: null,
    title: "Style",
  },
);

const emit = defineEmits<{
  close: [];
  pick: [model: StyleMenuModel];
  "pick-missing": [model: string];
  browse: [];
}>();

useMobileBack(toRef(props, "open"), () => emit("close"));
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-style-sheet");
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
  async (open) => {
    if (open) {
      restoreFocus = document.activeElement as HTMLElement | null;
      // The panel, never the filter field: a sheet that raises the keyboard
      // hides the very rows it was opened to show.
      await nextTick();
      if (props.open && isTop()) panel.value?.focus?.();
    } else {
      resetDrag();
      restoreFocus?.focus?.();
      restoreFocus = null;
    }
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  resetDrag();
  restoreFocus = null;
});

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

/**
 * Escape and Tab. The arrow walk and Enter belong to StyleMenu's own root,
 * which is inside this panel, so they never reach here.
 */
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
    } else if (
      !event.shiftKey &&
      (document.activeElement === last || document.activeElement === panel.value)
    ) {
      event.preventDefault();
      first.focus();
    }
  }
}
</script>

<template>
  <div
    class="mobile-sheet mobile-style-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-modal="true"
    :inert="!open"
    :aria-label="title"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-style-sheet"
    @keydown="onKeydown"
  >
    <button
      class="mobile-sheet-scrim"
      type="button"
      data-sheet-close
      data-test="mobile-style-sheet-scrim"
      :aria-label="`Close ${title}`"
      :style="backdropStyle"
      @click="emit('close')"
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
      <header class="mobile-sheet-head">
        <!-- The title is centred against the whole row, so the leading side
             holds the same width as the trailing control even when empty. -->
        <span class="mobile-sheet-head-slot" aria-hidden="true" />
        <div class="mobile-sheet-heading">
          <h2 class="mobile-sheet-title">{{ title }}</h2>
          <p v-if="kicker" class="mobile-sheet-kicker">{{ kicker }}</p>
        </div>
        <div class="mobile-sheet-head-slot">
          <button
            class="mobile-sheet-action is-strong"
            type="button"
            data-sheet-close
            data-test="mobile-style-sheet-done"
            @click="emit('close')"
          >
            Done
          </button>
        </div>
      </header>
      <div ref="body" class="mobile-sheet-body">
        <StyleMenu
          v-if="open"
          :models="models"
          :selected="selected"
          :missing-model="missingModel"
          :kicker="null"
          :empty-label="emptyLabel"
          :availability-tag="availabilityTag"
          :disabled-reason="disabledReason"
          touch
          :autofocus-filter="false"
          @pick="emit('pick', $event)"
          @pick-missing="emit('pick-missing', $event)"
          @browse="emit('browse')"
        />
      </div>
    </div>
  </div>
</template>
