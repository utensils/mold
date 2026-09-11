<script setup lang="ts">
/*
 * Library bottom sheet — the iPhone home for the Library's editors (tag
 * editor, collection checklist, new/rename collection, the "More tags" list,
 * and the viewer's print info). Follows MobileAdvancedSheet
 * rather than @ui/SheetPanel: a fixed overlay whose body owns its scroll and
 * every safe-area inset, with the head row rendered in the body so it can
 * never vanish the way SheetPanel's `full` variant drops its #header slot.
 */
import { useSheetDismiss } from "./useSheetDismiss";
import { useSheetFocus } from "./useSheetFocus";
import { useMobileBack } from "./useMobileBack";
import { ref, toRef } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";

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
    /** Let a downward drag from the top dismiss the sheet. */
    swipeToDismiss?: boolean;
  }>(),
  {
    focusFirstControl: true,
    doneLabel: "Done",
    testId: "mobile-library-sheet",
    touchTargetSize: 46,
    swipeToDismiss: true,
  },
);

const emit = defineEmits<{ close: [] }>();

useMobileBack(toRef(props, "open"), () => emit("close"));
const panel = ref<HTMLElement | null>(null);
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-library-sheet");
const { dragging, panelStyle, backdropStyle, beginDismiss, moveDismiss, finishDismiss, resetDrag } =
  useSheetDismiss({
    enabled: () => props.open && isTop() && props.swipeToDismiss,
    close: () => emit("close"),
  });

// Editing sheets may raise the keyboard immediately. Read-first sheets focus
// the panel so the keyboard waits for an explicit field tap.
const { onKeydown } = useSheetFocus({
  panel,
  open: () => props.open,
  isTop,
  onClose: () => emit("close"),
  onBeforeClose: resetDrag,
  firstControl: () =>
    props.focusFirstControl
      ? (panel.value?.querySelector<HTMLElement>(
          "input, textarea, select, button:not([data-sheet-close])",
        ) ?? null)
      : null,
});

</script>

<template>
  <div
    class="mobile-library-sheet"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-modal="true"
    :inert="!open"
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
      <div class="mobile-library-sheet-body">
        <p class="mobile-library-sheet-head" :data-test="`${testId}-head`">{{ title }}</p>
        <slot />
      </div>
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
</template>
