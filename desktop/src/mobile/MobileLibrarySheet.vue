<script setup lang="ts">
/*
 * Library bottom sheet — the iPhone home for the Library's editors (tag
 * editor, collection checklist, new/rename collection, the "More tags" list,
 * and the viewer's print info). Follows MobileSeamSheet / MobileAdvancedSheet
 * rather than @ui/SheetPanel: a fixed overlay whose body owns its scroll and
 * every safe-area inset, with the head row rendered in the body so it can
 * never vanish the way SheetPanel's `full` variant drops its #header slot.
 */
import { onBeforeUnmount, ref, watch } from "vue";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title: string;
    /** Trailing label of the one closing control. */
    doneLabel?: string;
    testId?: string;
  }>(),
  { doneLabel: "Done", testId: "mobile-library-sheet" },
);

const emit = defineEmits<{ close: [] }>();

const panel = ref<HTMLElement | null>(null);
let restoreFocus: HTMLElement | null = null;

watch(
  () => props.open,
  (open) => {
    if (open) {
      restoreFocus = document.activeElement as HTMLElement | null;
      // Land focus on the first editable control so the keyboard rises with
      // the sheet; fall back to the panel itself.
      queueMicrotask(() => {
        const first = panel.value?.querySelector<HTMLElement>(
          "input, textarea, select, button:not([data-sheet-close])",
        );
        (first ?? panel.value)?.focus?.();
      });
    } else {
      restoreFocus?.focus?.();
      restoreFocus = null;
    }
  },
);

onBeforeUnmount(() => {
  restoreFocus = null;
});

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
    @keydown="onKeydown"
  >
    <button
      class="mobile-library-sheet-backdrop"
      type="button"
      data-sheet-close
      :aria-label="`Close ${title}`"
      :data-test="`${testId}-backdrop`"
      @click="emit('close')"
    />
    <div ref="panel" class="mobile-library-sheet-panel" tabindex="-1">
      <span class="mobile-library-sheet-grabber" aria-hidden="true" />
      <div class="mobile-library-sheet-body">
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
