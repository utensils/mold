<script setup lang="ts">
import { ref, toRef, watch, nextTick, onBeforeUnmount } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";

const props = defineProps<{
  open: boolean;
  count: number;
}>();

const emit = defineEmits<{
  (event: "close"): void;
  (event: "reset"): void;
}>();
const panel = ref<HTMLElement | null>(null);
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-more-settings");
let previousFocus: HTMLElement | null = null;
watch(
  () => props.open,
  async (open) => {
    if (open) {
      previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      await nextTick();
      if (props.open) panel.value?.focus();
    } else {
      previousFocus?.focus();
      previousFocus = null;
    }
  },
  { immediate: true },
);
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
  previousFocus = null;
});
</script>

<template>
  <!--
    Full-screen Advanced surface (Mold Studio iOS spec). The hosted controls
    stay mounted whether the sheet is open or not so their validity wiring keeps
    reporting and the Generate flow can read every advanced field; the overlay
    only toggles visibility.
  -->
  <div
    ref="panel"
    tabindex="-1"
    class="mobile-advanced-sheet"
    :inert="!open"
    aria-modal="true"
    @keydown="onKeydown"
    :class="{ 'is-open': open }"
    role="dialog"
    aria-label="More settings"
    :aria-hidden="open ? undefined : 'true'"
    data-test="mobile-advanced-sheet"
  >
    <header class="mobile-advanced-sheet-head">
      <div class="mobile-advanced-sheet-title">
        <strong>More settings</strong>
        <span
          v-if="count > 0"
          class="mobile-advanced-sheet-badge"
          data-test="mobile-advanced-count"
        >
          {{ count }}
        </span>
      </div>
      <div class="mobile-advanced-sheet-actions">
        <button
          class="mobile-advanced-sheet-reset"
          type="button"
          data-test="mobile-advanced-reset"
          @click="$emit('reset')"
        >
          Reset
        </button>
        <button
          class="mobile-advanced-sheet-close"
          type="button"
          aria-label="Done with more settings"
          data-test="mobile-advanced-close"
          @click="$emit('close')"
        >
          Done
        </button>
      </div>
    </header>
    <div class="mobile-advanced-sheet-body">
      <slot />
    </div>
  </div>
</template>
