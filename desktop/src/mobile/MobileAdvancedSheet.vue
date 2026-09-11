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
import { useSheetDismiss } from "./useSheetDismiss";
import { useSheetFocus } from "./useSheetFocus";
import { useMobileBack } from "./useMobileBack";
import { ref, toRef } from "vue";
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
const { isTop } = useOverlayStack(toRef(props, "open"), "mobile-more-settings");
const { dragging, panelStyle, backdropStyle, beginDismiss, moveDismiss, finishDismiss, resetDrag } =
  useSheetDismiss({
    enabled: () => props.open && isTop(),
    close: () => emit("close"),
  });

const { onKeydown } = useSheetFocus({
  panel,
  open: () => props.open,
  isTop,
  onClose: () => emit("close"),
  onBeforeClose: resetDrag,
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
      <div class="mobile-advanced-sheet-body">
        <slot />
      </div>
    </div>
  </div>
</template>
