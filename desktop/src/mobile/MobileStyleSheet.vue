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
import { ref, toRef } from "vue";
import { useOverlayStack } from "@ui/lib/overlayStack";
import StyleMenu from "@studio/components/StyleMenu.vue";
import type { StyleMenuModel } from "@studio/lib/styleMenu";
import { useSheetDismiss } from "./useSheetDismiss";
import { useSheetFocus } from "./useSheetFocus";
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
const { dragging, panelStyle, backdropStyle, beginDismiss, moveDismiss, finishDismiss, resetDrag } =
  useSheetDismiss({
    enabled: () => props.open && isTop(),
    close: () => emit("close"),
  });

// The panel, never the filter field: a sheet that raises the keyboard hides
// the very rows it was opened to show. Escape and Tab belong to the top sheet
// only; StyleMenu's own root owns the arrow walk and Enter inside it.
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
      <div class="mobile-sheet-body">
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
