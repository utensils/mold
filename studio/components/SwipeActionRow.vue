<script setup lang="ts">
/**
 * A list row with trailing swipe actions, the standard phone list pattern.
 *
 * Right-to-left reveals a tray of 44pt buttons; from a REVEALED tray, a full
 * swipe past `SWIPE_COMMIT_FRACTION` of the row commits the action that opted
 * in. The reveal is step one and the tap or the second full swipe is step two,
 * so no destructive action ever happens in one gesture — a diagonal scroll
 * across a running row can only open its tray.
 *
 * Every action is also reachable without the gesture: the trailing "Actions"
 * button opens the same tray from the keyboard and from VoiceOver, because a
 * swipe that is the ONLY path to cancelling a job is not a path at all. The
 * gesture math lives in `studio/lib/swipeAction.ts` and is unit-tested; this
 * file only binds it to pointer events. Nothing here is iOS-specific — the
 * Android shell renders the same component.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import {
  beginSwipe,
  createSwipeState,
  endSwipe,
  moveSwipe,
  swipeIsOpen,
  type SwipeGestureConfig,
  type SwipeRowAction,
} from "../lib/swipeAction";

const props = withDefaults(
  defineProps<{
    actions: readonly SwipeRowAction[];
    /** Names the row for the non-gesture actions button. */
    label: string;
    /** True while one of this row's actions is in flight. */
    disabled?: boolean;
  }>(),
  { disabled: false },
);

const emit = defineEmits<{ (e: "act", id: string): void }>();

/** One 44pt target plus padding per action. */
const ACTION_WIDTH = 88;

const root = ref<HTMLElement | null>(null);
const state = ref(createSwipeState());
const pointerId = ref<number | null>(null);

const trayWidth = computed(() => ACTION_WIDTH * props.actions.length);
const commitAction = computed(
  () => props.actions.find((action) => action.commitOnFullSwipe) ?? null,
);
const open = computed(() => swipeIsOpen(state.value));

function config(): SwipeGestureConfig {
  return {
    trayWidth: trayWidth.value,
    rowWidth: root.value?.getBoundingClientRect().width || trayWidth.value * 3,
    commitEnabled: commitAction.value !== null,
    disabled: props.disabled,
  };
}

function onPointerDown(event: PointerEvent): void {
  if (props.disabled || event.pointerType === "mouse") return;
  pointerId.value = event.pointerId;
  state.value = beginSwipe(state.value, { x: event.clientX, y: event.clientY });
}

function onPointerMove(event: PointerEvent): void {
  if (pointerId.value !== event.pointerId) return;
  const next = moveSwipe(
    state.value,
    { x: event.clientX, y: event.clientY },
    config(),
  );
  // Claim the pointer only once the row has actually won the horizontal axis,
  // so an ambiguous drag still scrolls the list.
  if (next.captured && !state.value.captured) {
    (event.currentTarget as HTMLElement | null)?.setPointerCapture?.(
      event.pointerId,
    );
  }
  state.value = next;
}

function onPointerUp(event: PointerEvent, cancelled = false): void {
  if (pointerId.value !== event.pointerId) return;
  pointerId.value = null;
  const release = endSwipe({ ...state.value, cancelled }, config());
  state.value = release.state;
  if (release.commit && commitAction.value) emit("act", commitAction.value.id);
}

function close(): void {
  state.value = createSwipeState();
}

// The tray can gain or lose an action while open (Retry appears once a hold
// is retryable); keep the revealed offset equal to the tray it now reveals.
watch(trayWidth, (width) => {
  if (open.value) state.value = { ...createSwipeState(), offset: -width };
});

// A pointer landing anywhere outside this row closes its tray, the standard
// phone list behaviour, so at most one row is ever revealed at a time.
function onDocumentPointerDown(event: PointerEvent): void {
  if (!open.value || pointerId.value !== null) return;
  const target = event.target;
  if (target instanceof Node && root.value?.contains(target)) return;
  close();
}
onMounted(() =>
  document.addEventListener("pointerdown", onDocumentPointerDown),
);
onBeforeUnmount(() =>
  document.removeEventListener("pointerdown", onDocumentPointerDown),
);

function act(id: string): void {
  close();
  emit("act", id);
}

function toggleTray(): void {
  state.value = open.value
    ? createSwipeState()
    : { ...createSwipeState(), offset: -trayWidth.value };
}

defineExpose({ close });
</script>

<template>
  <div ref="root" class="swipe-row" data-test="swipe-action-row">
    <div
      class="swipe-row__tray"
      :style="{ width: `${trayWidth}px` }"
      :aria-hidden="open ? undefined : 'true'"
    >
      <button
        v-for="action in actions"
        :key="action.id"
        type="button"
        class="swipe-row__action"
        :class="{ 'swipe-row__action--danger': action.tone === 'danger' }"
        :disabled="disabled"
        :tabindex="open ? 0 : -1"
        :data-test="`swipe-action-${action.id}`"
        @click="act(action.id)"
      >
        {{ action.label }}
      </button>
    </div>
    <div
      class="swipe-row__surface"
      :class="{ 'swipe-row__surface--settling': pointerId === null }"
      :style="{ transform: `translateX(${state.offset}px)` }"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp($event)"
      @pointercancel="onPointerUp($event, true)"
    >
      <slot />
      <button
        v-if="actions.length > 0"
        type="button"
        class="swipe-row__more"
        :aria-label="`Actions for ${label}`"
        :aria-expanded="open"
        data-test="swipe-row-actions"
        @click="toggleTray"
      >
        ⋯
      </button>
    </div>
  </div>
</template>

<style scoped>
.swipe-row {
  position: relative;
  overflow: hidden;
  /* Scoped to the row: the list still scrolls, and the Library grid's pinch
   * and the gallery viewer's swipe are untouched. */
  touch-action: pan-y;
  overscroll-behavior: none;
}
.swipe-row__tray {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  display: flex;
}
.swipe-row__action {
  min-width: 44px;
  min-height: 44px;
  flex: 1;
  border: 0;
  background: var(--surface, transparent);
  color: var(--ink, currentColor);
  font-size: 14px;
}
.swipe-row__action--danger {
  background: var(--stop);
  color: var(--on-status, #fff);
}
.swipe-row__surface {
  position: relative;
  display: flex;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  align-items: center;
  gap: 8px;
  background: var(--bench, var(--bath));
  will-change: transform;
}
.swipe-row__surface--settling {
  transition: transform var(--dur-base, 180ms) var(--ease, ease);
}
.swipe-row__more {
  min-width: 44px;
  min-height: 44px;
  border: 0;
  background: none;
  color: var(--ink-3, currentColor);
  font-size: 16px;
}
@media (prefers-reduced-motion: reduce) {
  .swipe-row__surface--settling {
    transition: none;
  }
}
</style>
