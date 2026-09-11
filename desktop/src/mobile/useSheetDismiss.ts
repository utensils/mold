/*
 * Drag a bottom sheet down to dismiss it — the one implementation.
 *
 * Every `.mobile-sheet-*` surface (Style, More settings, Library, Filters, Add
 * a machine) carried a verbatim copy of this block: the same 96px threshold,
 * the same 0.82 damping, the same 280px cap and the same 0.24/320 scrim curve.
 * Five copies is five places for a coefficient to drift, and the block had no
 * test at all — the sheets' own tests only assert a grabber exists.
 *
 * The gesture is deliberately conservative, because the sheet body under the
 * finger owns the far more common gesture (scrolling):
 *
 * - One finger only. A second finger means a pinch or a two-handed scroll.
 * - The body must be at its top. Flicking a scrolled list back up must not
 *   throw the sheet off the screen.
 * - Never from a control. A field, a button or a slider keeps its own gesture.
 * - Downward and more vertical than horizontal, decided on the first move.
 *
 * The identifier of the starting touch is retained so a finger landing
 * mid-gesture pauses the drag rather than teleporting the panel to it.
 */
import { computed, ref, type Ref } from "vue";

/** How far the panel must fall before releasing dismisses the sheet. */
const DISMISS_DISTANCE = 96;
/** Finger travel is damped, so the panel trails the thumb rather than racing it. */
const DRAG_DAMPING = 0.82;
/** The panel stops here; past it the gesture is committed and the offset says nothing more. */
const MAX_DRAG = 280;
/** The scrim clears over this much travel, never past the floor. */
const SCRIM_FADE_DISTANCE = 320;
/** The sheet behind the scrim stays legible as a dismissal, not as a page. */
const SCRIM_FLOOR = 0.24;

/** Anything with its own gesture: the drag must not steal from these. */
const INTERACTIVE = "input, textarea, select, button, a, [contenteditable='true']";

export interface SheetDismissOptions {
  /** The scrolling body. A drag only begins while it is at its top. */
  body: Ref<HTMLElement | null>;
  /** Read-first sheets opt out entirely; omitted means always draggable. */
  enabled?: Ref<boolean> | undefined;
  /** Called once, on release, when the panel fell far enough. */
  onDismiss: () => void;
}

export interface SheetDismiss {
  /** True while the panel is following the finger — drop transitions for it. */
  dragging: Ref<boolean>;
  /** Bind to the panel. Undefined transform while at rest, so CSS owns the resting state. */
  panelStyle: Ref<{ transform: string | undefined }>;
  /** Bind to the scrim. */
  backdropStyle: Ref<{ opacity: number | undefined }>;
  beginDismiss: (event: TouchEvent) => void;
  moveDismiss: (event: TouchEvent) => void;
  finishDismiss: () => void;
  /** Bind to `touchcancel`, and call when the sheet closes or unmounts. */
  resetDrag: () => void;
}

export function useSheetDismiss(options: SheetDismissOptions): SheetDismiss {
  const dragOffset = ref(0);
  const dragging = ref(false);
  let dragTouchId: number | null = null;
  let dragStartX = 0;
  let dragStartY = 0;

  const panelStyle = computed(() => ({
    transform: dragOffset.value > 0 ? `translateY(${dragOffset.value}px)` : undefined,
  }));
  const backdropStyle = computed(() => ({
    opacity:
      dragOffset.value > 0
        ? Math.max(SCRIM_FLOOR, 1 - dragOffset.value / SCRIM_FADE_DISTANCE)
        : undefined,
  }));

  function resetDrag(): void {
    dragTouchId = null;
    dragOffset.value = 0;
    dragging.value = false;
  }

  function beginDismiss(event: TouchEvent): void {
    if (
      options.enabled?.value === false ||
      event.touches.length !== 1 ||
      (options.body.value?.scrollTop ?? 0) > 0 ||
      (event.target instanceof Element && Boolean(event.target.closest(INTERACTIVE)))
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
    dragOffset.value = Math.min(MAX_DRAG, deltaY * DRAG_DAMPING);
    event.preventDefault();
  }

  function finishDismiss(): void {
    if (dragTouchId === null) return;
    const dismiss = dragOffset.value >= DISMISS_DISTANCE;
    resetDrag();
    if (dismiss) options.onDismiss();
  }

  return {
    dragging,
    panelStyle,
    backdropStyle,
    beginDismiss,
    moveDismiss,
    finishDismiss,
    resetDrag,
  };
}
