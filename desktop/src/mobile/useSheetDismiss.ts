import { computed, onBeforeUnmount, ref, watch } from "vue";

/** One downward gesture for phone sheets. A drag owns its initial direction;
 * scrolling, controls and multi-touch never turn into a dismissal mid-gesture.
 * Chrome outside the scroll body remains draggable even after scrolling. */
export function useSheetDismiss(options: {
  enabled: () => boolean;
  close: () => void;
  canStart?: (target: Element) => boolean;
}) {
  const offset = ref(0);
  const dragging = ref(false);
  let touchId: number | null = null;
  let startX = 0;
  let startY = 0;

  const panelStyle = computed(() => ({
    transform: offset.value > 0 ? `translateY(${offset.value}px)` : undefined,
  }));
  const backdropStyle = computed(() => ({
    opacity: offset.value > 0 ? Math.max(0.24, 1 - offset.value / 320) : undefined,
  }));

  function resetDrag(): void {
    touchId = null;
    offset.value = 0;
    dragging.value = false;
  }

  function beginDismiss(event: TouchEvent): void {
    resetDrag();
    const target = event.target;
    if (!options.enabled() || event.touches.length !== 1 || !(target instanceof Element)) return;
    if (target.closest("input, textarea, select, button, a, [contenteditable], [role='slider']"))
      return;
    if (options.canStart && !options.canStart(target)) return;
    // Only scroll containers on the touch's path matter. A scrolled sibling
    // body must not disable the grabber or header above it.
    for (let node: Element | null = target; node; node = node.parentElement) {
      if (node.scrollTop > 0) return;
      if (node === event.currentTarget) break;
    }
    const touch = event.touches[0];
    if (!touch) return;
    touchId = touch.identifier;
    startX = touch.clientX;
    startY = touch.clientY;
  }

  function moveDismiss(event: TouchEvent): void {
    if (touchId === null) return;
    const touch = Array.from(event.touches).find((candidate) => candidate.identifier === touchId);
    if (!options.enabled() || event.touches.length !== 1 || !touch) {
      resetDrag();
      return;
    }
    const dx = touch.clientX - startX;
    const dy = touch.clientY - startY;
    if (!dragging.value) {
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 8) return;
      if (dy <= 0 || Math.abs(dx) >= dy) {
        resetDrag();
        return;
      }
      dragging.value = true;
    }
    offset.value = Math.min(280, Math.max(0, dy) * 0.82);
    if (event.cancelable) event.preventDefault();
  }

  function finishDismiss(event: TouchEvent): void {
    const dismiss =
      touchId !== null &&
      event.touches.length === 0 &&
      Array.from(event.changedTouches).some((touch) => touch.identifier === touchId) &&
      options.enabled() &&
      offset.value >= 96;
    resetDrag();
    if (dismiss) options.close();
  }

  watch(options.enabled, resetDrag);
  onBeforeUnmount(resetDrag);
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
