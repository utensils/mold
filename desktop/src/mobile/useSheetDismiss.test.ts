import { describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { useSheetDismiss } from "./useSheetDismiss";

/**
 * jsdom has no TouchEvent, and the composable reads `touches[i].identifier`,
 * `.clientX` and `.clientY` plus `preventDefault()`. A plain object with those
 * fields is the whole contract, so build one rather than a real event.
 */
function touchEvent(
  touches: { id?: number; x: number; y: number }[],
  target: EventTarget | null = null,
): TouchEvent {
  const preventDefault = vi.fn();
  return {
    touches: touches.map((touch) => ({
      identifier: touch.id ?? 0,
      clientX: touch.x,
      clientY: touch.y,
    })),
    target,
    preventDefault,
  } as unknown as TouchEvent;
}

function scroller(scrollTop: number): HTMLElement {
  const element = document.createElement("div");
  Object.defineProperty(element, "scrollTop", { value: scrollTop, writable: true });
  return element;
}

function setup(options: { scrollTop?: number; enabled?: boolean } = {}) {
  const onDismiss = vi.fn();
  const body = ref<HTMLElement | null>(scroller(options.scrollTop ?? 0));
  const dismiss = useSheetDismiss({
    body,
    enabled: options.enabled === undefined ? undefined : ref(options.enabled),
    onDismiss,
  });
  return { ...dismiss, body, onDismiss };
}

/** Drag from (x0,y0) by (dx,dy) and release. */
function drag(
  sheet: ReturnType<typeof setup>,
  start: { x: number; y: number },
  delta: { x: number; y: number },
  target: EventTarget | null = null,
): TouchEvent {
  sheet.beginDismiss(touchEvent([{ x: start.x, y: start.y }], target));
  const move = touchEvent([{ x: start.x + delta.x, y: start.y + delta.y }], target);
  sheet.moveDismiss(move);
  sheet.finishDismiss();
  return move;
}

describe("useSheetDismiss", () => {
  it("dismisses on a downward drag past the threshold and follows the finger on the way", () => {
    const sheet = setup();
    sheet.beginDismiss(touchEvent([{ x: 100, y: 40 }]));
    const move = touchEvent([{ x: 100, y: 240 }]);
    sheet.moveDismiss(move);
    expect(sheet.dragging.value).toBe(true);
    // 200px of finger travel, damped by 0.82.
    expect(sheet.panelStyle.value.transform).toBe("translateY(164px)");
    expect(move.preventDefault).toHaveBeenCalled();
    sheet.finishDismiss();
    expect(sheet.onDismiss).toHaveBeenCalledTimes(1);
    expect(sheet.dragging.value).toBe(false);
    expect(sheet.panelStyle.value.transform).toBeUndefined();
  });

  it("springs back when the drag stops short of the threshold", () => {
    const sheet = setup();
    // 96px threshold ÷ 0.82 = 117px of travel; 100px is short.
    drag(sheet, { x: 100, y: 40 }, { x: 0, y: 100 });
    expect(sheet.onDismiss).not.toHaveBeenCalled();
  });

  it("caps the panel offset so a long drag never leaves the screen behind", () => {
    const sheet = setup();
    sheet.beginDismiss(touchEvent([{ x: 100, y: 0 }]));
    sheet.moveDismiss(touchEvent([{ x: 100, y: 900 }]));
    expect(sheet.panelStyle.value.transform).toBe("translateY(280px)");
  });

  it("fades the scrim as the panel falls and never below a legible floor", () => {
    const sheet = setup();
    sheet.beginDismiss(touchEvent([{ x: 100, y: 0 }]));
    expect(sheet.backdropStyle.value.opacity).toBeUndefined();
    sheet.moveDismiss(touchEvent([{ x: 100, y: 100 }]));
    expect(sheet.backdropStyle.value.opacity).toBeCloseTo(1 - 82 / 320, 5);
    // The 280px cap is reached before the fade curve would, so the floor holds.
    sheet.moveDismiss(touchEvent([{ x: 100, y: 900 }]));
    expect(sheet.backdropStyle.value.opacity).toBeCloseTo(0.24, 5);
  });

  it("leaves an upward or sideways drag to the sheet itself", () => {
    const up = setup();
    drag(up, { x: 100, y: 300 }, { x: 0, y: -200 });
    expect(up.onDismiss).not.toHaveBeenCalled();

    const sideways = setup();
    sideways.beginDismiss(touchEvent([{ x: 100, y: 40 }]));
    const move = touchEvent([{ x: 400, y: 240 }]);
    sideways.moveDismiss(move);
    expect(sideways.dragging.value).toBe(false);
    expect(move.preventDefault).not.toHaveBeenCalled();
    sideways.finishDismiss();
    expect(sideways.onDismiss).not.toHaveBeenCalled();
  });

  it("never drags a scrolled body, so a flick back to the top is not a dismissal", () => {
    const sheet = setup({ scrollTop: 24 });
    drag(sheet, { x: 100, y: 40 }, { x: 0, y: 240 });
    expect(sheet.onDismiss).not.toHaveBeenCalled();
    expect(sheet.dragging.value).toBe(false);
  });

  it("never drags from a control, so a slider or a field keeps its own gesture", () => {
    const sheet = setup();
    const field = document.createElement("input");
    drag(sheet, { x: 100, y: 40 }, { x: 0, y: 240 }, field);
    expect(sheet.onDismiss).not.toHaveBeenCalled();
  });

  it("never drags while disabled, which is how a read-first sheet opts out", () => {
    const sheet = setup({ enabled: false });
    drag(sheet, { x: 100, y: 40 }, { x: 0, y: 240 });
    expect(sheet.onDismiss).not.toHaveBeenCalled();
    expect(sheet.dragging.value).toBe(false);
  });

  it("ignores a second finger and a move that never began", () => {
    const sheet = setup();
    sheet.beginDismiss(
      touchEvent([
        { id: 1, x: 100, y: 40 },
        { id: 2, x: 200, y: 40 },
      ]),
    );
    sheet.moveDismiss(touchEvent([{ id: 1, x: 100, y: 240 }]));
    sheet.finishDismiss();
    expect(sheet.onDismiss).not.toHaveBeenCalled();

    const stray = setup();
    stray.moveDismiss(touchEvent([{ x: 100, y: 240 }]));
    expect(stray.dragging.value).toBe(false);
    stray.finishDismiss();
    expect(stray.onDismiss).not.toHaveBeenCalled();
  });

  it("tracks the finger that started the drag when another lands mid-gesture", () => {
    const sheet = setup();
    sheet.beginDismiss(touchEvent([{ id: 7, x: 100, y: 40 }]));
    // A second finger arrives: the gesture pauses rather than jumping to it.
    sheet.moveDismiss(
      touchEvent([
        { id: 7, x: 100, y: 240 },
        { id: 8, x: 300, y: 240 },
      ]),
    );
    expect(sheet.dragging.value).toBe(false);
    sheet.moveDismiss(touchEvent([{ id: 7, x: 100, y: 240 }]));
    expect(sheet.panelStyle.value.transform).toBe("translateY(164px)");
  });

  it("cancels cleanly, leaving no offset for the next gesture to inherit", () => {
    const sheet = setup();
    sheet.beginDismiss(touchEvent([{ x: 100, y: 40 }]));
    sheet.moveDismiss(touchEvent([{ x: 100, y: 240 }]));
    sheet.resetDrag();
    expect(sheet.dragging.value).toBe(false);
    expect(sheet.panelStyle.value.transform).toBeUndefined();
    sheet.finishDismiss();
    expect(sheet.onDismiss).not.toHaveBeenCalled();
  });
});
