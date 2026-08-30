import { describe, expect, it } from "vitest";
import {
  SWIPE_AXIS_LOCK_PX,
  SWIPE_AXIS_INTENT_RATIO,
  SWIPE_COMMIT_FRACTION,
  beginSwipe,
  createSwipeState,
  endSwipe,
  moveSwipe,
  resolveSwipeAxis,
  swipeIsOpen,
  type SwipeGestureConfig,
} from "./swipeAction";

const config: SwipeGestureConfig = { trayWidth: 96, rowWidth: 390 };

function drag(dx: number, dy = 0, options: Partial<SwipeGestureConfig> = {}) {
  let state = beginSwipe(createSwipeState(), { x: 300, y: 100 });
  state = moveSwipe(
    state,
    { x: 300 + dx, y: 100 + dy },
    { ...config, ...options },
  );
  return state;
}

describe("swipe gesture math", () => {
  it("stays idle until the pointer clears the axis lock", () => {
    expect(drag(-(SWIPE_AXIS_LOCK_PX - 1)).phase).toBe("undecided");
    expect(drag(-(SWIPE_AXIS_LOCK_PX - 1)).offset).toBe(0);
  });

  it("locks horizontal on a right-to-left drag and reveals the tray", () => {
    const state = drag(-40);
    expect(state.phase).toBe("horizontal");
    expect(state.offset).toBe(-40);
  });

  it("locks vertical on a mostly-vertical drag and never moves the row", () => {
    const state = drag(-20, -60);
    expect(state.phase).toBe("vertical");
    expect(state.offset).toBe(0);
  });

  it("keeps early diagonal scroll jitter inert until vertical intent wins", () => {
    let state = beginSwipe(createSwipeState(), { x: 300, y: 100 });
    state = moveSwipe(state, { x: 287, y: 111 }, config);
    expect(state.phase).toBe("undecided");
    expect(state.offset).toBe(0);
    expect(state.captured).toBe(false);

    state = moveSwipe(state, { x: 280, y: 160 }, config);
    expect(state.phase).toBe("vertical");
    expect(state.offset).toBe(0);
    expect(state.captured).toBe(false);
  });

  it("requires clear axis dominance after crossing the travel threshold", () => {
    expect(resolveSwipeAxis(-20, 18)).toBe("undecided");
    expect(resolveSwipeAxis(-30, 20)).toBe("horizontal");
    expect(resolveSwipeAxis(-20, 30)).toBe("vertical");
    expect(SWIPE_AXIS_INTENT_RATIO).toBeGreaterThan(1);
  });

  it("stays vertical for the rest of the gesture once the scroll wins", () => {
    let state = beginSwipe(createSwipeState(), { x: 300, y: 100 });
    state = moveSwipe(state, { x: 290, y: 40 }, config);
    state = moveSwipe(state, { x: 100, y: 40 }, config);
    expect(state.phase).toBe("vertical");
    expect(state.offset).toBe(0);
  });

  it("does not reveal anything on a left-to-right drag from closed", () => {
    const state = drag(60);
    expect(state.offset).toBe(0);
  });

  it("rubber-bands past the tray instead of tracking the finger", () => {
    const state = drag(-200);
    expect(state.offset).toBeLessThan(-config.trayWidth);
    expect(state.offset).toBeGreaterThan(-200);
    expect(Math.abs(state.offset)).toBeLessThanOrEqual(config.rowWidth);
  });

  it("closes on release before the tray is half revealed", () => {
    const outcome = endSwipe(drag(-30), config);
    expect(outcome.state.offset).toBe(0);
    expect(outcome.commit).toBe(false);
    expect(swipeIsOpen(outcome.state)).toBe(false);
  });

  it("settles open on release past half the tray", () => {
    const outcome = endSwipe(drag(-70), config);
    expect(outcome.state.offset).toBe(-config.trayWidth);
    expect(outcome.commit).toBe(false);
    expect(swipeIsOpen(outcome.state)).toBe(true);
  });

  it("a full swipe from a closed row only reveals the tray", () => {
    const past = config.rowWidth * SWIPE_COMMIT_FRACTION + 20;
    const outcome = endSwipe(drag(-past), config);
    expect(outcome.commit).toBe(false);
    expect(outcome.state.offset).toBe(-config.trayWidth);
    expect(swipeIsOpen(outcome.state)).toBe(true);
  });

  it("commits a full swipe past the row-width fraction from a revealed tray", () => {
    const opened = endSwipe(drag(-70), config).state;
    const past = config.rowWidth * SWIPE_COMMIT_FRACTION + 20;
    let state = beginSwipe(opened, { x: 300, y: 100 });
    state = moveSwipe(state, { x: 300 - past, y: 100 }, config);
    const outcome = endSwipe(state, config);
    expect(outcome.commit).toBe(true);
    expect(outcome.state.offset).toBe(0);
  });

  it("never commits when no action opted into the full swipe", () => {
    const off = { ...config, commitEnabled: false };
    const opened = endSwipe(drag(-70, 0, { commitEnabled: false }), off).state;
    const past = config.rowWidth * SWIPE_COMMIT_FRACTION + 20;
    let state = beginSwipe(opened, { x: 300, y: 100 });
    state = moveSwipe(state, { x: 300 - past, y: 100 }, off);
    const outcome = endSwipe(state, off);
    expect(outcome.commit).toBe(false);
    expect(outcome.state.offset).toBe(-config.trayWidth);
  });

  it("closes an open row with a left-to-right drag", () => {
    const opened = endSwipe(drag(-70), config).state;
    let state = beginSwipe(opened, { x: 300, y: 100 });
    state = moveSwipe(state, { x: 360, y: 100 }, config);
    expect(state.offset).toBe(-config.trayWidth + 60);
    expect(endSwipe(state, config).state.offset).toBe(0);
  });

  it("cannot be dragged past closed from an open row", () => {
    const opened = endSwipe(drag(-70), config).state;
    let state = beginSwipe(opened, { x: 300, y: 100 });
    state = moveSwipe(
      state,
      { x: 300 + config.trayWidth + 50, y: 100 },
      config,
    );
    expect(state.offset).toBe(0);
  });

  it("is inert while an action is in flight", () => {
    const state = drag(-80, 0, { disabled: true });
    expect(state.phase).toBe("idle");
    expect(state.offset).toBe(0);
    expect(endSwipe(state, { ...config, disabled: true }).commit).toBe(false);
  });

  it("cancels back to its starting offset", () => {
    const opened = endSwipe(drag(-70), config).state;
    let state = beginSwipe(opened, { x: 300, y: 100 });
    state = moveSwipe(state, { x: 220, y: 100 }, config);
    expect(endSwipe({ ...state, cancelled: true }, config).state.offset).toBe(
      -config.trayWidth,
    );
  });

  it("reports whether the row should claim the horizontal pan", () => {
    expect(drag(-40).captured).toBe(true);
    expect(drag(-10, -50).captured).toBe(false);
  });
});
