/**
 * Swipe-to-act gesture math for a list row, kept pure so it can be tested
 * without a DOM and shared by every phone surface (iOS and Android render the
 * same Vue).
 *
 * The reveal is step one and the tap or full-swipe commit is step two, so a
 * destructive action is never one gesture. The axis lock is what keeps the
 * row from fighting the list scroll: until the pointer clears
 * `SWIPE_AXIS_LOCK_PX` the gesture is undecided and the row does not move,
 * and once the scroll wins it stays won for the rest of the gesture.
 */

/** Pointer travel before the gesture picks an axis. */
export const SWIPE_AXIS_LOCK_PX = 12;
/** Fraction of the row width at which a full swipe commits. */
export const SWIPE_COMMIT_FRACTION = 0.6;
/** Resistance applied past the tray so the row never tracks the finger 1:1. */
const RUBBER_BAND_RESISTANCE = 0.35;

export type SwipePhase = "idle" | "undecided" | "horizontal" | "vertical";

export interface SwipeGestureState {
  phase: SwipePhase;
  /** Row translation in px. 0 is closed; negative reveals the trailing tray. */
  offset: number;
  /** Offset the current gesture started from. */
  startOffset: number;
  startX: number;
  startY: number;
  /** Un-damped reveal the finger asked for. The full-swipe threshold reads
   * this rather than `offset`, so rubber-banding changes how the row LOOKS
   * without changing how far someone has to swipe to commit. */
  intent: number;
  /** Whether the row is claiming the horizontal pan from the list scroll. */
  captured: boolean;
  /** Set by the shell on pointercancel so release restores rather than acts. */
  cancelled: boolean;
}

export interface SwipeGestureConfig {
  /** Total width of the revealed action tray. */
  trayWidth: number;
  /** Row width, which the full-swipe fraction is measured against. */
  rowWidth: number;
  /** Whether a full swipe commits an action at all. */
  commitEnabled?: boolean;
  /** True while an action is in flight — the row must not accept a second. */
  disabled?: boolean;
}

export interface SwipeRelease {
  state: SwipeGestureState;
  /** Whether the full swipe committed its action. */
  commit: boolean;
}

export function createSwipeState(): SwipeGestureState {
  return {
    phase: "idle",
    offset: 0,
    intent: 0,
    startOffset: 0,
    startX: 0,
    startY: 0,
    captured: false,
    cancelled: false,
  };
}

export function swipeIsOpen(state: SwipeGestureState): boolean {
  return state.offset < 0;
}

export function beginSwipe(
  state: SwipeGestureState,
  point: { x: number; y: number },
): SwipeGestureState {
  return {
    ...state,
    phase: "undecided",
    startOffset: state.offset,
    startX: point.x,
    startY: point.y,
    captured: false,
    cancelled: false,
  };
}

/** Sub-linear travel past the tray, clamped to the row so it cannot fly off. */
function rubberBand(overshoot: number, rowWidth: number): number {
  return Math.min(overshoot * RUBBER_BAND_RESISTANCE, rowWidth);
}

export function moveSwipe(
  state: SwipeGestureState,
  point: { x: number; y: number },
  config: SwipeGestureConfig,
): SwipeGestureState {
  if (config.disabled) return { ...createSwipeState(), offset: state.offset };
  if (state.phase === "idle" || state.phase === "vertical") return state;

  const dx = point.x - state.startX;
  const dy = point.y - state.startY;
  let phase: SwipePhase = state.phase;
  if (phase === "undecided") {
    if (Math.abs(dx) < SWIPE_AXIS_LOCK_PX && Math.abs(dy) < SWIPE_AXIS_LOCK_PX)
      return { ...state, phase, captured: false };
    // The list scroll wins a tie: a row that stole an ambiguous drag would
    // make the queue impossible to scroll past.
    phase = Math.abs(dx) > Math.abs(dy) ? "horizontal" : "vertical";
    if (phase === "vertical")
      return {
        ...state,
        phase,
        offset: state.startOffset,
        intent: 0,
        captured: false,
      };
  }

  const raw = state.startOffset + dx;
  // Closed is the hard stop in the leading direction; there is no leading tray.
  const offset =
    raw >= 0
      ? 0
      : raw >= -config.trayWidth
        ? raw
        : -(
            config.trayWidth +
            rubberBand(-raw - config.trayWidth, config.rowWidth)
          );
  return {
    ...state,
    phase,
    offset: Math.max(offset, -config.rowWidth),
    intent: Math.max(0, -raw),
    captured: true,
  };
}

export function endSwipe(
  state: SwipeGestureState,
  config: SwipeGestureConfig,
): SwipeRelease {
  const settled = (offset: number): SwipeRelease => ({
    state: { ...createSwipeState(), offset },
    commit: false,
  });
  if (config.disabled) return settled(state.offset);
  if (state.cancelled) return settled(state.startOffset);
  if (state.phase !== "horizontal") return settled(state.offset);

  const travelled = -state.offset;
  if (
    config.commitEnabled !== false &&
    state.intent >= config.rowWidth * SWIPE_COMMIT_FRACTION
  ) {
    return { state: { ...createSwipeState(), offset: 0 }, commit: true };
  }
  return settled(travelled >= config.trayWidth / 2 ? -config.trayWidth : 0);
}
