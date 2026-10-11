/**
 * How tall the composer's prompt is. The composer sits on the bottom edge of
 * the canvas, so its prompt grows UPWARD as the handle above it is dragged —
 * the native Mac composer's policy (`PromptEditorHeight.swift`), ported number
 * for number so both desktop apps feel the same.
 *
 * The preference is the person's; the window only clamps what is DRAWN. A
 * short window never rewrites the saved height, so making the window tall
 * again brings the prompt back to the size they chose.
 */
import { MIN_CANVAS_HEIGHT } from "./benchLayout";

export const PROMPT_HEIGHT_INITIAL = 72;
export const PROMPT_HEIGHT_MIN = 48;
export const PROMPT_HEIGHT_MAX = 560;
/** One keyboard step (↑/↓ on the handle). */
export const PROMPT_HEIGHT_STEP = 40;

/** Everything in the window that is not canvas or prompt: title bar, view
 *  toolbar, status bar and the composer's own control row and padding. */
const CHROME_AROUND_PROMPT = 260;

const STORAGE_KEY = "mold.desktop.promptEditorHeight.v1";

export function resolvePromptHeight(preferred: number, available: number): number {
  const ceiling = Math.max(0, Math.min(PROMPT_HEIGHT_MAX, available));
  const floor = Math.min(PROMPT_HEIGHT_MIN, ceiling);
  const wanted = Number.isFinite(preferred) ? preferred : PROMPT_HEIGHT_INITIAL;
  return Math.min(ceiling, Math.max(floor, wanted));
}

/** A drag's vertical translation, in screen pixels; upward is negative. */
export function draggedPromptHeight(from: number, translation: number, available: number): number {
  return resolvePromptHeight(from - translation, available);
}

/** The room a window of this height can give the prompt without squeezing
 *  the canvas below its floor (which would clip Generate off the bottom). */
export function availablePromptHeight(viewportHeight: number): number {
  return Math.max(0, viewportHeight - MIN_CANVAS_HEIGHT - CHROME_AROUND_PROMPT);
}

export function loadPromptHeight(): number {
  try {
    const stored = Number(localStorage.getItem(STORAGE_KEY));
    return localStorage.getItem(STORAGE_KEY) !== null && Number.isFinite(stored)
      ? resolvePromptHeight(stored, PROMPT_HEIGHT_MAX)
      : PROMPT_HEIGHT_INITIAL;
  } catch {
    return PROMPT_HEIGHT_INITIAL;
  }
}

export function savePromptHeight(height: number): void {
  try {
    localStorage.setItem(STORAGE_KEY, String(Math.round(height)));
  } catch {
    // A private window or blocked storage keeps the size for this session only.
  }
}
