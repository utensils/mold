import { describe, expect, it } from "vitest";
import { COMPOSER_FALLBACK_HEIGHT, MIN_CANVAS_HEIGHT } from "./benchLayout";
import { PROMPT_HEIGHT_INITIAL, availablePromptHeight } from "./promptEditorHeight";

/*
 * The workbench is `overflow-hidden`, so whatever the canvas floor fails to
 * leave is cut off the bottom — and the bottom is the composer, which carries
 * the only Generate button. This pins the reservation itself rather than any
 * one number.
 */

// TitleBar + the per-view toolbar + the status bar (ui/mold-desktop.css).
const SHELL_CHROME = 44 + 40 + 26;
/** Window heights from desktop/src-tauri/tauri.conf.json. */
const MIN_WINDOW = 700;

describe("workbench canvas floor", () => {
  it("leaves the composer its pixels at the minimum window", () => {
    expect(MIN_CANVAS_HEIGHT + COMPOSER_FALLBACK_HEIGHT).toBeLessThanOrEqual(
      MIN_WINDOW - SHELL_CHROME,
    );
  });

  it("lets the prompt grow only as far as the minimum window still fits Generate", () => {
    // The composer around the prompt (handle, padding, control row) plus the
    // tallest prompt this window allows plus the canvas floor must fit, or
    // dragging the handle up would push Generate off the bottom.
    const aroundPrompt = COMPOSER_FALLBACK_HEIGHT - PROMPT_HEIGHT_INITIAL;
    expect(PROMPT_HEIGHT_INITIAL).toBeLessThanOrEqual(availablePromptHeight(MIN_WINDOW));
    expect(
      MIN_CANVAS_HEIGHT + aroundPrompt + availablePromptHeight(MIN_WINDOW),
    ).toBeLessThanOrEqual(MIN_WINDOW - SHELL_CHROME);
  });

  it("keeps the retired scene bench's constants gone", async () => {
    // A clip has ONE way of being made: nothing sits between the canvas and
    // the composer, so there is no bench to clamp and no resizer to reserve.
    const layout = await import("./benchLayout");
    expect(Object.keys(layout).sort()).toEqual(["COMPOSER_FALLBACK_HEIGHT", "MIN_CANVAS_HEIGHT"]);
  });
});
