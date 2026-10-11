import { beforeEach, describe, expect, it } from "vitest";
import {
  PROMPT_HEIGHT_INITIAL,
  PROMPT_HEIGHT_STEP,
  availablePromptHeight,
  draggedPromptHeight,
  loadPromptHeight,
  resolvePromptHeight,
  savePromptHeight,
} from "./promptEditorHeight";
import { MIN_CANVAS_HEIGHT } from "./benchLayout";

// The same policy the native Mac composer pins (PromptEditorHeight.swift):
// the prompt grows UPWARD from a bottom-anchored composer, and a short window
// clamps what is drawn without discarding the saved preference.
describe("prompt editor height", () => {
  beforeEach(() => localStorage.clear());

  it("grows upward and never exceeds the room it is given", () => {
    expect(draggedPromptHeight(72, -100, 400)).toBe(172);
    expect(draggedPromptHeight(72, 100, 400)).toBe(48);
    expect(resolvePromptHeight(560, 120)).toBe(120);
    expect(resolvePromptHeight(560, 24)).toBe(24);
    expect(resolvePromptHeight(560, -1)).toBe(0);
    expect(resolvePromptHeight(Number.NaN, 400)).toBe(PROMPT_HEIGHT_INITIAL);
    expect(resolvePromptHeight(2000, 1000)).toBe(560);
    expect(PROMPT_HEIGHT_STEP).toBe(40);
  });

  it("leaves the canvas its floor", () => {
    expect(availablePromptHeight(1000)).toBeLessThanOrEqual(1000 - MIN_CANVAS_HEIGHT);
    expect(availablePromptHeight(300)).toBe(0);
  });

  it("remembers the preference and survives a bad stored value", () => {
    expect(loadPromptHeight()).toBe(PROMPT_HEIGHT_INITIAL);
    savePromptHeight(200);
    expect(loadPromptHeight()).toBe(200);
    localStorage.setItem("mold.desktop.promptEditorHeight.v1", "tall");
    expect(loadPromptHeight()).toBe(PROMPT_HEIGHT_INITIAL);
  });
});
