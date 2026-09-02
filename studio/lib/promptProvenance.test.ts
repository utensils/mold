import { describe, expect, it } from "vitest";
import {
  applyAuthoredPrompt,
  quickTransformSurvivesAuthoring,
} from "./promptProvenance";

describe("applyAuthoredPrompt", () => {
  it("retires provenance that no longer has active quick-transform authority", () => {
    const draft = {
      prompt: "expanded lighthouse",
      originalPrompt: "a lighthouse",
    };

    applyAuthoredPrompt(draft, "a completely new print", false);

    expect(draft).toEqual({
      prompt: "a completely new print",
      originalPrompt: null,
    });
  });

  it("preserves provenance while quick-transform stale recovery still owns it", () => {
    const draft = {
      prompt: "expanded lighthouse",
      originalPrompt: "a lighthouse",
    };

    applyAuthoredPrompt(draft, "an edited expanded lighthouse", true);

    expect(draft).toEqual({
      prompt: "an edited expanded lighthouse",
      originalPrompt: "a lighthouse",
    });
  });

  it("treats an untagged authoring event as typing", () => {
    const draft = {
      prompt: "expanded lighthouse",
      originalPrompt: "a lighthouse",
    };
    applyAuthoredPrompt(draft, "edited", true, "typed");
    expect(draft.originalPrompt).toBe("a lighthouse");
  });

  it("retires provenance on a history recall even while a quick transform is active", () => {
    const draft = {
      prompt: "expanded lighthouse",
      originalPrompt: "a lighthouse",
    };

    applyAuthoredPrompt(draft, "yesterday's harbour print", true, "recalled");

    expect(draft).toEqual({
      prompt: "yesterday's harbour print",
      originalPrompt: null,
    });
  });
});

describe("quickTransformSurvivesAuthoring", () => {
  it("keeps the transform across a hand edit so stale recovery can still offer restore", () => {
    expect(quickTransformSurvivesAuthoring("typed")).toBe(true);
  });

  it("releases the transform on a ↑/↓ history recall", () => {
    expect(quickTransformSurvivesAuthoring("recalled")).toBe(false);
  });
});
