import { describe, expect, it } from "vitest";
import { applyAuthoredPrompt } from "./promptProvenance";

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
});
