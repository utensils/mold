import { describe, expect, it } from "vitest";
import { PromptClearRecovery } from "./promptClearRecovery";
import { applyAuthoredPrompt } from "./promptProvenance";
describe("prompt clear recovery", () => {
  for (const active of [false, true]) {
    it(`restores immediate provenance with active transform ${active}`, () => {
      const draft = {
        prompt: "expanded words",
        originalPrompt: "root words" as string | null,
      };
      const recovery = new PromptClearRecovery();
      expect(recovery.apply(draft, "", "clear")).toBe(true);
      expect(draft).toEqual({ prompt: "", originalPrompt: null });
      expect(recovery.apply(draft, "expanded words", "undo-clear")).toBe(true);
      expect(draft.originalPrompt).toBe("root words");
      recovery.apply(draft, "", "clear");
      expect(recovery.apply(draft, "fresh", "typed")).toBe(false);
      applyAuthoredPrompt(draft, "fresh", active);
      expect(recovery.apply(draft, "expanded words", "undo-clear")).toBe(false);
    });
  }
  it("separate composers cannot restore another draft's provenance", () => {
    const first = new PromptClearRecovery(),
      second = new PromptClearRecovery();
    const draft = { prompt: "one", originalPrompt: "root" };
    first.apply(draft, "", "clear");
    expect(second.apply(draft, "one", "undo-clear")).toBe(false);
    expect(draft.originalPrompt).toBe(null);
  });
});
