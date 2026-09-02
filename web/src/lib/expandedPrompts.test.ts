import { describe, expect, it } from "vitest";
import { validateExpandedPrompts } from "./expandedPrompts";

describe("validateExpandedPrompts", () => {
  it("trims and returns exactly the requested count", () => {
    expect(validateExpandedPrompts([" one ", "two", "three"], 3)).toEqual([
      "one",
      "two",
      "three",
    ]);
  });

  it("rejects a short answer for a recipe that reads the prompt", () => {
    expect(() => validateExpandedPrompts(["only one"], 3)).toThrow(
      "Expected exactly 3 non-empty expanded prompts.",
    );
  });

  it("rejects an empty rewrite whatever the count", () => {
    expect(() => validateExpandedPrompts(["one", "   ", "three"], 3)).toThrow(
      "Expected exactly 3 non-empty expanded prompts.",
    );
  });

  // A prompt-ignoring recipe gets the guide's advice as ONE result, so the
  // batch is complete at one — but a two-of-three answer still is not.
  it("accepts the single advisory answer when the prompt is ignored", () => {
    expect(
      validateExpandedPrompts(["Prepare the image instead."], 3, {
        promptIgnored: true,
      }),
    ).toEqual(["Prepare the image instead."]);
    expect(() =>
      validateExpandedPrompts(["one", "two"], 3, { promptIgnored: true }),
    ).toThrow("Expected exactly 3 non-empty expanded prompts.");
  });
});
