import { describe, expect, it } from "vitest";
import { validateExpandedPrompts } from "./expandedPrompts";

/**
 * The one rule for whether an Expand answer is complete, shared by web and
 * desktop so the two surfaces cannot disagree about what a full batch is.
 */
describe("validateExpandedPrompts", () => {
  it("trims and returns exactly the requested count", () => {
    expect(validateExpandedPrompts([" one ", "two", "three"], 3)).toEqual([
      "one",
      "two",
      "three",
    ]);
  });

  it("rejects a short or long answer for a recipe that reads the prompt", () => {
    expect(() => validateExpandedPrompts(["only one"], 3)).toThrow(
      "Expected exactly 3 non-empty prompts, but the host returned 1.",
    );
    expect(() =>
      validateExpandedPrompts(["one", "two", "three", "four"], 3),
    ).toThrow("Expected exactly 3 non-empty prompts, but the host returned 4.");
  });

  it("names the empty rewrite it refuses", () => {
    expect(() => validateExpandedPrompts(["one", "   ", "three"], 3)).toThrow(
      "Prompt 2 was empty. Expected exactly 3 non-empty prompts.",
    );
  });

  // Some expanders answer each rewrite as a one-element JSON array; the
  // prompt inside is what the user asked for, not the brackets around it.
  it("unwraps a single-string JSON array answer", () => {
    expect(
      validateExpandedPrompts(['["a lighthouse at dusk"]', "a lantern"], 2),
    ).toEqual(["a lighthouse at dusk", "a lantern"]);
  });

  it("leaves ordinary text, longer arrays and non-string JSON alone", () => {
    expect(
      validateExpandedPrompts(['["one", "two"]', "[1]", "{not json"], 3),
    ).toEqual(['["one", "two"]', "[1]", "{not json"]);
  });

  it("refuses an unwrapped answer that turns out to be blank", () => {
    expect(() => validateExpandedPrompts(['["   "]'], 1)).toThrow(
      "Prompt 1 was empty. Expected exactly 1 non-empty prompts.",
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
    ).toThrow("Expected exactly 3 non-empty prompts, but the host returned 2.");
  });
});
