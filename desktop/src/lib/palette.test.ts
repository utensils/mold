import { describe, expect, it } from "vitest";
import { matchCommands, scoreText, type Matchable } from "./palette";

describe("scoreText", () => {
  it("ranks prefix over word-start over substring", () => {
    expect(scoreText("gen", "Generate")).toBe(3); // prefix
    expect(scoreText("set", "Go to Settings")).toBe(2); // word-start
    expect(scoreText("era", "Generate")).toBe(1); // substring
    expect(scoreText("zzz", "Generate")).toBe(0); // none
  });

  it("treats separators as word boundaries", () => {
    expect(scoreText("q8", "flux-dev:q8")).toBe(2);
    expect(scoreText("dev", "flux-dev:q8")).toBe(2);
  });
});

describe("matchCommands", () => {
  const items: Matchable[] = [
    { title: "Go to Generate" },
    { title: "Go to Gallery" },
    { title: "Randomize seed" },
    { title: "Generate with flux-dev:q8", keywords: ["flux", "model"] },
  ];

  it("returns all items in registry order for an empty query", () => {
    expect(matchCommands("", items)).toEqual(items);
    expect(matchCommands("   ", items)).toEqual(items);
  });

  it("drops non-matches and orders by score then registry position", () => {
    const out = matchCommands("gen", items).map((i) => i.title);
    // "Generate with..." — "gen" is word-start of "Generate" (score 2);
    // "Go to Generate"/"Go to Gallery" — word-start of "Generate"/"Gallery" (2);
    // "Randomize seed" — no match, dropped.
    expect(out).not.toContain("Randomize seed");
    expect(out).toContain("Go to Generate");
    expect(out).toContain("Generate with flux-dev:q8");
  });

  it("prefers a prefix hit over a mid-word hit", () => {
    const out = matchCommands("rand", items).map((i) => i.title);
    expect(out).toEqual(["Randomize seed"]);
  });

  it("matches on keywords too", () => {
    const out = matchCommands("flux", items).map((i) => i.title);
    expect(out).toEqual(["Generate with flux-dev:q8"]);
  });

  it("keeps stable order for equal scores", () => {
    const out = matchCommands("go to", items).map((i) => i.title);
    expect(out).toEqual(["Go to Generate", "Go to Gallery"]);
  });
});
