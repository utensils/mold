import { describe, expect, it } from "vitest";
import source from "./EstimateBadge.vue?raw";

/*
 * Tailwind's colour utilities all carry specificity 0,1,0, so which one paints
 * is decided by EMITTED RULE ORDER, not by the order they appear in the class
 * attribute. `.text-fg-dim` is emitted after `.text-accent` and `.text-error`,
 * so a static `text-fg-dim` beside the bound map silently won both: a tight
 * VRAM fit and an outright "won't fit" both rendered dim grey, exactly like an
 * ordinary reading. The badge's colour must live in the bound map alone.
 */

function staticClasses(): string {
  return source.match(/\n\s+class="([^"]*)"/)?.[1] ?? "";
}

function boundClassMap(): string {
  return source.match(/:class="\{([\s\S]*?)\}"/)?.[1] ?? "";
}

describe("EstimateBadge colour", () => {
  it("carries no colour utility in the static class list", () => {
    const statics = staticClasses();
    expect(statics).not.toBe("");
    expect(statics).toContain("font-mono");
    expect(statics).toContain("text-micro");
    for (const colour of ["text-fg-dim", "text-accent", "text-error", "text-sapphire"]) {
      expect(statics).not.toContain(colour);
    }
  });

  it("maps every verdict, including the unavailable fallback, in :class", () => {
    const map = boundClassMap();
    expect(map).toContain("'text-sapphire'");
    expect(map).toContain("'text-accent'");
    expect(map).toContain("'text-error'");
    expect(map).toContain("'text-fg-dim'");
    expect(map).toMatch(/'text-accent':\s*fit === 'tight'/);
    expect(map).toMatch(/'text-error':\s*fit === 'wont-fit'/);
    expect(map).toMatch(/'text-fg-dim':\s*fit === 'unavailable'/);
  });
});
