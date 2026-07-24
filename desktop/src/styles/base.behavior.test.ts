import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/styles/base.css", "utf8");

describe("desktop document behavior", () => {
  it("disables elastic overscroll without trapping wheel input in WebKitGTK", () => {
    expect(css).toMatch(
      /:root:not\(\[data-platform="linux"\]\)[^{]*{[^}]*overscroll-behavior:\s*none;/s,
    );
    expect(css).not.toMatch(/(?:^|\n)\s*\*\s*{\s*overscroll-behavior:\s*none;/s);
  });
});
