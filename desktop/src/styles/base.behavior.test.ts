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

  it("hides WebKit's Picture-in-Picture control for desktop videos", () => {
    expect(css).toContain("video::-webkit-media-controls-picture-in-picture-button");
    expect(css).toMatch(
      /video::\-webkit-media-controls-picture-in-picture-button\s*\{[^}]*display:\s*none/s,
    );
  });
});
