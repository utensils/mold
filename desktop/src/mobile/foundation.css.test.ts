import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

/*
 * `foundation.css` is the phone surface's whole global sheet — it stands in for
 * `styles/base.css`, which the phone deliberately does not import. Four rules
 * in that sheet are not decoration but behaviour, and dropping them turned the
 * app back into a web page: elastic rubber-banding at every scroll limit, a
 * long-press text selection over the chrome, WebKit's Picture-in-Picture
 * button on every video, and full-speed animation for someone who asked the
 * system for less motion.
 */
const css = readFileSync("src/mobile/foundation.css", "utf8");

describe("the phone's global rules", () => {
  it("uses the system Dynamic Type style only in the native iOS shell", () => {
    expect(css).toMatch(/html\.mobile-surface\.native-ios\s*\{[^}]*font: -apple-system-body;/s);
    const entry = readFileSync("src/mobile/main.ts", "utf8");
    expect(entry).toContain('classList.toggle("native-ios", isNativeIOSRuntime())');
  });

  it("honours the system's reduced-motion setting", () => {
    expect(css).toMatch(/@media \(prefers-reduced-motion: reduce\) \{/);
    expect(css).toMatch(/animation-duration: 0\.01ms !important;/);
    expect(css).toMatch(/transition-duration: 0\.01ms !important;/);
  });

  it("stops every scroll region dead at its limits, not only the document", () => {
    expect(css).toMatch(/\*\s*\{[^}]*overscroll-behavior: none;/s);
  });

  it("hides WebKit's Picture-in-Picture control", () => {
    expect(css).toMatch(
      /video::-webkit-media-controls-picture-in-picture-button \{[^}]*display: none;/s,
    );
  });

  it("makes the chrome unselectable and content selectable", () => {
    expect(css).toMatch(/body \{[^}]*user-select: none;/s);
    expect(css).toMatch(/\[data-selectable\]/);
    expect(css).toMatch(/\[data-selectable\][^{]*\{[^}]*user-select: text;/s);
  });
});
