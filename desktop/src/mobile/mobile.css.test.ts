import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/mobile/mobile.css", "utf8");

describe("mobile editable controls", () => {
  it("keeps every editable control at the iOS no-focus-zoom size", () => {
    const editables = css.match(
      /input,\s*textarea,\s*select,\s*\[contenteditable="true"\]\s*\{([^}]*)\}/s,
    );

    expect(editables?.[1]).toMatch(/font-size:\s*16px\s*;/);
  });
});
