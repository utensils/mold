import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

const css = readFileSync("src/styles/base.css", "utf8");

describe("desktop document behavior", () => {
  it("disables WebKit elastic overscroll across the app shell", () => {
    expect(css).toMatch(/html,\s*body,\s*#app\s*{[^}]*overscroll-behavior:\s*none;/s);
  });
});
