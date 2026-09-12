import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The quiet row's three buttons are not all buttons: "Download" / "Save" is an
 * `<a role="button">`, and an anchor is an INLINE box. As a flex item it still
 * stretched to the row's height, but its label sat on the first line — at the
 * top of a 44px row, beside two `<button>`s whose UA style centres their text.
 * `display: inline-flex` + `align-items: center` + `justify-content: center`
 * makes the anchor centre its own label the way a button does.
 *
 * `white-space: nowrap` with `flex: 1 1 auto` (basis = content) is what stops
 * "Use as source" breaking mid-phrase in a 3-up row, while `.lb__pair`'s
 * `flex-wrap: wrap` still breaks the ROW onto two lines on a phone.
 * `overflow-wrap: anywhere` was the opposite instruction and has to go.
 *
 * All of that is scoped to `.lb__pair > .lb__quiet`, the ROW. The bare class
 * is also worn by two direct children of `.lb__panel`, which is a COLUMN, and
 * a `flex-basis` there is a HEIGHT — those keep the `1 1 5rem` they have
 * always had.
 */
const root = resolve(__dirname, "../../../..");
const source = readFileSync(
  resolve(root, "web/src/components/gallery/Lightbox.vue"),
  "utf8",
);

/** The declarations of one top-level rule, by exact selector. */
function ruleBody(selector: string): string {
  const marker = `\n${selector} {`;
  const start = source.indexOf(marker);
  expect(start, `no \`${selector} {\` rule in Lightbox.vue`).toBeGreaterThan(0);
  const open = start + marker.length;
  const close = source.indexOf("\n}", open);
  expect(close).toBeGreaterThan(open);
  return source.slice(open, close);
}

describe("Lightbox .lb__quiet", () => {
  const quiet = ruleBody(".lb__quiet");
  const inRow = ruleBody(".lb__pair > .lb__quiet");

  it("centres its label on both axes, so the anchor matches the buttons", () => {
    expect(inRow).toMatch(/display:\s*inline-flex\s*;/);
    expect(inRow).toMatch(/align-items:\s*center\s*;/);
    expect(inRow).toMatch(/justify-content:\s*center\s*;/);
  });

  it("keeps each label on one line and sizes from its content", () => {
    expect(inRow).toMatch(/white-space:\s*nowrap\s*;/);
    expect(inRow).toMatch(/flex:\s*1\s+1\s+auto\s*;/);
  });

  it("truncates a label too wide for its line instead of spilling", () => {
    expect(inRow).toMatch(/overflow:\s*hidden\s*;/);
    expect(inRow).toMatch(/text-overflow:\s*ellipsis\s*;/);
  });

  it("no longer tells the label to break anywhere", () => {
    expect(quiet).not.toMatch(/overflow-wrap/);
    expect(inRow).not.toMatch(/overflow-wrap/);
  });

  it("keeps the row's touch target and its zero minimum width", () => {
    expect(quiet).toMatch(/min-height:\s*44px\s*;/);
    expect(quiet).toMatch(/min-width:\s*0\s*;/);
  });

  // `.lb__export` and "Delete forever" are direct children of the COLUMN
  // `.lb__panel`, where a basis is a height. They keep the one they had.
  it("leaves the column-direction buttons on their original basis", () => {
    expect(quiet).toMatch(/flex:\s*1\s+1\s+5rem\s*;/);
  });

  it("still lets the pair wrap on a narrow screen", () => {
    expect(ruleBody(".lb__pair")).toMatch(/flex-wrap:\s*wrap\s*;/);
  });
});
