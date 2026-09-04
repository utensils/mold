import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * CSS Cascade 5: an UNLAYERED normal declaration beats every layered one,
 * whatever its specificity or source order. Tailwind puts every generated
 * utility inside `@layer utilities`, so while `ui/kit.css` was unlayered its
 * `.ms-toolbar-button` / `.ms-group-label` declarations silently won over any
 * utility a call site wrote beside them — `text-accent` on a paused queue
 * button, `h-[28px]` on a Lightbox button, `font-semibold` on the update
 * banner all rendered as no-ops. Wrapping the kit in `@layer components`
 * restores the order every call site already assumes.
 *
 * The layer lives in `ui/kit.css` itself rather than at an import site so the
 * one edit covers all three surfaces that consume it.
 */
const root = resolve(__dirname, "../../..");
const read = (relative: string) => readFileSync(resolve(root, relative), "utf8");

/** Every file that imports the kit. Each must sit downstream of Tailwind's
 * `@layer theme, base, components, utilities` declaration for the order to
 * mean anything, which `@import "tailwindcss"` is what provides. */
const IMPORTERS = [
  "desktop/src/styles/base.css",
  "web/src/style.css",
  "desktop/src/mobile/legacy.css",
];

describe("ui/kit.css cascade layer", () => {
  const kit = read("ui/kit.css");

  it("declares its rules inside @layer components", () => {
    expect(kit).toMatch(/@layer\s+components\s*\{/);
  });

  it("says why the layer is there, so nobody unlayers it again", () => {
    expect(kit).toMatch(/Cascade 5/);
  });

  it("puts every class rule inside the layer", () => {
    const opened = kit.indexOf("@layer components {");
    expect(opened).toBeGreaterThanOrEqual(0);
    const before = kit.slice(0, opened);
    expect(before).not.toMatch(/^\s*\./m);
  });

  it.each(IMPORTERS)("%s imports the kit, so the layer reaches it", (file) => {
    const css = read(file);
    expect(css).toMatch(/@import\s+"[^"]*ui\/kit\.css"/);
    expect(css).toMatch(/@import\s+"tailwindcss"|tokens\.css/);
  });

  it("keeps base.css's deliberately unlayered form-control rule unlayered", () => {
    // A control's resting hairline must beat `border-border`, which IS a
    // utility — that one rule is unlayered on purpose (see base.css).
    const base = read("desktop/src/styles/base.css");
    const rule = base.indexOf(":where(input, textarea, select, button).border-border");
    expect(rule).toBeGreaterThanOrEqual(0);
    expect(base.slice(0, rule)).not.toMatch(/@layer\s+\w+\s*\{/);
  });
});
