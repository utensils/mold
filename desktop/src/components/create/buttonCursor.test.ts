import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import createHeaderSource from "./CreateHeader.vue?raw";
import inspectorSource from "./InspectorPanel.vue?raw";
import modelPickerSource from "./ModelPicker.vue?raw";
import sequenceComposerSource from "./SequenceComposer.vue?raw";
import viewSource from "../../views/GenerateView.vue?raw";

/*
 * Tailwind v4's preflight sets `appearance: button` on <button> and nothing
 * else — there is no `button { cursor: pointer }` anywhere in the emitted
 * stylesheet. So a button whose own rule omits it shows the ARROW cursor, and
 * on this view that was almost every control: Add a scene, File tools, Check
 * the plan, Clear the clip, Write more for me, both header doors, all four
 * caption actions, the seed reroll and lock, and the whole style picker.
 */

// Vitest stubs `?raw` on a .css import (its own CSS handling is off), so the
// shared kit sheet is read from disk the way tokens.legacy.test.ts reads it.
const kitSource = readFileSync("../ui/kit.css", "utf8");

/** The declarations inside `selector { … }` in a stylesheet or SFC. */
function block(source: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = source.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`));
  return match?.[1] ?? "";
}

const CLICKABLE: ReadonlyArray<readonly [string, string, string]> = [
  ["ui/kit.css", ".ms-toolbar-button", kitSource],
  ["CreateHeader.vue", ".ms-header__door", createHeaderSource],
  ["InspectorPanel.vue", ".ms-seed__reroll", inspectorSource],
  ["InspectorPanel.vue", ".ms-seed__lock", inspectorSource],
  ["ModelPicker.vue", ".ms-model__button", modelPickerSource],
  ["ModelPicker.vue", ".ms-model__option", modelPickerSource],
  ["ModelPicker.vue", ".ms-model__browse", modelPickerSource],
  ["GenerateView.vue", ".caption-action", viewSource],
];

describe("clickable controls show the hand cursor", () => {
  it("recognises a rule that omits it (positive control)", () => {
    expect(block(".ms-thing {\n  height: 26px;\n}", ".ms-thing")).not.toContain("cursor: pointer");
    expect(block(".ms-thing {\n  cursor: pointer;\n}", ".ms-thing")).toContain("cursor: pointer");
  });

  it.each(CLICKABLE)("%s %s declares cursor: pointer", (_file, selector, source) => {
    const declarations = block(source, selector);
    expect(declarations).not.toBe("");
    expect(declarations).toContain("cursor: pointer");
  });

  it("gives the timeline's bare-Tailwind help button the utility", () => {
    const tag = sequenceComposerSource.match(/<[^>]*data-test="timeline-help"[^>]*>/s)?.[0] ?? "";
    expect(tag).toContain("cursor-pointer");
  });
});
