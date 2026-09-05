import { describe, expect, it } from "vitest";
import inspectorSource from "./InspectorPanel.vue?raw";
import sequenceComposerSource from "./SequenceComposer.vue?raw";
import loraStackSource from "../generate/LoraStack.vue?raw";

/*
 * `.ms-group-label` deliberately does NOT set `text-transform` — the kit
 * leaves the case to the caller, because a header that reads as a sentence
 * keeps its case. Every group header on the New-image inspector is a LABEL,
 * and the mock uppercases all of them; with only two call sites passing
 * `uppercase` the Settings tab showed one uppercased header and five that
 * were not, in the same mono, tracked, dim treatment.
 */

const SURFACES: ReadonlyArray<readonly [string, string]> = [
  ["InspectorPanel.vue", inspectorSource],
  ["SequenceComposer.vue", sequenceComposerSource],
  ["LoraStack.vue", loraStackSource],
];

/** Every `class="…"` attribute that names `ms-group-label`. */
function groupLabelClasses(source: string): string[] {
  return [...source.matchAll(/class="([^"]*\bms-group-label\b[^"]*)"/g)].map((m) => m[1]!);
}

describe("inspector group labels", () => {
  it("finds the labels it is guarding (positive control)", () => {
    expect(groupLabelClasses('<div class="ms-group-label">Quality</div>')).toEqual([
      "ms-group-label",
    ]);
    expect(groupLabelClasses("<div>Quality</div>")).toEqual([]);
  });

  it.each(SURFACES)("%s uppercases every group label", (_name, source) => {
    const classes = groupLabelClasses(source);
    expect(classes.length).toBeGreaterThan(0);
    expect(classes.filter((attr) => !attr.includes("uppercase"))).toEqual([]);
  });

  it("keeps the case decision with the caller, not the kit", () => {
    // A kit-level text-transform would uppercase table headers and sentences
    // on every other surface too.
    expect(inspectorSource).not.toContain("text-transform: uppercase");
  });
});
