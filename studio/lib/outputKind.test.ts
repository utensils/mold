import { describe, expect, it } from "vitest";
import {
  modelsForOutputKind,
  outputKindFor,
  outputKindForModel,
  OUTPUT_KIND_BROWSE_TARGET,
  OUTPUT_KIND_EMPTY,
  OUTPUT_KIND_MISSING,
  OUTPUT_KIND_LABEL,
  OUTPUT_KIND_PLACEHOLDER,
  OUTPUT_KIND_SECTION_LABEL,
  OUTPUT_KIND_TITLE,
  type OutputKind,
} from "./outputKind";

/*
 * The ONE authority for which section a style belongs to, and for the words
 * each section is named with. It had no direct test: every assertion on it
 * came through a desktop view, so a family sorted into the wrong section only
 * showed up as a picker rendering the wrong rows.
 */

const KINDS: OutputKind[] = ["still", "clip", "mesh"];

describe("outputKindForModel", () => {
  it.each([
    ["flux", "still"],
    ["sdxl", "still"],
    ["sd15", "still"],
    ["z-image", "still"],
    ["qwen-image-edit", "still"],
    ["flux2", "still"],
    ["ltx2", "clip"],
    ["ltx-video", "clip"],
    ["wan", "clip"],
    ["minimax-h3", "clip"],
    ["hunyuan3d", "mesh"],
  ])("sorts %s into %s", (family, kind) => {
    expect(outputKindForModel({ family })).toBe(kind);
    expect(outputKindFor(family)).toBe(kind);
  });

  /* A family nobody has taught it yet must still land SOMEWHERE, or its
   * styles become unreachable in every section at once. */
  it("lands an unknown family in the still section rather than nowhere", () => {
    expect(outputKindForModel({ family: "a-family-from-the-future" })).toBe(
      "still",
    );
    expect(outputKindFor(null)).toBe("still");
    expect(outputKindFor(undefined)).toBe("still");
    expect(outputKindFor("")).toBe("still");
  });
});

describe("modelsForOutputKind", () => {
  const models = [
    { name: "flux-dev:q8", family: "flux" },
    { name: "wan22-ti2v-5b:fp16", family: "wan" },
    { name: "hunyuan3d-2.1:fp16", family: "hunyuan3d" },
    { name: "z-image:bf16", family: "z-image" },
  ];

  it("is a PARTITION: every style lands in exactly one section", () => {
    const sections = KINDS.map((kind) => modelsForOutputKind(models, kind));
    expect(
      sections
        .flat()
        .map((m) => m.name)
        .sort(),
    ).toEqual(models.map((m) => m.name).sort());
  });

  it("keeps the order the caller handed in", () => {
    expect(modelsForOutputKind(models, "still").map((m) => m.name)).toEqual([
      "flux-dev:q8",
      "z-image:bf16",
    ]);
  });
});

describe("the section's words", () => {
  it("names all three kinds in every table", () => {
    for (const table of [
      OUTPUT_KIND_LABEL,
      OUTPUT_KIND_TITLE,
      OUTPUT_KIND_PLACEHOLDER,
      OUTPUT_KIND_SECTION_LABEL,
      OUTPUT_KIND_EMPTY,
      OUTPUT_KIND_MISSING,
      OUTPUT_KIND_BROWSE_TARGET,
    ]) {
      expect(Object.keys(table).sort()).toEqual([...KINDS].sort());
    }
  });

  it("keeps the binding lexicon", () => {
    expect(OUTPUT_KIND_LABEL).toEqual({
      still: "Still picture",
      clip: "Short clip",
      mesh: "3-D object",
    });
    expect(OUTPUT_KIND_TITLE.mesh).toBe("New 3-D object");
    expect(OUTPUT_KIND_SECTION_LABEL.still).toBe("still picture styles");
    // "3-D" keeps its capitals; the other two are lowercased in the kicker.
    expect(OUTPUT_KIND_SECTION_LABEL.mesh).toBe("3-D object styles");
    expect(OUTPUT_KIND_EMPTY.clip).toBe(
      "No short clip styles on this machine.",
    );
  });

  /* The sentence a door says when the user asked for a kind nothing on any
   * reachable machine can make. It is the same three words' table, so the
   * door and the empty section cannot drift apart. */
  it("tells a door with nothing to open onto what to get", () => {
    expect(OUTPUT_KIND_MISSING.clip).toBe(
      "Get a short clip style ready on a machine first.",
    );
    expect(OUTPUT_KIND_MISSING.still).toBe(
      "Get a still picture style ready on a machine first.",
    );
    expect(OUTPUT_KIND_MISSING.mesh).toBe(
      "Get a 3-D object style ready on a machine first.",
    );
  });

  /* Browse more must land on the Styles filter that holds this section's
   * rows, using `mediaTypeFromQuery`'s own values and no others. */
  it("sends Browse more to the same kind's Styles filter", () => {
    expect(OUTPUT_KIND_BROWSE_TARGET).toEqual({
      still: "/models?type=image",
      clip: "/models?type=video",
      mesh: "/models?type=mesh",
    });
  });
});
