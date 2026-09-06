import { describe, expect, it } from "vitest";
import {
  modelsForOutputKind,
  outputKindFor,
  outputKindForModel,
  OUTPUT_KIND_BROWSE_TARGET,
  OUTPUT_KIND_LABEL,
  OUTPUT_KIND_EMPTY,
  OUTPUT_KIND_SECTION_LABEL,
  type OutputKind,
} from "./useCreateOutputKind";
import { mediaTypeFromQuery } from "../lib/modelAvailability";

/*
 * The New image view has three sections — Still picture | Short clip | 3-D
 * object — and this module is the ONE place that answers which styles belong
 * to each. The picker, the header's 3-D door and the inspector's clip swap all
 * read it, so a style can never be offered under a section that cannot make it.
 */

function style(family: string, extra: Record<string, unknown> = {}) {
  return { name: `${family}-model`, family, ...extra };
}

describe("outputKindForModel — every style has exactly one section", () => {
  const table: [string, OutputKind][] = [
    ["flux", "still"],
    ["sdxl", "still"],
    ["sd15", "still"],
    ["z-image", "still"],
    ["qwen-image", "still"],
    ["qwen-image-edit", "still"],
    ["flux2", "still"],
    // Every video family in `studio/lib/generationCapabilities.ts`.
    ["ltx-video", "clip"],
    ["ltx2", "clip"],
    ["ltx-2", "clip"],
    ["wan", "clip"],
    ["minimax-h3", "clip"],
    ["minimax_h3", "clip"],
    ["minimaxh3", "clip"],
    ["hunyuan3d", "mesh"],
    // A family this build has never heard of is a picture style, which is
    // what every non-video, non-mesh checkpoint has always been.
    ["some-future-family", "still"],
  ];

  for (const [family, kind] of table) {
    it(`puts ${family} in ${kind}`, () => {
      expect(outputKindForModel(style(family))).toBe(kind);
    });
  }

  it("reads the family case- and whitespace-insensitively", () => {
    expect(outputKindForModel(style(" Hunyuan3D "))).toBe("mesh");
    expect(outputKindForModel(style("LTX-2"))).toBe("clip");
  });
});

describe("modelsForOutputKind", () => {
  const flux = style("flux");
  const ltx = style("ltx-video", { supports_sequence: true });
  const h3 = style("minimax-h3", { supports_sequence: false });
  const mesh = style("hunyuan3d");
  const all = [flux, ltx, h3, mesh];

  it("keeps only picture styles under Still picture", () => {
    expect(modelsForOutputKind(all, "still")).toEqual([flux]);
  });

  it("keeps every clip style under Short clip, chain-capable or not", () => {
    // A one-shot-only clip style (H3 advertises `supports_sequence: false`)
    // still MAKES a clip, so this is where a person looks for it. Whether it
    // can author a multi-scene sequence is a separate refusal the picker
    // spells out on the row itself.
    expect(modelsForOutputKind(all, "clip")).toEqual([ltx, h3]);
  });

  it("keeps only 3-D styles under 3-D object", () => {
    expect(modelsForOutputKind(all, "mesh")).toEqual([mesh]);
  });

  it("partitions the list — no style is unreachable", () => {
    const sections: OutputKind[] = ["still", "clip", "mesh"];
    const seen = sections.flatMap((kind) => modelsForOutputKind(all, kind));
    expect(seen).toHaveLength(all.length);
    expect(new Set(seen)).toEqual(new Set(all));
  });

  it("preserves the order it was handed", () => {
    expect(modelsForOutputKind([h3, ltx], "clip")).toEqual([h3, ltx]);
  });
});

describe("what each section says about itself", () => {
  /*
   * ONE set of words for the three kinds, wherever a kind is chosen or
   * filtered: the Create toolbar's control and the Styles view's kind filter
   * both read it, so a person learns the mapping once. The Styles filter used
   * to say Pictures · Clips against Still picture | Short clip, and had no
   * 3-D kind at all.
   */
  it("names the three kinds once, in the Create toolbar's words", () => {
    expect(OUTPUT_KIND_LABEL).toEqual({
      still: "Still picture",
      clip: "Short clip",
      mesh: "3-D object",
    });
  });

  it("names the section in the menu's kicker from the same words, the lexicon word being style", () => {
    expect(OUTPUT_KIND_SECTION_LABEL.still).toBe("still picture styles");
    expect(OUTPUT_KIND_SECTION_LABEL.clip).toBe("short clip styles");
    expect(OUTPUT_KIND_SECTION_LABEL.mesh).toBe("3-D object styles");
    for (const label of Object.values(OUTPUT_KIND_SECTION_LABEL)) {
      expect(label).not.toMatch(/model/i);
    }
  });

  it("names the section when it holds nothing at all", () => {
    expect(OUTPUT_KIND_EMPTY.still).toContain("still picture");
    expect(OUTPUT_KIND_EMPTY.clip).toContain("clip");
    expect(OUTPUT_KIND_EMPTY.mesh).toContain("3-D");
    for (const sentence of Object.values(OUTPUT_KIND_EMPTY)) {
      expect(sentence).not.toMatch(/model/i);
    }
  });

  it("deep-links Browse more to the Styles view's own kind filter", () => {
    // `?type=` and its values are `mediaTypeFromQuery`'s — never invented here.
    expect(OUTPUT_KIND_BROWSE_TARGET.still).toBe("/models?type=image");
    expect(OUTPUT_KIND_BROWSE_TARGET.clip).toBe("/models?type=video");
    expect(OUTPUT_KIND_BROWSE_TARGET.mesh).toBe("/models?type=mesh");
    for (const target of Object.values(OUTPUT_KIND_BROWSE_TARGET)) {
      const type = new URL(target, "http://x").searchParams.get("type");
      expect(mediaTypeFromQuery({ type })).toBe(type);
    }
  });
});

describe("outputKindFor — the view's own kind", () => {
  it("reads the kind from the style's family", () => {
    expect(outputKindFor("hunyuan3d")).toBe("mesh");
    expect(outputKindFor("flux")).toBe("still");
  });

  /*
   * A clip has ONE way of being made, so the STYLE is the whole decision: a
   * render on a clip style shows Short clip in the toolbar and New clip in
   * the title bar. Nothing outranks the family any more.
   */
  it("calls a render on a clip style a clip", () => {
    expect(outputKindFor("ltx-video")).toBe("clip");
    expect(outputKindFor("ltx2")).toBe("clip");
    expect(outputKindFor("wan")).toBe("clip");
    expect(outputKindFor("minimax-h3")).toBe("clip");
  });

  it("answers the same question the picker's partition does", () => {
    for (const family of ["flux", "ltx-video", "hunyuan3d", "some-future-family"]) {
      expect(outputKindFor(family), family).toBe(outputKindForModel(style(family)));
    }
  });

  it("reads an empty or absent family as a still picture", () => {
    expect(outputKindFor("")).toBe("still");
    expect(outputKindFor(null)).toBe("still");
    expect(outputKindFor(undefined)).toBe("still");
  });
});
