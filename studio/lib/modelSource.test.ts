import { describe, expect, it } from "vitest";
import { modelSource, wireSourceGlyph } from "./modelSource";

describe("modelSource", () => {
  it("maps catalog prefixes and hf_repo to their source", () => {
    expect(modelSource({ name: "cv:2319074" })).toBe("civitai");
    expect(modelSource({ name: "hf:Qwen/Qwen-Image" })).toBe("hf");
    expect(
      modelSource({
        name: "flux-dev:q8",
        hf_repo: "black-forest-labs/FLUX.1-dev",
      }),
    ).toBe("hf");
    expect(modelSource({ name: "my-merge", hf_repo: null })).toBe("local");
    expect(modelSource({ name: "my-merge" })).toBe("local");
  });
});

describe("wireSourceGlyph", () => {
  /*
   * `CatalogEntryWire.source` is typed `"hf" | "civitai"` on the wire, but
   * `CatalogEntry` (the desktop/mobile catalog union, which also carries an
   * installed row already classified through `modelSource`) types the same
   * field as a plain `string` — so a caller trusting the narrow type without
   * a runtime guard has nothing stopping a value it did not expect. This is
   * the ONE guard, used wherever a wire/union `source` string becomes a
   * glyph, so a divergence between the two callers is not a copy/paste risk.
   */
  it("passes through the two known wire values", () => {
    expect(wireSourceGlyph("hf")).toBe("hf");
    expect(wireSourceGlyph("civitai")).toBe("civitai");
  });

  it("falls back to local for anything else, including an installed row's own local", () => {
    expect(wireSourceGlyph("local")).toBe("local");
    expect(wireSourceGlyph("")).toBe("local");
    expect(wireSourceGlyph("unknown-future-source")).toBe("local");
  });
});
