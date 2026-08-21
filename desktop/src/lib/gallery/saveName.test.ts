import { describe, expect, it } from "vitest";
import { suggestedSaveName } from "./saveName";

const META = { model: "flux-dev:q8", seed: 7 };

describe("suggestedSaveName", () => {
  it("labels an untitled print with its model and seed", () => {
    expect(suggestedSaveName({ filename: "mold-flux-dev-1700000000.png", metadata: META })).toBe(
      "flux-dev-q8__s7.png",
    );
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        title: "   ",
        metadata: META,
      }),
    ).toBe("flux-dev-q8__s7.png");
  });

  it("leads with the title slug", () => {
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        title: "Smurf Village at Dusk",
        metadata: META,
      }),
    ).toBe("smurf-village-at-dusk__flux-dev-q8__s7.png");
    expect(
      suggestedSaveName({
        filename: "mold-ltx2-1700000000.mp4",
        title: "Opening shot",
        metadata: { model: "ltx-2-19b-distilled:fp8", seed: 42 },
      }),
    ).toBe("opening-shot__ltx-2-19b-distilled-fp8__s42.mp4");
  });

  it("drops a title that slugs to nothing rather than writing it", () => {
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        title: "日本語",
        metadata: META,
      }),
    ).toBe("flux-dev-q8__s7.png");
  });

  it("keeps the gallery filename when the row carries no usable provenance", () => {
    expect(suggestedSaveName({ filename: "mold-flux-dev-1700000000.png" })).toBe(
      "mold-flux-dev-1700000000.png",
    );
    expect(
      suggestedSaveName({ filename: "mold-flux-dev-1700000000.png", metadata: { seed: null } }),
    ).toBe("mold-flux-dev-1700000000.png");
  });

  it("keeps a full-range u64 seed exactly", () => {
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        metadata: { model: "flux-dev", seed: "18446744073709551615" },
      }),
    ).toBe("flux-dev__s18446744073709551615.png");
  });

  it("applies an optional suffix before the extension and an explicit extension override", () => {
    expect(
      suggestedSaveName(
        { filename: "mold-flux-dev-1700000000.png", title: "Smurf village", metadata: META },
        { suffix: "-upscaled" },
      ),
    ).toBe("smurf-village__flux-dev-q8__s7-upscaled.png");
    expect(
      suggestedSaveName(
        {
          filename: "mold-ltx2-1700000000.mp4",
          title: "Opening shot",
          metadata: { model: "ltx2", seed: 1 },
        },
        { extension: "webm" },
      ),
    ).toBe("opening-shot__ltx2__s1.webm");
    // A provenance-less row keeps today's `${stem}${suffix}.${ext}` shape.
    expect(
      suggestedSaveName({ filename: "mold-flux-dev-1700000000.png" }, { suffix: "-upscaled" }),
    ).toBe("mold-flux-dev-1700000000-upscaled.png");
  });

  it("handles filenames without an extension", () => {
    expect(suggestedSaveName({ filename: "mystery", title: "Named", metadata: META })).toBe(
      "named__flux-dev-q8__s7",
    );
    expect(suggestedSaveName({ filename: "mystery" })).toBe("mystery");
  });

  it("reads the creation-time metadata title when the row carries none", () => {
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        metadata: { ...META, title: "From metadata" },
      }),
    ).toBe("from-metadata__flux-dev-q8__s7.png");
  });
});
