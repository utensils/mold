import { describe, expect, it } from "vitest";
import { suggestedSaveName } from "./saveName";

describe("suggestedSaveName", () => {
  it("keeps the gallery filename for an untitled print", () => {
    expect(suggestedSaveName({ filename: "mold-flux-dev-1700000000.png" })).toBe(
      "mold-flux-dev-1700000000.png",
    );
    expect(suggestedSaveName({ filename: "mold-flux-dev-1700000000.png", title: "   " })).toBe(
      "mold-flux-dev-1700000000.png",
    );
    expect(suggestedSaveName({ filename: "mold-flux-dev-1700000000.png", title: null })).toBe(
      "mold-flux-dev-1700000000.png",
    );
  });

  it("uses the title slug with the print's own extension", () => {
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        title: "Smurf Village at Dusk",
      }),
    ).toBe("smurf-village-at-dusk.png");
    expect(suggestedSaveName({ filename: "mold-ltx2-1700000000.mp4", title: "Opening shot" })).toBe(
      "opening-shot.mp4",
    );
  });

  it("falls back to the gallery filename when nothing of the title survives slugging", () => {
    expect(suggestedSaveName({ filename: "mold-flux-dev-1700000000.png", title: "日本語" })).toBe(
      "mold-flux-dev-1700000000.png",
    );
  });

  it("applies an optional suffix before the extension and an explicit extension override", () => {
    expect(
      suggestedSaveName(
        { filename: "mold-flux-dev-1700000000.png", title: "Smurf village" },
        { suffix: "-upscaled" },
      ),
    ).toBe("smurf-village-upscaled.png");
    expect(
      suggestedSaveName(
        { filename: "mold-ltx2-1700000000.mp4", title: "Opening shot" },
        { extension: "webm" },
      ),
    ).toBe("opening-shot.webm");
    // Untitled + suffix keeps today's `${stem}${suffix}.${ext}` shape.
    expect(
      suggestedSaveName({ filename: "mold-flux-dev-1700000000.png" }, { suffix: "-upscaled" }),
    ).toBe("mold-flux-dev-1700000000-upscaled.png");
  });

  it("handles filenames without an extension", () => {
    expect(suggestedSaveName({ filename: "mystery", title: "Named" })).toBe("named");
    expect(suggestedSaveName({ filename: "mystery" })).toBe("mystery");
  });

  it("reads the creation-time metadata title when the row carries none", () => {
    expect(
      suggestedSaveName({
        filename: "mold-flux-dev-1700000000.png",
        metadata: { title: "From metadata" },
      }),
    ).toBe("from-metadata.png");
  });
});
