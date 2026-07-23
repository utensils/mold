import { describe, expect, it } from "vitest";
import { modelDisplayName } from "./modelDisplay";

describe("modelDisplayName", () => {
  it("prefers the server-provided display name", () => {
    expect(
      modelDisplayName({
        name: "cv:23423432",
        display_name: "Juggernaut XL - Ragnarok",
        description: "Older fallback title",
      }),
    ).toBe("Juggernaut XL - Ragnarok");
  });

  it("uses the human-readable catalog description instead of a Civitai id", () => {
    expect(
      modelDisplayName({
        name: "cv:23423432",
        description: "RealVisXL V5.0 by SG161222",
      }),
    ).toBe("RealVisXL V5.0 by SG161222");
  });

  it("keeps stable built-in names and falls back when catalog metadata is missing", () => {
    expect(
      modelDisplayName({
        name: "flux-schnell:q8",
        description: "Fast FLUX generation",
      }),
    ).toBe("flux-schnell:q8");
    expect(modelDisplayName({ name: "cv:23423432", description: "  " })).toBe(
      "cv:23423432",
    );
  });
});
