import { describe, expect, it } from "vitest";
import {
  isCatalogModelId,
  modelDisplayName,
  modelDisplayNameForId,
} from "./modelDisplay";

describe("isCatalogModelId", () => {
  it("recognizes only opaque catalog identifiers", () => {
    expect(isCatalogModelId("cv:252914")).toBe(true);
    expect(isCatalogModelId("hf:org/repo")).toBe(true);
    expect(isCatalogModelId("flux-dev:q8")).toBe(false);
  });
});

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

  it("keeps stable built-in names and uses a readable catalog fallback", () => {
    expect(
      modelDisplayName({
        name: "flux-schnell:q8",
        description: "Fast FLUX generation",
      }),
    ).toBe("flux-schnell:q8");
    expect(modelDisplayName({ name: "cv:23423432", description: "  " })).toBe(
      "Civitai model #23423432",
    );
    expect(modelDisplayName({ name: "hf:black-forest-labs/FLUX.1-dev" })).toBe(
      "FLUX.1 dev",
    );
  });

  it("resolves a wire model id through the model inventory", () => {
    expect(
      modelDisplayNameForId("cv:23423432", [
        {
          name: "cv:23423432",
          display_name: "Juggernaut XL - Ragnarok",
        },
      ]),
    ).toBe("Juggernaut XL - Ragnarok");
    expect(modelDisplayNameForId("flux-dev:q8", [])).toBe("flux-dev:q8");
  });
});
