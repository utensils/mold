import { describe, expect, it } from "vitest";
import { isCatalogModelId, modelDisplayName } from "./modelName";

describe("isCatalogModelId", () => {
  it("recognizes opaque catalog install ids", () => {
    expect(isCatalogModelId("cv:252914")).toBe(true);
    expect(isCatalogModelId("hf:org/repo")).toBe(true);
    expect(isCatalogModelId("flux-dev:q8")).toBe(false);
  });
});

describe("modelDisplayName", () => {
  it("prefers the server's display_name", () => {
    expect(
      modelDisplayName({
        name: "cv:1759168",
        display_name: "Juggernaut XL - Ragnarok",
        description: "ignored",
      }),
    ).toBe("Juggernaut XL - Ragnarok");
  });

  it("falls back to the description for catalog ids on older servers", () => {
    expect(
      modelDisplayName({
        name: "cv:1759168",
        description: "Juggernaut XL - Ragnarok by KandooAI",
      }),
    ).toBe("Juggernaut XL - Ragnarok by KandooAI");
  });

  it("keeps the raw name for catalog ids with no readable metadata", () => {
    expect(modelDisplayName({ name: "cv:1759168", description: "" })).toBe(
      "cv:1759168",
    );
  });

  it("never swaps manifest names for their marketing description", () => {
    expect(
      modelDisplayName({ name: "flux-dev:q8", description: "The dev model." }),
    ).toBe("flux-dev:q8");
  });
});
