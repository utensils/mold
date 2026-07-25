import { describe, expect, it } from "vitest";
import {
  isCatalogModelId,
  modelDisplayName,
  modelDisplayNameForId,
} from "./modelName";

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

  it("uses a readable fallback for catalog ids with no title metadata", () => {
    expect(modelDisplayName({ name: "cv:1759168", description: "" })).toBe(
      "Civitai model #1759168",
    );
  });

  it("never swaps manifest names for their marketing description", () => {
    expect(
      modelDisplayName({ name: "flux-dev:q8", description: "The dev model." }),
    ).toBe("flux-dev:q8");
  });

  it("resolves queue and download ids through the current model list", () => {
    expect(
      modelDisplayNameForId("cv:1759168", [
        { name: "cv:1759168", display_name: "Pony Diffusion V6 XL" },
      ]),
    ).toBe("Pony Diffusion V6 XL");
  });
});
