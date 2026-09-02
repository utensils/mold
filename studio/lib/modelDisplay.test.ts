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

  it("names every registered MiniMax H3 manifest consistently", () => {
    expect(
      modelDisplayName({ name: "minimax-h3-fl2va:comfy-pruned-int8" }),
    ).toBe("MiniMax H3 FL2VA");
    expect(
      modelDisplayName({ name: "minimax-h3-ref2va:comfy-pruned-int8" }),
    ).toBe("MiniMax H3 Ref2VA");
    expect(modelDisplayName({ name: "minimax-h3-fl2va:official-bf16" })).toBe(
      "MiniMax H3 FL2VA · Official BF16",
    );
    expect(
      modelDisplayName({ name: "minimax-h3-ref2va:comfy-pruned-nvfp4" }),
    ).toBe("MiniMax H3 Ref2VA · NVFP4");
    expect(
      modelDisplayName({
        name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
      }),
    ).toBe("MiniMax H3 FL2VA Turbo 4-step 768p");
    expect(
      modelDisplayName({
        name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-v1.1",
      }),
    ).toBe("MiniMax H3 FL2VA Turbo 4-step 768p v1.1");
    expect(
      modelDisplayName({
        name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-768p",
      }),
    ).toBe("MiniMax H3 FL2VA Turbo 8-step 768p");
    expect(
      modelDisplayName({
        name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p-r21",
      }),
    ).toBe("MiniMax H3 FL2VA Turbo 4-step 768p (rank 21)");
    expect(
      modelDisplayName({
        name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step-r21",
      }),
    ).toBe("MiniMax H3 FL2VA Turbo 8-step (rank 21)");
    expect(
      modelDisplayName({
        name: "minimax-h3-ref2va:comfy-pruned-int8-turbo-4step-r21",
      }),
    ).toBe("MiniMax H3 Ref2VA Turbo 4-step (rank 21)");
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
