import { describe, expect, it } from "vitest";
import { MODEL_PRESENTATIONS } from "./generated/modelPresentationV1";
import {
  isCatalogModelId,
  isOpaqueModelId,
  modelDisplayName,
  modelDisplayNameForId,
  modelPresentation,
  styleLabel,
  styleName,
} from "./modelDisplay";

describe("isOpaqueModelId", () => {
  it("hides only a Civitai install id beside a title", () => {
    expect(isOpaqueModelId("cv:1759168")).toBe(true);
    expect(isOpaqueModelId("hf:black-forest-labs/FLUX.1-dev")).toBe(false);
    expect(isOpaqueModelId("flux-dev:q4")).toBe(false);
  });
});

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

  it("names a manifest id through the generated authority, never as its id", () => {
    expect(
      modelDisplayName({
        name: "flux-schnell:q8",
        description: "Fast FLUX generation",
      }),
    ).toBe("FLUX.1 Schnell Q8");
    expect(modelDisplayName({ name: "cv:23423432", description: "  " })).toBe(
      "Civitai model #23423432",
    );
    expect(modelDisplayName({ name: "hf:black-forest-labs/FLUX.1-dev" })).toBe(
      "FLUX.1 dev",
    );
    expect(modelDisplayName({ name: "my-custom-model" })).toBe(
      "my-custom-model",
    );
  });

  it("names MiniMax H3 by what it does, from the generated table", () => {
    expect(
      modelDisplayName({ name: "minimax-h3-fl2va:comfy-pruned-int8" }),
    ).toBe("MiniMax H3 Frames");
    expect(
      modelDisplayName({
        name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-4step-768p",
      }),
    ).toBe("MiniMax H3 Frames Turbo 4-step 768p");
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
    expect(modelDisplayNameForId("flux-dev:q8", [])).toBe("FLUX.1 Dev Q8");
    expect(modelDisplayNameForId("not-a-known-id", [])).toBe("not-a-known-id");
  });
});

describe("modelPresentation", () => {
  it("reads the generated table for an id no machine reports", () => {
    for (const id of [
      "flux-dev:q4",
      "wan22-ti2v-5b:turbo",
      "minimax-h3-fl2va:comfy-pruned-int8-dmad-4step",
      "hunyuan3d-2mv:fp16",
    ]) {
      const generated = MODEL_PRESENTATIONS[id]!;
      const presented = modelPresentation(id);
      expect(presented.title, id).toBe(generated.title);
      expect(presented.variant, id).toBe(generated.variant ?? null);
      expect(presented.summary, id).toBe(generated.summary ?? null);
      expect(presented.familyLabel, id).toBe(generated.family_label ?? null);
      expect(presented.displayName, id).toBe(
        [generated.title, generated.variant].filter(Boolean).join(" "),
      );
    }
  });

  it("prefers the row's own server fields over the generated table", () => {
    const presented = modelPresentation({
      name: "flux-dev:q4",
      title: "FLUX.1 Dev (server)",
      variant: "Q4_K",
      summary: "Whatever the server says",
      family_label: "FLUX One",
    });
    expect(presented).toEqual({
      title: "FLUX.1 Dev (server)",
      variant: "Q4_K",
      summary: "Whatever the server says",
      familyLabel: "FLUX One",
      displayName: "FLUX.1 Dev (server) Q4_K",
    });
  });

  it("keeps a catalog install's provider title and never renames it", () => {
    expect(
      modelPresentation({
        name: "cv:23423432",
        display_name: "Juggernaut XL - Ragnarok",
        description: "Photoreal SDXL fine-tune",
        family: "sdxl",
      }),
    ).toEqual({
      title: "Juggernaut XL - Ragnarok",
      variant: null,
      summary: "Photoreal SDXL fine-tune",
      familyLabel: "SDXL",
      displayName: "Juggernaut XL - Ragnarok",
    });
    expect(
      modelPresentation({ name: "cv:1", description: "RealVisXL V5.0" }).title,
    ).toBe("RealVisXL V5.0");
  });

  it("stands the family label in as the title of an unknown id, but keeps the id as its name", () => {
    const presented = modelPresentation({ name: "custom", family: "wan" });
    expect(presented.title).toBe("Wan Video");
    expect(presented.displayName).toBe("custom");
    expect(modelPresentation("custom").title).toBe("custom");
    expect(
      modelPresentation({ name: "custom", description: "My own fine-tune" })
        .summary,
    ).toBe("My own fine-tune");
  });

  it("uses an older server's display_name for an id this build does not know", () => {
    expect(
      modelPresentation({ name: "brand-new:q4", display_name: "Brand New Q4" })
        .displayName,
    ).toBe("Brand New Q4");
  });
});

describe("styleName", () => {
  it("says the title and variant, and never the bare id beside the id", () => {
    expect(styleName({ name: "flux-dev:q4", family: "flux" })).toBe(
      "FLUX.1 Dev Q4",
    );
    expect(styleName({ name: "custom", family: "wan" })).toBe("Wan Video");
    expect(styleName({ name: "custom" })).toBe("custom");
  });
});

describe("styleLabel", () => {
  it("keeps the exact runnable id beside the friendly name", () => {
    expect(styleLabel({ name: "flux-dev:q8", family: "flux" })).toBe(
      "FLUX.1 Dev Q8 · flux-dev:q8",
    );
    expect(
      styleLabel({
        name: "cv:23423432",
        family: "sdxl",
        description: "RealVisXL V5.0",
      }),
    ).toBe("RealVisXL V5.0 · cv:23423432");
    expect(styleLabel({ name: "custom", family: "wan" })).toBe(
      "Wan Video · custom",
    );
  });

  it("says a name once when the friendly name IS the id", () => {
    expect(styleLabel({ name: "custom" })).toBe("custom");
  });
});
