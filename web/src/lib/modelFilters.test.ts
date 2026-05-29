import { describe, expect, it } from "vitest";
import type { ModelInfoExtended } from "../types";
import {
  applyModelFilters,
  familyLabel,
  groupModelsByFamily,
  isStandaloneGenerationModel,
  modelQuantization,
  sortModels,
  variantRank,
} from "./modelFilters";

function model(
  name: string,
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name,
    family: overrides.family ?? "flux",
    size_gb: overrides.size_gb ?? 8,
    is_loaded: false,
    last_used: null,
    hf_repo: overrides.hf_repo ?? "example/repo",
    downloaded: overrides.downloaded ?? true,
    default_width: 1024,
    default_height: 1024,
    default_steps: 28,
    default_guidance: 3.5,
    description: overrides.description ?? "",
    ...overrides,
  };
}

describe("modelFilters", () => {
  it("keeps standalone generation models and excludes utility assets", () => {
    const models = [
      model("flux-dev:q8", { family: "flux" }),
      model("real-esrgan-x4plus", { family: "upscaler" }),
      model("qwen3-expand", { family: "qwen3-expand" }),
      model("sdxl-canny-controlnet", { family: "controlnet" }),
      model("clip-g", {
        family: "companion",
        description: "CLIP-G companion text encoder",
      }),
    ];

    expect(
      models.filter(isStandaloneGenerationModel).map((m) => m.name),
    ).toEqual(["flux-dev:q8"]);
  });

  it("derives quantization and variant rank from model names", () => {
    expect(modelQuantization(model("flux-dev:q8"))).toBe("q8");
    expect(modelQuantization(model("z-image-turbo-bf16"))).toBe("bf16");
    expect(variantRank(model("flux-dev:bf16"))).toBeLessThan(
      variantRank(model("flux-dev:q8")),
    );
    expect(variantRank(model("flux-dev:q8"))).toBeLessThan(
      variantRank(model("flux-dev:q4")),
    );
  });

  it("sorts by multiple keys while preserving stable fallback order", () => {
    const models = [
      model("flux-schnell:q4", { family: "flux", size_gb: 4 }),
      model("sdxl:fp16", { family: "sdxl", size_gb: 7 }),
      model("flux-dev:bf16", { family: "flux", size_gb: 22 }),
      model("flux-schnell:q8", { family: "flux", size_gb: 8 }),
    ];

    expect(
      sortModels(models, ["family", "variantRank", "name"]).map((m) => m.name),
    ).toEqual([
      "flux-dev:bf16",
      "flux-schnell:q8",
      "flux-schnell:q4",
      "sdxl:fp16",
    ]);
  });

  it("applies ALL, ANY, and NOT boolean filter modes", () => {
    const models = [
      model("flux-dev:q8", { family: "flux", size_gb: 8, downloaded: true }),
      model("flux-dev:q4", { family: "flux", size_gb: 4, downloaded: false }),
      model("sdxl:fp16", { family: "sdxl", size_gb: 7, downloaded: true }),
    ];

    expect(
      applyModelFilters(models, {
        mode: "all",
        query: "flux",
        families: ["flux"],
        quantizations: ["q8"],
        sizeBuckets: [],
        downloaded: "downloaded",
      }).map((m) => m.name),
    ).toEqual(["flux-dev:q8"]);

    expect(
      applyModelFilters(models, {
        mode: "any",
        query: "sdxl",
        families: [],
        quantizations: ["q4"],
        sizeBuckets: [],
        downloaded: "any",
      }).map((m) => m.name),
    ).toEqual(["flux-dev:q4", "sdxl:fp16"]);

    expect(
      applyModelFilters(models, {
        mode: "not",
        query: "",
        families: ["flux"],
        quantizations: [],
        sizeBuckets: [],
        downloaded: "missing",
      }).map((m) => m.name),
    ).toEqual(["sdxl:fp16"]);
  });

  it("groups by readable family labels in stable order", () => {
    const groups = groupModelsByFamily([
      model("sdxl:fp16", { family: "sdxl" }),
      model("flux-dev:q8", { family: "flux" }),
      model("z-image-turbo:q8", { family: "z-image" }),
    ]);

    expect(groups.map((g) => [g.family, g.label])).toEqual([
      ["flux", "FLUX"],
      ["sdxl", "SDXL"],
      ["z-image", "Z-Image"],
    ]);
    expect(familyLabel("qwen-image-edit")).toBe("Qwen Image Edit");
  });
});
