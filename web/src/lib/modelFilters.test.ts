import { describe, expect, it } from "vitest";
import type { ModelInfoExtended } from "../types";
import {
  isStandaloneGenerationModel,
  modelQuantization,
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
});
