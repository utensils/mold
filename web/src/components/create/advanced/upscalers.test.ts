import { describe, expect, it } from "vitest";
import type { ModelInfoExtended } from "../../../types";
import {
  defaultUpscaler,
  isUpscalerModel,
  selectUpscalers,
  upscaleFactor,
} from "./upscalers";

function model(
  name: string,
  overrides: Partial<ModelInfoExtended> = {},
): ModelInfoExtended {
  return {
    name,
    family: "upscaler",
    size_gb: 0.03,
    is_loaded: false,
    last_used: null,
    hf_repo: "",
    downloaded: false,
    default_steps: 0,
    default_guidance: 0,
    default_width: 0,
    default_height: 0,
    description: "",
    ...overrides,
  } as ModelInfoExtended;
}

describe("upscaler model selection", () => {
  it("recognises the server's upscaler family", () => {
    expect(isUpscalerModel(model("real-esrgan-x4plus:fp16"))).toBe(true);
    // Older/remote servers that tag the family differently still match on name.
    expect(
      isUpscalerModel(model("Real-ESRGAN-x2plus", { family: "real-esrgan" })),
    ).toBe(true);
    expect(isUpscalerModel(model("flux-dev:q8", { family: "flux" }))).toBe(
      false,
    );
  });

  it("keeps downloaded upscalers first, then sorts by name", () => {
    const picked = selectUpscalers([
      model("flux-dev:q8", { family: "flux" }),
      model("real-esrgan-x4plus:fp32"),
      model("real-esrgan-x2plus:fp16", { downloaded: true }),
      model("real-esrgan-x4plus:fp16", { downloaded: true }),
    ]).map((m) => m.name);

    expect(picked).toEqual([
      "real-esrgan-x2plus:fp16",
      "real-esrgan-x4plus:fp16",
      "real-esrgan-x4plus:fp32",
    ]);
  });

  it("defaults to a downloaded upscaler, and to none when the host has none", () => {
    expect(
      defaultUpscaler([
        model("real-esrgan-x4plus:fp32", { downloaded: true }),
        model("real-esrgan-x2plus:fp16"),
      ]),
    ).toBe("real-esrgan-x4plus:fp32");
    expect(defaultUpscaler([model("flux-dev:q8", { family: "flux" })])).toBe(
      "",
    );
  });

  it("defaults to the smallest scale when several upscalers are on disk", () => {
    // 4× quadruples the canvas; 2× is the conservative first choice.
    expect(
      defaultUpscaler([
        model("real-esrgan-x4plus:fp16", { downloaded: true }),
        model("real-esrgan-x2plus:fp16", { downloaded: true }),
      ]),
    ).toBe("real-esrgan-x2plus:fp16");
  });

  it("derives the scale factor from the model name or description", () => {
    expect(upscaleFactor(model("real-esrgan-x4plus:fp16"))).toBe(4);
    expect(upscaleFactor(model("real-esrgan-x2plus:fp32"))).toBe(2);
    expect(
      upscaleFactor(
        model("real-esrgan-anime-v3:fp32", {
          description: "Real-ESRGAN Anime Video v3 FP32 — fast anime 4x",
        }),
      ),
    ).toBe(4);
    expect(upscaleFactor(model("mystery-upscaler:fp16"))).toBe(null);
  });
});
