import { describe, expect, it } from "vitest";
import { baseGenerationCapabilities } from "./generationCapabilities";
import { FLUX2_DEV_MAX_ATTACHMENTS, sourceMediaPlan } from "./sourceMediaPlan";

function plan(
  family: string,
  model = "",
  advertisedSourceImage: string | null = null,
) {
  return sourceMediaPlan(
    baseGenerationCapabilities(
      family,
      model,
      null,
      null,
      advertisedSourceImage,
    ),
  );
}

describe("sourceMediaPlan", () => {
  it("renders one optional well for image families", () => {
    expect(plan("flux")).toEqual({
      kind: "single",
      required: false,
      endFrame: false,
      video: false,
    });
  });

  it("renders nothing for a text-to-video family without image conditioning", () => {
    expect(plan("ltx-video")).toEqual({ kind: "none" });
  });

  it("follows the advertised per-model contract over the family heuristic", () => {
    // wan T2V says unsupported; wan I2V says required with keyframes.
    expect(plan("wan", "wan-t2v", "unsupported")).toEqual({ kind: "none" });
    expect(plan("wan", "wan-i2v", "required")).toEqual({
      kind: "single",
      required: true,
      endFrame: true,
      video: true,
    });
  });

  it("caps FLUX.2 Dev references and leaves Qwen edit unbounded", () => {
    expect(plan("flux2", "flux2-dev")).toEqual({
      kind: "attachments",
      max: FLUX2_DEV_MAX_ATTACHMENTS,
      required: false,
      primary: null,
    });
    expect(plan("qwen-image-edit")).toEqual({
      kind: "attachments",
      max: null,
      required: true,
      primary: "target",
    });
  });

  it("maps MiniMax H3 tasks to boundary wells and the reference panel", () => {
    expect(plan("minimax-h3", "minimax-h3-fl2va:comfy-pruned-int8")).toEqual({
      kind: "h3-boundaries",
      requiredEndpoint: null,
    });
    expect(
      plan("minimax-h3", "minimax-h3-fl2va:comfy-pruned-int8", "required"),
    ).toEqual({ kind: "h3-boundaries", requiredEndpoint: "first" });
    expect(plan("minimax-h3", "minimax-h3-ref2va:comfy-pruned-int8")).toEqual({
      kind: "h3-references",
    });
  });
});
