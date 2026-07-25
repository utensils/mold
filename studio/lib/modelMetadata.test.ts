import { describe, expect, it } from "vitest";
import {
  modelKindLabel,
  modelKindValue,
  modelWeightsLabel,
} from "./modelMetadata";

describe("model metadata labels", () => {
  it.each([
    ["checkpoint", "Checkpoint"],
    ["lora", "LoRA"],
    ["vae", "VAE"],
    ["text-encoder", "Text encoder"],
    ["tokenizer", "Tokenizer"],
    ["clip", "CLIP"],
    ["control-net", "ControlNet"],
    ["upscaler", "Upscaler"],
    ["prompt-expander", "Prompt expander"],
  ])("renders %s as %s", (kind, label) => {
    expect(modelKindLabel(kind)).toBe(label);
  });

  it("keeps unknown future kinds readable instead of dropping the badge", () => {
    expect(modelKindLabel("motion_adapter")).toBe("Motion adapter");
  });

  it("infers precise built-in utility kinds and defaults generators to checkpoint", () => {
    expect(modelKindValue({ family: "real-esrgan" })).toBe("upscaler");
    expect(modelKindValue({ family: "upscaler" })).toBe("upscaler");
    expect(modelKindValue({ family: "controlnet" })).toBe("control-net");
    expect(modelKindValue({ family: "qwen3-expand" })).toBe("prompt-expander");
    expect(modelKindValue({ family: "flux2" })).toBe("checkpoint");
  });

  it("trusts an explicit catalog kind over family inference", () => {
    expect(modelKindValue({ family: "flux2", kind: "lora" })).toBe("lora");
  });

  it.each([
    ["checkpoint", "Checkpoint weights"],
    ["lora", "LoRA weights"],
    ["vae", "VAE weights"],
    ["text-encoder", "Text encoder weights"],
    ["tokenizer", "Tokenizer files"],
    ["clip", "CLIP weights"],
    ["control-net", "ControlNet weights"],
    ["upscaler", "Upscaler weights"],
    ["prompt-expander", "Prompt expander weights"],
  ])("renders the %s size heading as %s", (kind, heading) => {
    expect(modelWeightsLabel(kind)).toBe(heading);
  });

  it("uses a generic files heading for unknown additive kinds", () => {
    expect(modelWeightsLabel("motion_adapter")).toBe("Model files");
  });
});
