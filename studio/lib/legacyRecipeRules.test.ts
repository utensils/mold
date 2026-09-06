import { describe, expect, it } from "vitest";

import {
  legacyPromptRequirementForFamily,
  legacySupportsStrength,
} from "./legacyRecipeRules";

// These are the rules every client carried BEFORE the generation profile
// advertised `prompt` and `supports_strength`. They survive only as the
// answer for a host that predates those fields, so they must keep saying
// exactly what they said then.
describe("legacyPromptRequirementForFamily", () => {
  it("makes the prompt optional only for LTX-2 visual conditioning", () => {
    expect(legacyPromptRequirementForFamily("ltx2")).toBe("optional");
    expect(legacyPromptRequirementForFamily("ltx-2")).toBe("optional");
    expect(legacyPromptRequirementForFamily("ltx-video")).toBe("required");
  });

  it("normalizes case and surrounding whitespace like the server", () => {
    expect(legacyPromptRequirementForFamily("  LTX2 ")).toBe("optional");
    expect(legacyPromptRequirementForFamily("LTX-Video")).toBe("required");
  });

  it("requires a prompt for image families and an unknown family", () => {
    expect(legacyPromptRequirementForFamily("flux")).toBe("required");
    expect(legacyPromptRequirementForFamily("qwen-image-edit")).toBe(
      "required",
    );
    expect(legacyPromptRequirementForFamily("sd35")).toBe("required");
    expect(legacyPromptRequirementForFamily("")).toBe("required");
    expect(legacyPromptRequirementForFamily(null)).toBe("required");
    expect(legacyPromptRequirementForFamily(undefined)).toBe("required");
  });

  it("never answers ignored — no pre-profile family lacked a text encoder", () => {
    expect(legacyPromptRequirementForFamily("hunyuan3d")).toBe("required");
  });
});

describe("legacySupportsStrength", () => {
  it("withholds strength from the families whose engines never read it", () => {
    expect(legacySupportsStrength("wan", "wan22-i2v-a14b:q8")).toBe(false);
    expect(legacySupportsStrength("qwen-image-edit")).toBe(false);
    expect(legacySupportsStrength("flux2", "flux2-dev:q8")).toBe(false);
    expect(legacySupportsStrength("minimax-h3")).toBe(false);
    expect(legacySupportsStrength("", "minimax-h3-fl2va:official-bf16")).toBe(
      false,
    );
  });

  it("keeps strength for the families that denoise from a source image", () => {
    expect(legacySupportsStrength("flux")).toBe(true);
    expect(legacySupportsStrength("sdxl", "cyberrealistic-pony:fp16")).toBe(
      true,
    );
    expect(legacySupportsStrength("ltx2")).toBe(true);
    expect(legacySupportsStrength("flux2", "flux2-klein:q8")).toBe(true);
  });
});
