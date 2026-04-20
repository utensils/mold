import { describe, expect, it } from "vitest";
import { usePlacement } from "./usePlacement";

describe("usePlacement", () => {
  it("defaults to null placement", () => {
    const { placement } = usePlacement();
    expect(placement.value).toBeNull();
  });

  it("supportsAdvanced returns true for Tier 2 families", () => {
    const { supportsAdvanced } = usePlacement();
    expect(supportsAdvanced("flux")).toBe(true);
    expect(supportsAdvanced("flux2")).toBe(true);
    expect(supportsAdvanced("z-image")).toBe(true);
    expect(supportsAdvanced("qwen-image")).toBe(true);
  });

  it("supportsAdvanced returns false for Tier 1 only families (including sd3)", () => {
    const { supportsAdvanced } = usePlacement();
    // SD3.5 was marked stretch in the spec and cut cleanly (see PR #256).
    // The engine only honors Tier 1; surfacing Advanced here would be a
    // leaky abstraction since the server would silently no-op overrides.
    expect(supportsAdvanced("sd3")).toBe(false);
    expect(supportsAdvanced("sd3.5")).toBe(false);
    expect(supportsAdvanced("stable-diffusion-3")).toBe(false);
    expect(supportsAdvanced("stable-diffusion-3.5")).toBe(false);
    expect(supportsAdvanced("sdxl")).toBe(false);
    expect(supportsAdvanced("sd15")).toBe(false);
    expect(supportsAdvanced("wuerstchen")).toBe(false);
    expect(supportsAdvanced("ltx-video")).toBe(false);
    expect(supportsAdvanced("ltx2")).toBe(false);
  });

  it("sets text encoders Tier 1 without creating advanced", () => {
    const { placement, setTextEncoders } = usePlacement();
    setTextEncoders({ kind: "cpu" });
    expect(placement.value).toEqual({
      text_encoders: { kind: "cpu" },
      advanced: null,
    });
  });

  it("setAdvancedField promotes to Tier 2 automatically", () => {
    const { placement, setAdvancedField } = usePlacement();
    setAdvancedField("transformer", { kind: "gpu", ordinal: 1 });
    expect(placement.value?.advanced?.transformer).toEqual({
      kind: "gpu",
      ordinal: 1,
    });
    expect(placement.value?.text_encoders).toEqual({ kind: "auto" });
  });

  it("loadSaved overwrites current state", () => {
    const { placement, loadSaved } = usePlacement();
    loadSaved({
      text_encoders: { kind: "cpu" },
      advanced: null,
    });
    expect(placement.value?.text_encoders).toEqual({ kind: "cpu" });
  });

  it("clear resets placement to null", () => {
    const { placement, setTextEncoders, clear } = usePlacement();
    setTextEncoders({ kind: "cpu" });
    clear();
    expect(placement.value).toBeNull();
  });

  it("gpuList is a stub until Agent B merges useResources", () => {
    const { gpuList } = usePlacement();
    expect(Array.isArray(gpuList.value)).toBe(true);
  });
});
