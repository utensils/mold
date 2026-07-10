import { describe, expect, it } from "vitest";
import {
  defaultAdvanced,
  defaultPlacement,
  optionToRef,
  refToOption,
  setAdvancedField,
  setTextEncoders,
  supportsAdvanced,
  TIER2_FAMILIES,
  type DevicePlacement,
} from "./placement";

describe("refToOption / optionToRef", () => {
  it("round-trips auto, cpu, and gpu ordinals", () => {
    for (const opt of ["auto", "cpu", "gpu:0", "gpu:3"]) {
      expect(refToOption(optionToRef(opt))).toBe(opt);
    }
  });

  it("maps DeviceRef variants to option strings", () => {
    expect(refToOption({ kind: "auto" })).toBe("auto");
    expect(refToOption({ kind: "cpu" })).toBe("cpu");
    expect(refToOption({ kind: "gpu", ordinal: 2 })).toBe("gpu:2");
  });

  it("falls back to auto for unrecognized option strings", () => {
    expect(optionToRef("bogus")).toEqual({ kind: "auto" });
    expect(optionToRef("gpu:")).toEqual({ kind: "auto" });
    expect(optionToRef("")).toEqual({ kind: "auto" });
  });
});

describe("supportsAdvanced (tier-2 gating)", () => {
  it("returns true for every tier-2 family", () => {
    for (const fam of TIER2_FAMILIES) {
      expect(supportsAdvanced(fam)).toBe(true);
    }
  });

  it("returns false for non-tier-2 families", () => {
    for (const fam of ["sd1.5", "sdxl", "sd3.5", "ltx-video", "ltx2", "wuerstchen", ""]) {
      expect(supportsAdvanced(fam)).toBe(false);
    }
  });
});

describe("defaultAdvanced / defaultPlacement", () => {
  it("defaults transformer and vae to auto and encoders to null", () => {
    expect(defaultAdvanced()).toEqual({
      transformer: { kind: "auto" },
      vae: { kind: "auto" },
      clip_l: null,
      clip_g: null,
      t5: null,
      qwen: null,
    });
  });

  it("default placement is auto text encoders with no advanced block", () => {
    expect(defaultPlacement()).toEqual({
      text_encoders: { kind: "auto" },
      advanced: null,
    });
  });
});

describe("setTextEncoders", () => {
  it("sets the tier-1 group knob and preserves the advanced block", () => {
    const start: DevicePlacement = {
      text_encoders: { kind: "auto" },
      advanced: { ...defaultAdvanced(), transformer: { kind: "gpu", ordinal: 1 } },
    };
    const next = setTextEncoders(start, { kind: "cpu" });
    expect(next.text_encoders).toEqual({ kind: "cpu" });
    expect(next.advanced?.transformer).toEqual({ kind: "gpu", ordinal: 1 });
  });

  it("works from a null placement", () => {
    const next = setTextEncoders(null, { kind: "gpu", ordinal: 0 });
    expect(next).toEqual({ text_encoders: { kind: "gpu", ordinal: 0 }, advanced: null });
  });
});

describe("setAdvancedField (reducer semantics)", () => {
  it("initializes a full advanced block from a null placement", () => {
    const next = setAdvancedField(null, "transformer", { kind: "cpu" });
    expect(next.text_encoders).toEqual({ kind: "auto" });
    expect(next.advanced).toEqual({
      transformer: { kind: "cpu" },
      vae: { kind: "auto" },
      clip_l: null,
      clip_g: null,
      t5: null,
      qwen: null,
    });
  });

  it("never lets transformer or vae become null (null coerces to auto)", () => {
    const t = setAdvancedField(null, "transformer", null);
    expect(t.advanced?.transformer).toEqual({ kind: "auto" });
    const v = setAdvancedField(null, "vae", null);
    expect(v.advanced?.vae).toEqual({ kind: "auto" });
  });

  it("keeps clip_l / clip_g / t5 / qwen nullable", () => {
    let p: DevicePlacement | null = null;
    for (const field of ["clip_l", "clip_g", "t5", "qwen"] as const) {
      p = setAdvancedField(p, field, { kind: "gpu", ordinal: 0 });
      expect(p.advanced?.[field]).toEqual({ kind: "gpu", ordinal: 0 });
      p = setAdvancedField(p, field, null);
      expect(p.advanced?.[field]).toBeNull();
    }
  });

  it("preserves the text-encoders group knob across advanced edits", () => {
    const start: DevicePlacement = { text_encoders: { kind: "cpu" }, advanced: null };
    const next = setAdvancedField(start, "vae", { kind: "gpu", ordinal: 1 });
    expect(next.text_encoders).toEqual({ kind: "cpu" });
    expect(next.advanced?.vae).toEqual({ kind: "gpu", ordinal: 1 });
  });
});
