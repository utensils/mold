import { describe, expect, it } from "vitest";
import { familyIdentity } from "./modelFamilyIdentity";

describe("familyIdentity", () => {
  it("gives every known family a short mark and a tone", () => {
    expect(familyIdentity("flux2")).toEqual({ mark: "F2", tone: "cool" });
    expect(familyIdentity("flux")).toEqual({ mark: "F", tone: "cool" });
    expect(familyIdentity("sd15")).toEqual({ mark: "1.5", tone: "warm" });
    expect(familyIdentity("ltx2")).toEqual({ mark: "LTX", tone: "neutral" });
  });

  it("marks an unknown family with the start of its slug", () => {
    expect(familyIdentity("hunyuan3d")).toEqual({
      mark: "HUN",
      tone: "neutral",
    });
    expect(familyIdentity("")).toEqual({ mark: "MOD", tone: "neutral" });
  });
});
