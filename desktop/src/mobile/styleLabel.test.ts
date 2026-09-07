import { describe, expect, it } from "vitest";
import { mobileStyleLabel } from "./styleLabel";

describe("mobile style labels", () => {
  it("puts a built-in description before its exact runnable ID", () => {
    expect(
      mobileStyleLabel({
        name: "flux-dev:q8",
        family: "flux",
        description: "Detailed still images",
      }),
    ).toBe("Detailed still images · flux-dev:q8");
  });
  it("retains opaque catalog IDs beside their friendly description", () => {
    expect(
      mobileStyleLabel({ name: "cv:1234", family: "sdxl", description: "Portrait photography" }),
    ).toBe("Portrait photography · cv:1234");
  });
  it("falls back to the shared display name or family for older and missing models", () => {
    expect(
      mobileStyleLabel({
        name: "custom:q8",
        family: "flux",
        display_name: "Studio style",
        description: "  ",
      }),
    ).toBe("Studio style · custom:q8");
    expect(mobileStyleLabel({ name: "flux-dev:q8", family: "flux" })).toBe("FLUX · flux-dev:q8");
    expect(
      mobileStyleLabel({ name: "flux-dev:q8", family: "flux", description: "flux-dev:q8" }),
    ).toBe("FLUX · flux-dev:q8");
  });
});
