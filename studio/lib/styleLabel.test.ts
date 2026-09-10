import { describe, expect, it } from "vitest";
import { styleDisplayName, styleLabel } from "./styleLabel";

/*
 * One friendly style name across web, desktop and the phone. It was pinned
 * only through a mobile re-export, so the rule itself — which every style chip
 * and menu row now reads — had no test of its own. The re-export is gone; this
 * is where the rule is answered for.
 */

describe("styleDisplayName", () => {
  it("prefers a catalog description, which is the model's real name", () => {
    expect(
      styleDisplayName({
        name: "cv:23423432",
        family: "sdxl",
        description: "RealVisXL V5.0 by SG161222",
      }),
    ).toBe("RealVisXL V5.0 by SG161222");
  });

  it("takes the server's display name when there is no description", () => {
    expect(
      styleDisplayName({
        name: "cv:1",
        family: "sdxl",
        display_name: "Studio style",
      }),
    ).toBe("Studio style");
  });

  /* A bare manifest id is not a plain word, so the FAMILY's friendly label
   * stands in for it — the id still rides beside in mono wherever this is
   * used, so nothing is lost by not repeating it. */
  it("stands the family's label in for a bare manifest id", () => {
    expect(styleDisplayName({ name: "flux-dev:q8", family: "flux" })).toBe(
      "FLUX",
    );
    expect(
      styleDisplayName({ name: "wan22-ti2v-5b:fp16", family: "wan" }),
    ).toBe("Wan Video");
    expect(styleDisplayName({ name: "custom", family: "a-new-family" })).toBe(
      "A New Family",
    );
  });

  it("never echoes the id back as if it were a name", () => {
    expect(
      styleDisplayName({
        name: "flux-dev:q8",
        family: "flux",
        description: "flux-dev:q8",
      }),
    ).toBe("FLUX");
  });
});

describe("styleLabel", () => {
  it("keeps the exact runnable id beside the friendly name", () => {
    expect(styleLabel({ name: "flux-dev:q8", family: "flux" })).toBe(
      "FLUX · flux-dev:q8",
    );
    expect(
      styleLabel({
        name: "cv:23423432",
        family: "sdxl",
        description: "RealVisXL V5.0",
      }),
    ).toBe("RealVisXL V5.0 · cv:23423432");
  });

  it("says a name once when the friendly name IS the id", () => {
    expect(styleLabel({ name: "FLUX", family: "flux" })).toBe("FLUX");
  });
});
