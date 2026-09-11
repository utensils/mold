import { describe, expect, it } from "vitest";
import { composeStyle, mergeStyleNegative, styleHint } from "./stylePresets";
import * as kit from "@ui/lib/stylePresets";

describe("stylePresets", () => {
  it("is the shared kit's composition, not a web-only fork", () => {
    expect(composeStyle).toBe(kit.composeStyle);
    expect(styleHint).toBe(kit.styleHint);
    expect(mergeStyleNegative).toBe(kit.mergeStyleNegative);
  });
});
