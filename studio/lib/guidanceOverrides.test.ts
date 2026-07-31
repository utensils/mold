import { describe, expect, it } from "vitest";
import {
  emptyGuidanceOverrides,
  guidanceOverrideCount,
  guidanceOverridesAreEmpty,
  guidanceOverridesError,
  guidanceOverridesFromWire,
  guidanceOverridesToWire,
  MAX_GUIDANCE_SKIP_STEP,
  MAX_STG_BLOCK_INDEX,
  skipStepError,
  stgBlocksError,
} from "./guidanceOverrides";

describe("guidance overrides", () => {
  it("sends nothing when no control was touched", () => {
    const state = emptyGuidanceOverrides();
    expect(guidanceOverridesAreEmpty(state)).toBe(true);
    expect(guidanceOverridesToWire(state)).toBeUndefined();
    expect(guidanceOverridesToWire(null)).toBeUndefined();
  });

  it("sends only the fields that were set", () => {
    const wire = guidanceOverridesToWire({
      ...emptyGuidanceOverrides(),
      stgScale: 1.5,
      skipStep: 2,
    });

    expect(wire).toEqual({ stg_scale: 1.5, skip_step: 2 });
  });

  it("keeps an explicit zero, which is a real setting", () => {
    // stg_scale 0 disables STG — dropping it as falsy would silently restore
    // the pipeline default instead.
    expect(
      guidanceOverridesToWire({ ...emptyGuidanceOverrides(), stgScale: 0 }),
    ).toEqual({ stg_scale: 0 });
  });

  it("parses the block list and reports what it cannot parse", () => {
    expect(
      guidanceOverridesToWire({
        ...emptyGuidanceOverrides(),
        stgBlocks: " 28 , 29 ",
      }),
    ).toEqual({ stg_blocks: [28, 29] });

    expect(stgBlocksError("")).toBeNull();
    expect(stgBlocksError("28,29")).toBeNull();
    expect(stgBlocksError("28,twenty")).toContain("not a block index");
    expect(stgBlocksError("28,28")).toContain("listed twice");
    expect(stgBlocksError(String(MAX_STG_BLOCK_INDEX))).toContain("deeper");
    expect(stgBlocksError("1,2,3,4,5,6,7,8,9")).toContain("At most");
  });

  it("refuses to serialize a skip stride the u32 wire cannot carry", () => {
    // 1.5 would fail JSON deserialization outright and come back as an
    // opaque body-parse error rather than a field-named 422.
    expect(
      guidanceOverridesToWire({
        ...emptyGuidanceOverrides(),
        stgScale: 1,
        skipStep: 1.5,
      }),
    ).toEqual({ stg_scale: 1 });
    expect(skipStepError(1.5)).toContain("whole number");
    expect(skipStepError(null)).toBeNull();
    expect(skipStepError(0)).toBeNull();
    expect(skipStepError(-1)).toContain("between 0 and");
    expect(skipStepError(MAX_GUIDANCE_SKIP_STEP + 1)).toContain(
      "between 0 and",
    );
  });

  it("reports every out-of-contract control through one submit gate", () => {
    expect(guidanceOverridesError(emptyGuidanceOverrides())).toBeNull();
    expect(guidanceOverridesError(undefined)).toBeNull();
    expect(
      guidanceOverridesError({
        ...emptyGuidanceOverrides(),
        stgBlocks: "nope",
      }),
    ).toContain("STG blocks:");
    expect(
      guidanceOverridesError({ ...emptyGuidanceOverrides(), skipStep: 1.5 }),
    ).toContain("Guidance skip stride:");
    // Rescale is an interpolation factor, so its ceiling is 1 even though the
    // other scales accept much larger values.
    expect(
      guidanceOverridesError({
        ...emptyGuidanceOverrides(),
        rescaleScale: 1.5,
      }),
    ).toContain("CFG rescale");
    expect(
      guidanceOverridesError({ ...emptyGuidanceOverrides(), modalityScale: 3 }),
    ).toBeNull();
  });

  it("counts active overrides for the Advanced badge", () => {
    expect(guidanceOverrideCount(emptyGuidanceOverrides())).toBe(0);
    expect(guidanceOverrideCount(undefined)).toBe(0);
    expect(
      guidanceOverrideCount({
        stgScale: 1,
        stgBlocks: "29",
        rescaleScale: 0.5,
        modalityScale: null,
        skipStep: null,
      }),
    ).toBe(3);
  });

  it("round-trips through the wire shape", () => {
    const state = {
      stgScale: 1.5,
      stgBlocks: "28, 29",
      rescaleScale: 0.7,
      modalityScale: 3,
      skipStep: 1,
    };

    expect(guidanceOverridesFromWire(guidanceOverridesToWire(state))).toEqual(
      state,
    );
    expect(guidanceOverridesFromWire(null)).toEqual(emptyGuidanceOverrides());
    expect(guidanceOverridesFromWire({ stg_blocks: [] })).toEqual(
      emptyGuidanceOverrides(),
    );
  });
});
