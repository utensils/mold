import { describe, expect, it } from "vitest";
import {
  REFERENCE_CROP_ASPECTS,
  REFERENCE_CROP_MIN_AXIS,
  applyReferenceCropTransform,
  moveReferenceCrop,
  normalizeReferenceCrop,
  referenceCropAspectId,
  referenceCropForAspect,
  referenceCropIsIdentity,
  referenceCropProvenance,
  referencePadEstimate,
  resizeReferenceCropFromCorner,
} from "./referenceCrop";

const SOURCE = { width: 1920, height: 1080 };

describe("normalizeReferenceCrop", () => {
  it("rounds to integers and clamps inside the source", () => {
    expect(
      normalizeReferenceCrop(
        { x: 10.4, y: -20, width: 3000.6, height: 500.2 },
        SOURCE,
      ),
    ).toEqual({ x: 10, y: 0, width: 1910, height: 500 });
  });

  it("enforces the 64 px minimum on each axis, pulling the origin back when needed", () => {
    expect(
      normalizeReferenceCrop({ x: 1900, y: 1070, width: 8, height: 8 }, SOURCE),
    ).toEqual({
      x: 1920 - REFERENCE_CROP_MIN_AXIS,
      y: 1080 - REFERENCE_CROP_MIN_AXIS,
      width: REFERENCE_CROP_MIN_AXIS,
      height: REFERENCE_CROP_MIN_AXIS,
    });
  });

  it("caps the minimum at the source itself for tiny images", () => {
    expect(
      normalizeReferenceCrop(
        { x: 0, y: 0, width: 1, height: 1 },
        { width: 40, height: 50 },
      ),
    ).toEqual({ x: 0, y: 0, width: 40, height: 50 });
  });
});

describe("referenceCropIsIdentity", () => {
  it("is true only for the whole source", () => {
    expect(referenceCropIsIdentity({ x: 0, y: 0, ...SOURCE }, SOURCE)).toBe(
      true,
    );
    expect(referenceCropIsIdentity(null, SOURCE)).toBe(true);
    expect(
      referenceCropIsIdentity(
        { x: 1, y: 0, width: 1919, height: 1080 },
        SOURCE,
      ),
    ).toBe(false);
  });
});

describe("referenceCropForAspect", () => {
  it("offers Free plus the canonical families without an orientation twin", () => {
    expect(REFERENCE_CROP_ASPECTS.map((aspect) => aspect.id)).toEqual([
      "free",
      "1:1",
      "4:3",
      "3:2",
      "16:9",
    ]);
  });

  it("centers the largest rect of that aspect inside the source", () => {
    expect(referenceCropForAspect(SOURCE, "1:1")).toEqual({
      x: 420,
      y: 0,
      width: 1080,
      height: 1080,
    });
    expect(
      referenceCropForAspect({ width: 1080, height: 1500 }, "16:9"),
    ).toEqual({
      x: 118,
      y: 0,
      width: 844,
      height: 1500,
    });
    expect(referenceCropForAspect(SOURCE, "free")).toEqual({
      x: 0,
      y: 0,
      ...SOURCE,
    });
  });

  it("follows the source's own orientation so a portrait photo gets a portrait 4:3", () => {
    expect(
      referenceCropForAspect({ width: 1080, height: 1920 }, "4:3"),
    ).toEqual({
      x: 0,
      y: 240,
      width: 1080,
      height: 1440,
    });
  });

  it("names the preset a rect matches within the output-shape tolerance", () => {
    expect(
      referenceCropAspectId({ x: 0, y: 0, width: 1080, height: 1080 }, SOURCE),
    ).toBe("1:1");
    expect(
      referenceCropAspectId({ x: 0, y: 0, width: 1920, height: 1080 }, SOURCE),
    ).toBe("16:9");
    expect(
      referenceCropAspectId({ x: 0, y: 0, width: 1000, height: 1080 }, SOURCE),
    ).toBe("free");
  });
});

describe("resizeReferenceCropFromCorner / moveReferenceCrop", () => {
  const crop = { x: 400, y: 200, width: 800, height: 600 };

  it("drags a corner while its opposite corner stays anchored", () => {
    expect(
      resizeReferenceCropFromCorner(
        crop,
        "se",
        { x: 1500, y: 900 },
        SOURCE,
        null,
      ),
    ).toEqual({ x: 400, y: 200, width: 1100, height: 700 });
    expect(
      resizeReferenceCropFromCorner(
        crop,
        "nw",
        { x: 100, y: 50 },
        SOURCE,
        null,
      ),
    ).toEqual({ x: 100, y: 50, width: 1100, height: 750 });
  });

  it("keeps a locked aspect and never leaves the source", () => {
    const locked = resizeReferenceCropFromCorner(
      crop,
      "se",
      { x: 1900, y: 1000 },
      SOURCE,
      1,
    );
    expect(locked).toEqual({ x: 400, y: 200, width: 880, height: 880 });
    const clamped = resizeReferenceCropFromCorner(
      crop,
      "ne",
      { x: 5000, y: -50 },
      SOURCE,
      null,
    );
    expect(clamped).toEqual({ x: 400, y: 0, width: 1520, height: 800 });
  });

  it("never collapses past the minimum axis", () => {
    expect(
      resizeReferenceCropFromCorner(
        crop,
        "se",
        { x: 401, y: 201 },
        SOURCE,
        null,
      ),
    ).toEqual({
      x: 400,
      y: 200,
      width: REFERENCE_CROP_MIN_AXIS,
      height: REFERENCE_CROP_MIN_AXIS,
    });
  });

  it("moves the whole rect and stops at the edges", () => {
    expect(moveReferenceCrop(crop, 100, -50, SOURCE)).toEqual({
      x: 500,
      y: 150,
      width: 800,
      height: 600,
    });
    expect(moveReferenceCrop(crop, 5000, 5000, SOURCE)).toEqual({
      x: 1120,
      y: 480,
      width: 800,
      height: 600,
    });
  });
});

describe("applyReferenceCropTransform", () => {
  it("is a SourceFitTransform the fit canvas executes unchanged: full-size draw, negative offset", () => {
    expect(
      applyReferenceCropTransform(
        { x: 420, y: 0, width: 1080, height: 1080 },
        SOURCE,
      ),
    ).toEqual({
      outputWidth: 1080,
      outputHeight: 1080,
      drawWidth: 1920,
      drawHeight: 1080,
      offsetX: -420,
      offsetY: 0,
      maskPaddedPixels: false,
    });
  });
});

describe("referencePadEstimate", () => {
  // Pinned to `image_reference_pad_fixtures_match_the_browser_estimate` in
  // crates/mold-core/src/types.rs — the Rust arithmetic is the authority.
  it.each([
    [{ width: 1920, height: 1080 }, 3648, 2048, 7296],
    [{ width: 1080, height: 1080 }, 2048, 2048, 4096],
    [{ width: 1024, height: 768 }, 2720, 2048, 5440],
    [{ width: 1080, height: 1920 }, 2048, 3648, 7296],
    [{ width: 1344, height: 768 }, 3584, 2048, 7168],
    [{ width: 1120, height: 1080 }, 2112, 2048, 4224],
  ])(
    "normalizes %o like the server and counts 32 px cells",
    (size, width, height, pads) => {
      expect(referencePadEstimate(size)).toEqual({
        normalizedWidth: width,
        normalizedHeight: height,
        pads,
      });
    },
  );

  it("estimates the crop rather than the source when one is given", () => {
    expect(
      referencePadEstimate(SOURCE, { x: 420, y: 0, width: 1080, height: 1080 })
        .pads,
    ).toBe(4096);
    expect(referencePadEstimate(SOURCE, null).pads).toBe(7296);
  });
});

describe("referenceCropProvenance", () => {
  it("records the rect beside the uncropped source facts", () => {
    expect(
      referenceCropProvenance(
        { x: 420, y: 0, width: 1080, height: 1080 },
        SOURCE,
        "ab".repeat(32),
      ),
    ).toEqual({
      x: 420,
      y: 0,
      width: 1080,
      height: 1080,
      source_width: 1920,
      source_height: 1080,
      source_sha256: "ab".repeat(32),
    });
  });
});
