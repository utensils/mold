import { describe, expect, it } from "vitest";

import { baseGenerationCapabilities } from "./generationCapabilities";
import {
  qwenImage21Recipe,
  sdxlRecipe,
} from "./generationProfile.testFixtures";
import {
  coerceFormatForTransparency,
  TRANSPARENCY_UNAVAILABLE_FORMAT_REASON,
  transparencyControl,
  transparencyFromProfile,
  transparencyRequestFields,
} from "./transparency";

const ADJUSTABLE = {
  mode: "adjustable" as const,
  default: false,
  formats: ["png", "webp"] as ("png" | "webp")[],
  native_alpha: true,
};

describe("transparencyFromProfile", () => {
  it("reads an absent block as an OLDER SERVER: no capability at all", () => {
    expect(transparencyFromProfile(undefined)).toBeNull();
    expect(transparencyFromProfile(null)).toBeNull();
  });

  it("projects an advertised block, keeping a hidden recipe's own sentence", () => {
    expect(transparencyFromProfile(ADJUSTABLE)).toEqual({
      mode: "adjustable",
      default: false,
      formats: ["png", "webp"],
      nativeAlpha: true,
      reason: null,
    });
    expect(
      transparencyFromProfile({
        mode: "hidden",
        default: false,
        formats: [],
        native_alpha: false,
        reason: "This model does not render transparent backgrounds.",
      }),
    ).toMatchObject({
      mode: "hidden",
      reason: "This model does not render transparent backgrounds.",
    });
  });
});

describe("transparencyControl", () => {
  it("offers the toggle only where the recipe advertises it adjustable", () => {
    const qwen = baseGenerationCapabilities(
      "qwen-image21",
      "qwen-image-2.1:bf16",
      null,
      null,
      null,
      qwenImage21Recipe(),
    );
    expect(transparencyControl(qwen)).toEqual({
      default: false,
      formats: ["png", "webp"],
      nativeAlpha: true,
    });
    // SDXL's fixture predates the block: an older server hides the toggle.
    const sdxl = baseGenerationCapabilities(
      "sdxl",
      "sdxl:fp16",
      null,
      null,
      null,
      sdxlRecipe(),
    );
    expect(sdxl.transparency).toBeNull();
    expect(transparencyControl(sdxl)).toBeNull();
    // A hidden block is a NO, too.
    expect(
      transparencyControl({
        transparency: {
          mode: "hidden",
          default: false,
          formats: [],
          nativeAlpha: false,
          reason: "no",
        },
      }),
    ).toBeNull();
    // No legacy family sniff: an older host carrying no recipe never offers
    // a control it did not advertise.
    expect(
      transparencyControl(
        baseGenerationCapabilities("qwen-image21", "qwen-image-2.1:bf16"),
      ),
    ).toBeNull();
    expect(transparencyControl(null)).toBeNull();
  });

  it("drops an alpha format list emptied by the binary as unusable", () => {
    expect(
      transparencyControl({
        transparency: {
          mode: "adjustable",
          default: false,
          formats: [],
          nativeAlpha: true,
          reason: null,
        },
      }),
    ).toBeNull();
  });
});

describe("coerceFormatForTransparency", () => {
  const control = {
    default: false,
    formats: ["png", "webp"],
    nativeAlpha: true,
  };

  it("leaves the format alone while the toggle is off or unavailable", () => {
    expect(coerceFormatForTransparency("jpeg", control, false)).toEqual({
      format: "jpeg",
      note: null,
    });
    expect(coerceFormatForTransparency("jpeg", null, true)).toEqual({
      format: "jpeg",
      note: null,
    });
  });

  it("keeps an alpha-carrying format", () => {
    expect(coerceFormatForTransparency("webp", control, true)).toEqual({
      format: "webp",
      note: null,
    });
  });

  it("moves JPEG to the first alpha format and says so", () => {
    expect(coerceFormatForTransparency("jpeg", control, true)).toEqual({
      format: "png",
      note: TRANSPARENCY_UNAVAILABLE_FORMAT_REASON,
    });
  });
});

describe("transparencyRequestFields", () => {
  const control = { default: false, formats: ["png"], nativeAlpha: true };

  it("sends the field only when the toggle is on AND advertised", () => {
    expect(transparencyRequestFields(true, control)).toEqual({
      transparent_background: true,
    });
    // Off is ABSENT, never `false`: an ordinary render stays byte-identical
    // on the wire to one from a client that predates the field.
    expect(transparencyRequestFields(false, control)).toEqual({});
    expect(transparencyRequestFields(null, control)).toEqual({});
    // A toggle left on for a model that does not take it parks, like a
    // staged identity photo: kept in the form, absent from the wire.
    expect(transparencyRequestFields(true, null)).toEqual({});
  });
});
