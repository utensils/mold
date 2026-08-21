import { describe, expect, it } from "vitest";
import {
  ID_IMAGE_LIMITS,
  ID_START_STEP_DEFAULT,
  ID_WEIGHT_DEFAULT,
  ID_WEIGHT_MAX,
  IDENTITY_PHOTO_LABEL,
  IDENTITY_START_STEP_LABEL,
  IDENTITY_WEIGHT_LABEL,
  identityActiveCount,
  identityImageError,
  identityProvenance,
  identityRequestFields,
  identityReuse,
  identityValidationError,
  supportsIdentity,
} from "./identityConditioning";

/** A genuine 1x1 PNG, the same fixture shape `mold-core` validates against. */
const PNG_1X1 = [
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49,
  0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06,
  0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44,
  0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0d,
  0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42,
  0x60, 0x82,
];

function toBase64(bytes: readonly number[]): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function pngBase64(width = 1, height = 1): string {
  const bytes = [...PNG_1X1];
  const writeU32 = (offset: number, value: number) => {
    bytes[offset] = (value >>> 24) & 0xff;
    bytes[offset + 1] = (value >>> 16) & 0xff;
    bytes[offset + 2] = (value >>> 8) & 0xff;
    bytes[offset + 3] = value & 0xff;
  };
  writeU32(16, width);
  writeU32(20, height);
  return toBase64(bytes);
}

const PHOTO = { base64: pngBase64(), filename: "ada.png" };

function recipe(supports: boolean, legacy = false) {
  return {
    ...(legacy ? { legacy_adapter: true as const } : {}),
    capabilities: { supports_identity: supports },
  };
}

describe("labels and constants", () => {
  it("names the three controls once for every surface", () => {
    expect(IDENTITY_PHOTO_LABEL).toBe("Identity photo");
    expect(IDENTITY_WEIGHT_LABEL).toBe("Identity strength");
    expect(IDENTITY_START_STEP_LABEL).toBe("Identity start step");
  });

  it("mirrors the mold-core wire constants exactly", () => {
    expect(ID_WEIGHT_DEFAULT).toBe(1.0);
    expect(ID_WEIGHT_MAX).toBe(3.0);
    expect(ID_START_STEP_DEFAULT).toBe(0);
    expect(ID_IMAGE_LIMITS.maxEncodedBytes).toBe(16 * 1024 * 1024);
    expect(ID_IMAGE_LIMITS.maxAxisPixels).toBe(8192);
    expect(ID_IMAGE_LIMITS.maxDecodedPixels).toBe(32_000_000);
  });
});

describe("supportsIdentity", () => {
  it("reads the server-authored recipe capability first", () => {
    expect(supportsIdentity(recipe(true), { supports_identity: false })).toBe(
      true,
    );
    expect(supportsIdentity(recipe(false), { supports_identity: true })).toBe(
      false,
    );
  });

  it("falls back to the model row when there is no authored recipe", () => {
    expect(supportsIdentity(null, { supports_identity: true })).toBe(true);
    expect(supportsIdentity(undefined, { supports_identity: false })).toBe(
      false,
    );
  });

  it("treats an absent field on an older server as no support", () => {
    expect(supportsIdentity(null, {})).toBe(false);
    expect(supportsIdentity(null, null)).toBe(false);
    expect(supportsIdentity(null, { supports_identity: null })).toBe(false);
  });

  it("never lets the client-side legacy adapter answer the question", () => {
    // The Release-N adapter synthesizes `supports_identity: false` locally; it
    // is not the server speaking, so the model row still answers.
    expect(
      supportsIdentity(recipe(false, true), { supports_identity: true }),
    ).toBe(true);
    expect(supportsIdentity(recipe(false, true), {})).toBe(false);
  });
});

describe("identityValidationError", () => {
  const base = {
    supported: true,
    image: PHOTO,
    weight: null,
    startStep: null,
    steps: 20,
    hasLora: false,
    hasSourceImage: false,
    model: "flux-dev:q8",
  };

  it("says nothing when identity is not used at all", () => {
    expect(
      identityValidationError({ ...base, image: null, supported: false }),
    ).toBeNull();
  });

  it("accepts a plain qualified identity photo", () => {
    expect(identityValidationError(base)).toBeNull();
  });

  it("refuses a knob with no photo", () => {
    expect(
      identityValidationError({ ...base, image: null, weight: 1.4 }),
    ).toMatch(/identity photo/i);
    expect(
      identityValidationError({ ...base, image: null, startStep: 3 }),
    ).toMatch(/identity photo/i);
  });

  it("refuses a photo on a model that is not qualified", () => {
    const message = identityValidationError({ ...base, supported: false });
    expect(message).toMatch(/flux-dev:q8/);
    expect(message).toMatch(/identity/i);
  });

  it("refuses identity combined with a LoRA", () => {
    expect(identityValidationError({ ...base, hasLora: true })).toMatch(
      /LoRA/i,
    );
  });

  it("refuses identity combined with a source image", () => {
    expect(identityValidationError({ ...base, hasSourceImage: true })).toMatch(
      /source image/i,
    );
  });

  it("bounds the weight to the server range", () => {
    expect(identityValidationError({ ...base, weight: 0 })).toBeNull();
    expect(identityValidationError({ ...base, weight: 3 })).toBeNull();
    expect(identityValidationError({ ...base, weight: 3.01 })).toMatch(
      /Identity strength/,
    );
    expect(identityValidationError({ ...base, weight: -0.1 })).toMatch(
      /Identity strength/,
    );
    expect(identityValidationError({ ...base, weight: Number.NaN })).toMatch(
      /Identity strength/,
    );
  });

  it("keeps the start step below the step count", () => {
    expect(identityValidationError({ ...base, startStep: 19 })).toBeNull();
    expect(identityValidationError({ ...base, startStep: 20 })).toMatch(
      /Identity start step/,
    );
    expect(identityValidationError({ ...base, startStep: -1 })).toMatch(
      /Identity start step/,
    );
    expect(identityValidationError({ ...base, startStep: 1.5 })).toMatch(
      /Identity start step/,
    );
  });

  it("applies the cheap decode pre-checks to the photo itself", () => {
    expect(
      identityValidationError({
        ...base,
        image: { base64: pngBase64(9000, 10), filename: "wide.png" },
      }),
    ).toMatch(/8192/);
  });
});

describe("identityImageError", () => {
  it("accepts a small PNG", () => {
    expect(identityImageError(pngBase64())).toBeNull();
  });

  it("refuses media that is not PNG or JPEG", () => {
    expect(identityImageError(btoa("GIF89a not an image"))).toMatch(
      /PNG or JPEG/i,
    );
  });

  it("refuses an empty payload", () => {
    expect(identityImageError("")).toMatch(/empty/i);
  });

  it("refuses an oversized axis", () => {
    expect(identityImageError(pngBase64(8193, 8))).toMatch(/8192/);
  });

  it("refuses more than the decode pixel budget", () => {
    expect(identityImageError(pngBase64(8000, 8000))).toMatch(/megapixel/i);
  });

  it("refuses a payload above the encoded byte limit before decoding it", () => {
    // 17 MiB of base64 padding characters — never handed to a decoder.
    const oversized = "A".repeat(Math.ceil((17 * 1024 * 1024 * 4) / 3));
    expect(identityImageError(oversized)).toMatch(/16 MiB/);
  });
});

describe("identityRequestFields", () => {
  it("produces nothing when the model cannot take an identity photo", () => {
    expect(
      identityRequestFields({
        supported: false,
        image: PHOTO,
        weight: 2,
        startStep: 1,
      }),
    ).toEqual({});
  });

  it("produces nothing without a photo, even when knobs are set", () => {
    expect(
      identityRequestFields({
        supported: true,
        image: null,
        weight: 2,
        startStep: 1,
      }),
    ).toEqual({});
  });

  it("ships the photo and its provenance label", () => {
    expect(
      identityRequestFields({
        supported: true,
        image: PHOTO,
        weight: null,
        startStep: null,
      }),
    ).toEqual({ id_image: PHOTO.base64, id_image_name: "ada.png" });
  });

  it("keeps untouched knobs absent so the server default stays authoritative", () => {
    const fields = identityRequestFields({
      supported: true,
      image: { base64: PHOTO.base64, filename: "  " },
      weight: null,
      startStep: null,
    });
    expect(fields.id_weight).toBeUndefined();
    expect(fields.id_start_step).toBeUndefined();
    expect(fields.id_image_name).toBeUndefined();
  });

  it("ships explicitly chosen knobs", () => {
    expect(
      identityRequestFields({
        supported: true,
        image: PHOTO,
        weight: 0.65,
        startStep: 2,
      }),
    ).toEqual({
      id_image: PHOTO.base64,
      id_image_name: "ada.png",
      id_weight: 0.65,
      id_start_step: 2,
    });
  });
});

describe("identityActiveCount", () => {
  it("counts only the Advanced knobs — the photo well is primary form", () => {
    expect(identityActiveCount({ weight: null, startStep: null })).toBe(0);
    expect(identityActiveCount({ weight: 2, startStep: null })).toBe(1);
    expect(identityActiveCount({ weight: 2, startStep: 3 })).toBe(2);
  });
});

describe("identityProvenance", () => {
  it("is absent for a print that carried no identity photo", () => {
    expect(identityProvenance({ prompt: "x" })).toBeNull();
    expect(identityProvenance(null)).toBeNull();
  });

  it("reports the recorded name, digest, weight, and start step", () => {
    const provenance = identityProvenance({
      id_image_name: "ada.png",
      id_image_sha256: "AB".repeat(32),
      id_weight: 0.8,
      id_start_step: 2,
    });
    expect(provenance).toEqual({
      name: "ada.png",
      sha256: "ab".repeat(32),
      shortSha: "abababababab",
      weight: 0.8,
      startStep: 2,
    });
  });

  it("fills the effective defaults a server may omit", () => {
    expect(identityProvenance({ id_image_sha256: "c".repeat(64) })).toEqual({
      name: null,
      sha256: "c".repeat(64),
      shortSha: "cccccccccccc",
      weight: ID_WEIGHT_DEFAULT,
      startStep: ID_START_STEP_DEFAULT,
    });
  });

  it("is recognised from the name alone", () => {
    expect(identityProvenance({ id_image_name: "ada.png" })?.name).toBe(
      "ada.png",
    );
  });

  it("ignores a malformed digest rather than rendering it", () => {
    const provenance = identityProvenance({
      id_image_name: "ada.png",
      id_image_sha256: "not-a-digest",
    });
    expect(provenance?.sha256).toBeNull();
    expect(provenance?.shortSha).toBeNull();
  });
});

describe("identityReuse", () => {
  it("restores the exact recorded knobs and the lookup key", () => {
    expect(
      identityReuse({
        id_image_name: "ada.png",
        id_image_sha256: "d".repeat(64),
        id_weight: 1.5,
        id_start_step: 4,
      }),
    ).toEqual({
      name: "ada.png",
      sha256: "d".repeat(64),
      weight: 1.5,
      startStep: 4,
    });
  });

  it("restores nothing for a print with no identity provenance", () => {
    expect(identityReuse({ prompt: "x" })).toBeNull();
  });
});
