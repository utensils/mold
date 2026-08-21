import { describe, expect, it } from "vitest";
import {
  IDENTITY_PHOTO_UNAVAILABLE,
  ID_START_STEP_DEFAULT,
  ID_WEIGHT_DEFAULT,
} from "@studio/lib/identityConditioning";
import { newGenerateForm, type GenerateForm } from "../lib/generateForm";
import { MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES } from "../lib/generateValidation";
import {
  ingestMobileIdentityPhoto,
  mobileIdentityAdvancedCount,
  mobileIdentityBudgetBytes,
  mobileIdentityFileRefusal,
  mobileIdentityMimeType,
  mobileIdentityNeedsReattach,
  mobileIdentityProvenanceRows,
  mobileIdentityStartStepMax,
  resolveMobileIdentityRestore,
  showMobileIdentityWell,
} from "./identity";

/** A 1×1 PNG the shared header pre-checks accept. */
const PNG_1X1 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

function form(overrides: Partial<GenerateForm> = {}): GenerateForm {
  return Object.assign(newGenerateForm(), overrides);
}

describe("mobileIdentityFileRefusal", () => {
  it("accepts PNG and JPEG", () => {
    expect(mobileIdentityFileRefusal({ type: "image/png", name: "face.png" })).toBeNull();
    expect(mobileIdentityFileRefusal({ type: "image/jpeg", name: "face.jpg" })).toBeNull();
  });

  it("accepts a typeless pick whose name still reads as a still image", () => {
    // iOS hands some picks over without a MIME type at all.
    expect(mobileIdentityFileRefusal({ type: "", name: "IMG_0042.HEIC" })).not.toBeNull();
    expect(mobileIdentityFileRefusal({ type: "", name: "IMG_0042.jpeg" })).toBeNull();
  });

  it("refuses anything else by name", () => {
    expect(mobileIdentityFileRefusal({ type: "image/gif", name: "face.gif" })).toContain(
      "PNG or JPEG",
    );
  });
});

describe("mobileIdentityBudgetBytes", () => {
  it("spends the same request budget as every other inline input", () => {
    expect(mobileIdentityBudgetBytes(form())).toBe(MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES);
  });

  it("excludes the identity photo already staged, so replacing one is not double-counted", () => {
    const staged = form({ identityImage: { filename: "face.png", base64: "A".repeat(4000) } });
    expect(mobileIdentityBudgetBytes(staged)).toBe(MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES);
  });

  it("is reduced by other staged media", () => {
    const staged = form({ sourceImage: "A".repeat(4000) });
    expect(mobileIdentityBudgetBytes(staged)).toBeLessThan(
      MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES,
    );
  });
});

describe("ingestMobileIdentityPhoto", () => {
  const budget = MAX_MOBILE_GENERATION_REQUEST_MEDIA_BYTES;

  it("stages a valid photo verbatim — never fitted or cropped to the canvas", () => {
    const result = ingestMobileIdentityPhoto({ filename: "face.png", base64: PNG_1X1 }, budget);
    expect(result).toEqual({ ok: true, image: { filename: "face.png", base64: PNG_1X1 } });
  });

  it("names the photo when the pick has no filename", () => {
    const result = ingestMobileIdentityPhoto({ filename: "", base64: PNG_1X1 }, budget);
    expect(result.ok && result.image.filename).toBe("identity photo");
  });

  it("reports the server's own header refusal", () => {
    const result = ingestMobileIdentityPhoto({ filename: "face.png", base64: "AAAA" }, budget);
    expect(result.ok).toBe(false);
    expect(result.ok === false && result.error).toContain("PNG or JPEG");
  });

  it("refuses a photo that would blow the phone's request budget", () => {
    const result = ingestMobileIdentityPhoto({ filename: "face.png", base64: PNG_1X1 }, 0);
    expect(result.ok).toBe(false);
    expect(result.ok === false && result.error).toContain("45 MiB");
  });
});

describe("mobileIdentityMimeType", () => {
  it("derives the preview type from the provenance label", () => {
    expect(mobileIdentityMimeType("face.jpg")).toBe("image/jpeg");
    expect(mobileIdentityMimeType("FACE.JPEG")).toBe("image/jpeg");
    expect(mobileIdentityMimeType("face.png")).toBe("image/png");
    expect(mobileIdentityMimeType(null)).toBe("image/png");
  });
});

describe("mobileIdentityNeedsReattach", () => {
  it("is true only for provenance with no bytes", () => {
    expect(mobileIdentityNeedsReattach(null)).toBe(false);
    expect(mobileIdentityNeedsReattach({ filename: "face.png", base64: PNG_1X1 })).toBe(false);
    expect(mobileIdentityNeedsReattach({ filename: "face.png", base64: "" })).toBe(true);
  });
});

describe("showMobileIdentityWell", () => {
  it("renders only on positive capability, and never for a sequence", () => {
    expect(showMobileIdentityWell(form({ identitySupported: true }), false)).toBe(true);
    expect(showMobileIdentityWell(form({ identitySupported: true }), true)).toBe(false);
    expect(showMobileIdentityWell(form({ identitySupported: false }), false)).toBe(false);
    // Unread capability is not evidence of support.
    expect(showMobileIdentityWell(form({ identitySupported: null }), false)).toBe(false);
  });

  it("keeps a parked photo hidden rather than rendering a dead control", () => {
    const parked = form({
      identitySupported: false,
      identityImage: { filename: "face.png", base64: PNG_1X1 },
    });
    expect(showMobileIdentityWell(parked, false)).toBe(false);
  });
});

describe("mobileIdentityAdvancedCount", () => {
  it("counts only the two knobs, and only on a qualified checkpoint", () => {
    expect(mobileIdentityAdvancedCount(form({ identitySupported: true }))).toBe(0);
    expect(
      mobileIdentityAdvancedCount(form({ identitySupported: true, identityWeight: 1.2 })),
    ).toBe(1);
    expect(
      mobileIdentityAdvancedCount(
        form({ identitySupported: true, identityWeight: 1.2, identityStartStep: 3 }),
      ),
    ).toBe(2);
    // The photo well itself is primary form, exactly like the source wells.
    expect(
      mobileIdentityAdvancedCount(
        form({ identitySupported: true, identityImage: { filename: "f.png", base64: PNG_1X1 } }),
      ),
    ).toBe(0);
    // Parked knobs on an unqualified checkpoint never reach the wire, so they
    // never claim a badge either.
    expect(
      mobileIdentityAdvancedCount(form({ identitySupported: false, identityWeight: 1.2 })),
    ).toBe(0);
  });
});

describe("mobileIdentityStartStepMax", () => {
  it("is one fewer than the steps this print renders", () => {
    expect(mobileIdentityStartStepMax(20)).toBe(19);
    expect(mobileIdentityStartStepMax(1)).toBe(0);
    expect(mobileIdentityStartStepMax(0)).toBe(0);
    expect(mobileIdentityStartStepMax(Number.NaN)).toBe(0);
  });
});

describe("mobileIdentityProvenanceRows", () => {
  it("is null for a print that carried no identity photo", () => {
    expect(mobileIdentityProvenanceRows({})).toBeNull();
    expect(mobileIdentityProvenanceRows(null)).toBeNull();
  });

  it("names the photo, its short digest, and the effective knobs", () => {
    const rows = mobileIdentityProvenanceRows({
      id_image_name: "ada.png",
      id_image_sha256: "a".repeat(64),
      id_weight: 0.8,
      id_start_step: 2,
    });
    expect(rows).toEqual([
      { label: "Identity photo", value: `ada.png · ${"a".repeat(12)}`, title: "a".repeat(64) },
      { label: "Identity strength", value: "0.8 · from step 2" },
    ]);
  });

  it("falls back to the shared defaults when an older host recorded only a name", () => {
    const rows = mobileIdentityProvenanceRows({ id_image_name: "ada.png" });
    expect(rows).toEqual([
      { label: "Identity photo", value: "ada.png", title: undefined },
      {
        label: "Identity strength",
        value: `${ID_WEIGHT_DEFAULT} · from step ${ID_START_STEP_DEFAULT}`,
      },
    ]);
  });
});

describe("resolveMobileIdentityRestore", () => {
  it("skips a form with no descriptor, or one the user already reattached", () => {
    expect(resolveMobileIdentityRestore(null, { base64: PNG_1X1, filename: "face.png" })).toEqual({
      kind: "skip",
    });
    expect(
      resolveMobileIdentityRestore(
        { filename: "face.png", base64: PNG_1X1 },
        { base64: PNG_1X1, filename: "face.png" },
      ),
    ).toEqual({ kind: "skip" });
  });

  it("re-attaches the stashed photo, preferring the stash's own filename", () => {
    expect(
      resolveMobileIdentityRestore(
        { filename: "identity photo", base64: "" },
        { base64: PNG_1X1, filename: "ada.png" },
      ),
    ).toEqual({ kind: "attached", image: { filename: "ada.png", base64: PNG_1X1 } });
  });

  it("keeps the recorded label when the stash has no filename of its own", () => {
    expect(
      resolveMobileIdentityRestore({ filename: "ada.png", base64: "" }, { base64: PNG_1X1 }),
    ).toEqual({ kind: "attached", image: { filename: "ada.png", base64: PNG_1X1 } });
  });

  it("discloses a miss instead of rendering a different person", () => {
    expect(resolveMobileIdentityRestore({ filename: "ada.png", base64: "" }, null)).toEqual({
      kind: "missing",
      note: IDENTITY_PHOTO_UNAVAILABLE,
    });
  });
});
