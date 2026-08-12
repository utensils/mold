import { describe, expect, it } from "vitest";

import {
  firstLastFrameKeyframes,
  firstLastFrameRestoreNotice,
  parseSourceImageCapability,
  resolveSourceImageCapability,
  sourceImageValidationError,
  supportsFirstLastFrames,
} from "./sourceImageCapability";

describe("parseSourceImageCapability", () => {
  it("accepts the three advertised modes", () => {
    expect(parseSourceImageCapability("unsupported")).toBe("unsupported");
    expect(parseSourceImageCapability("optional")).toBe("optional");
    expect(parseSourceImageCapability("required")).toBe("required");
  });

  it("reads absent, null, and unknown values as unknown", () => {
    // An older server omits the field, a current one omits it for an entry it
    // could not classify, and a future one may add a fourth mode. All three
    // must land on the family fallback rather than being guessed into one of
    // today's answers.
    expect(parseSourceImageCapability(undefined)).toBeNull();
    expect(parseSourceImageCapability(null)).toBeNull();
    expect(parseSourceImageCapability("first-frame-only")).toBeNull();
    expect(parseSourceImageCapability(true)).toBeNull();
  });
});

describe("resolveSourceImageCapability", () => {
  it("prefers the advertised mode over the family fallback", () => {
    expect(resolveSourceImageCapability("unsupported", "optional")).toBe(
      "unsupported",
    );
    expect(resolveSourceImageCapability("required", "optional")).toBe(
      "required",
    );
  });

  it("falls back to the family answer when nothing is advertised", () => {
    expect(resolveSourceImageCapability(undefined, "optional")).toBe(
      "optional",
    );
    expect(resolveSourceImageCapability(null, "unsupported")).toBe(
      "unsupported",
    );
  });
});

describe("supportsFirstLastFrames", () => {
  it("needs both a wan model and an advertised image-capable mode", () => {
    expect(supportsFirstLastFrames("wan", "optional")).toBe(true);
    expect(supportsFirstLastFrames("wan", "required")).toBe(true);
    expect(supportsFirstLastFrames("wan", "unsupported")).toBe(false);
  });

  it("stays off when the server advertises nothing", () => {
    // Unlike the source well there is no optimistic fallback: a server old
    // enough to omit the field also refuses wan keyframes outright, so
    // offering the end frame could only produce a rejected request.
    expect(supportsFirstLastFrames("wan", undefined)).toBe(false);
    expect(supportsFirstLastFrames("wan", null)).toBe(false);
  });

  it("stays off for every other family", () => {
    expect(supportsFirstLastFrames("ltx2", "optional")).toBe(false);
    expect(supportsFirstLastFrames("flux", "optional")).toBe(false);
    expect(supportsFirstLastFrames("", "optional")).toBe(false);
  });
});

describe("firstLastFrameKeyframes", () => {
  const first = { base64: "FIRST", filename: "open.png" };
  const last = { base64: "LAST", filename: "close.png" };

  it("emits exactly two entries at 0 and frames - 1", () => {
    expect(firstLastFrameKeyframes(first, last, 81)).toEqual([
      { frame: 0, image: "FIRST", name: "open.png" },
      { frame: 80, image: "LAST", name: "close.png" },
    ]);
  });

  it("recomputes the closing index from the frame count at submit time", () => {
    // The user may change clip length after attaching the end frame; a stale
    // index is rejected by the server.
    expect(firstLastFrameKeyframes(first, last, 49)?.[1]?.frame).toBe(48);
    expect(firstLastFrameKeyframes(first, last, 121)?.[1]?.frame).toBe(120);
  });

  it("omits the name when the attachment has none", () => {
    expect(
      firstLastFrameKeyframes({ base64: "FIRST" }, { base64: "LAST" }, 5),
    ).toEqual([
      { frame: 0, image: "FIRST" },
      { frame: 4, image: "LAST" },
    ]);
  });

  it("produces nothing for a lone first frame", () => {
    // That request is an ordinary source-image render; a single keyframe
    // would change its meaning.
    expect(firstLastFrameKeyframes(first, null, 81)).toBeNull();
    expect(firstLastFrameKeyframes(first, { base64: "" }, 81)).toBeNull();
  });

  it("produces nothing without a first frame or a usable frame count", () => {
    expect(firstLastFrameKeyframes(null, last, 81)).toBeNull();
    expect(firstLastFrameKeyframes(first, last, 1)).toBeNull();
    expect(firstLastFrameKeyframes(first, last, null)).toBeNull();
  });
});

describe("firstLastFrameRestoreNotice", () => {
  it("names BOTH endpoints when the opening frame has no restore handle", () => {
    // A first/last print carries its opening frame only in keyframes[0]
    // (the request holds no source_image), so without independent source
    // provenance the notice must not pretend only the end frame is missing.
    const notice = firstLastFrameRestoreNotice(true, [
      { name: "open.png" },
      { name: "close.png" },
    ]);
    expect(notice).toMatch(/end frame \(close\.png\)/);
    expect(notice).toMatch(/first frame \(open\.png\)/);
    expect(notice).toMatch(/Attach both again/);
    expect(firstLastFrameRestoreNotice(true, [{}, {}])).toMatch(
      /end frame and the first frame can't be restored/i,
    );
  });

  it("names only the end frame when the opening frame is restorable", () => {
    expect(
      firstLastFrameRestoreNotice(
        true,
        [{ name: "open.png" }, { name: "close.png" }],
        true,
      ),
    ).toMatch(/end frame \(close\.png\) can't be restored/);
  });

  it("stays quiet for prints that are not first/last-frame renders", () => {
    expect(firstLastFrameRestoreNotice(true, null)).toBeNull();
    expect(
      firstLastFrameRestoreNotice(true, [{ name: "open.png" }]),
    ).toBeNull();
    // An LTX-2 print's keyframes belong to its own control, not to wan FLF.
    expect(firstLastFrameRestoreNotice(false, [{}, {}])).toBeNull();
  });
});

describe("sourceImageValidationError", () => {
  it("passes an optional model with or without a source", () => {
    expect(
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: false,
      }),
    ).toBeNull();
    expect(
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: true,
      }),
    ).toBeNull();
  });

  it("requires a source image on a required model", () => {
    expect(
      sourceImageValidationError({
        capability: "required",
        hasSourceImage: false,
      }),
    ).toMatch(/image-to-video only/);
    expect(
      sourceImageValidationError({
        capability: "required",
        hasSourceImage: true,
      }),
    ).toBeNull();
  });

  it("refuses a source image on an unsupported model", () => {
    expect(
      sourceImageValidationError({
        capability: "unsupported",
        hasSourceImage: true,
      }),
    ).toMatch(/text-to-video only/);
  });

  it("refuses an end frame without a first frame", () => {
    expect(
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: false,
        hasEndFrame: true,
        frames: 81,
      }),
    ).toMatch(/needs a first frame/);
  });

  it("needs at least two frames for a first/last-frame render", () => {
    expect(
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: true,
        hasEndFrame: true,
        frames: 1,
      }),
    ).toMatch(/at least two frames/);
    expect(
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: true,
        hasEndFrame: true,
        frames: 81,
      }),
    ).toBeNull();
  });

  // #783: a continuation supplies its own first frames from the tail of the
  // clip it continues, which is exactly why the server counts it as carrying
  // source — `mold_core::validation::request_carries_source_frames`.
  it("satisfies a required-source checkpoint from the clip being continued", () => {
    expect(
      sourceImageValidationError({
        capability: "required",
        hasSourceImage: false,
        isExtend: true,
        frames: 49,
        model: "wan22-i2v-a14b:q8",
      }),
    ).toBeNull();
  });

  it("still blocks an ordinary image-to-video render with no image", () => {
    expect(
      sourceImageValidationError({
        capability: "required",
        hasSourceImage: false,
        isExtend: false,
        frames: 49,
        model: "wan22-i2v-a14b:q8",
      }),
    ).toMatch(/image-to-video only/);
    // Absent reads as "not a continuation", the pre-#783 meaning.
    expect(
      sourceImageValidationError({
        capability: "required",
        hasSourceImage: false,
      }),
    ).toMatch(/image-to-video only/);
  });

  it("refuses a continuation on a text-to-video-only checkpoint", () => {
    // The server rejects it at admission with the same reasoning; naming it
    // here keeps the surface from offering a guaranteed 422. The wording is
    // the continuation's, not "remove the image" — it has no image.
    expect(
      sourceImageValidationError({
        capability: "unsupported",
        hasSourceImage: false,
        isExtend: true,
        frames: 49,
        model: "wan22-t2v-a14b:q8",
      }),
    ).toMatch(/text-to-video only and cannot continue/);
    // An ordinary text-to-video render is untouched.
    expect(
      sourceImageValidationError({
        capability: "unsupported",
        hasSourceImage: false,
        isExtend: false,
      }),
    ).toBeNull();
  });

  it("enforces TI2V's latent-endpoint floor of nine frames", () => {
    // TI2V pins endpoints in latent space (4x temporal stride): a 5-frame
    // pixel clip is two latent frames, both anchored. The server rejects
    // this at admission; the client must not offer a guaranteed 422.
    const ti2v = (frames: number, model = "wan22-ti2v-5b:fp16") =>
      sourceImageValidationError({
        capability: "optional",
        hasSourceImage: true,
        hasEndFrame: true,
        frames,
        model,
      });
    expect(ti2v(5)).toMatch(/at least 9 frames/);
    expect(ti2v(9)).toBeNull();
    // A14B channel-concat has no such floor, and opaque catalog ids keep
    // the engine as authority — exactly like the server's name resolution.
    expect(ti2v(5, "wan22-i2v-a14b:q8")).toBeNull();
    expect(ti2v(5, "cv:12345/some-wan-finetune")).toBeNull();
  });
});
