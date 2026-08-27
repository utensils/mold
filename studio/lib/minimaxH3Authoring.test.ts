import { describe, expect, it, vi } from "vitest";
import type { GenerationReference } from "./generationReferences";
import { sha256PaddedBase64 } from "./base64Digest";
import {
  MINIMAX_H3_FIXED_FPS,
  MINIMAX_H3_FL2VA_COMFY,
  MINIMAX_H3_FL2VA_COMFY_NVFP4,
  MINIMAX_H3_REF2VA_COMFY,
  MINIMAX_H3_REF2VA_COMFY_NVFP4,
  MINIMAX_H3_RESYNTHESIS_GUIDANCE,
  appendMinimaxH3GalleryImageReference,
  appendMinimaxH3PickedImageReferences,
  canonicalMinimaxH3ModelName,
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Family,
  isMinimaxH3Identity,
  minimaxH3AuthoringCapabilities,
  minimaxH3AuthoringError,
  minimaxH3BoundaryFromSourceMetadata,
  minimaxH3BoundaryFromStagedImage,
  minimaxH3ClosingBoundaryFromMetadata,
  minimaxH3Mode,
  minimaxH3ReferenceBudget,
  minimaxH3ReferenceDraftsFromMetadata,
  minimaxH3TaskForModel,
  moveMinimaxH3Reference,
  applyMinimaxH3ReferenceCrops,
  reattachMinimaxH3Reference,
  setMinimaxH3ReferenceCrop,
  stripMinimaxH3AuthoringMedia,
  serializeMinimaxH3Authoring,
  setMinimaxH3BoundaryFile,
  setMinimaxH3GalleryImageBoundary,
  setMinimaxH3GalleryImageFirstFrame,
  setMinimaxH3PickedImageBoundary,
  setMinimaxH3PickedImageFirstFrame,
  stagedImageFromMinimaxH3Boundary,
} from "./minimaxH3Authoring";

const image = (name: string): GenerationReference => ({
  kind: "image",
  media: { authority: "inline", data: name.toUpperCase() },
  provenance: { name, sha256: name.repeat(64).slice(0, 64) },
  mime_type: "image/png",
  width: 1024,
  height: 768,
});

const video = (name: string, duration_ms = 4_000): GenerationReference => ({
  kind: "video",
  media: { authority: "upload", handle: `handle-${name}` },
  provenance: { name, sha256: "a".repeat(64) },
  mime_type: "video/mp4",
  width: 1280,
  height: 720,
  frame_count: 96,
  duration_ms,
  fps: 24,
  has_audio: true,
  audio_duration_ms: duration_ms,
  audio_sample_count: duration_ms * 48,
  audio_sample_rate: 48_000,
  audio_channels: 2,
});

const audio = (name: string, duration_ms = 3_000): GenerationReference => ({
  kind: "audio",
  media: { authority: "upload", handle: `handle-${name}` },
  provenance: { name, sha256: "b".repeat(64) },
  mime_type: "audio/wav",
  duration_ms,
  sample_rate: 48_000,
  channels: 2,
  sample_count: duration_ms * 48,
});

describe("MiniMax H3 Studio authority", () => {
  it("appends hashed picker images in one preserved semantic order", async () => {
    const result = await appendMinimaxH3PickedImageReferences(
      emptyMinimaxH3AuthoringState(),
      [
        { filename: "second.jpg", base64: "U0VDT05E", width: 8, height: 6 },
        { filename: "first.png", base64: "RklSU1Q=", width: 6, height: 8 },
      ],
    );
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(
      result.state.references.map((draft) => ({
        name: draft.reference.provenance?.name,
        mime: draft.reference.mime_type,
        sha256: draft.reference.provenance?.sha256,
      })),
    ).toEqual([
      {
        name: "second.jpg",
        mime: "image/jpeg",
        sha256:
          "84747dcd831c7131207a1042d74b60ac54fdcc10e7aef9e1f7940dda2c95dcae",
      },
      {
        name: "first.png",
        mime: "image/png",
        sha256:
          "267d3b81a9dcd937f3b46a17a57fc0ca2133373389336861142673a73fc17bc6",
      },
    ]);
  });

  it("hashes picker references sequentially to bound transient media memory", async () => {
    let resolveFirst!: (value: ArrayBuffer) => void;
    const firstDigest = new Promise<ArrayBuffer>((resolve) => {
      resolveFirst = resolve;
    });
    const digest = vi
      .spyOn(globalThis.crypto.subtle, "digest")
      .mockImplementationOnce(() => firstDigest)
      .mockResolvedValue(new ArrayBuffer(32));
    try {
      const pending = appendMinimaxH3PickedImageReferences(null, [
        { filename: "one.png", base64: "T05F", width: 1, height: 1 },
        { filename: "two.png", base64: "VFdP", width: 1, height: 1 },
      ]);
      await Promise.resolve();
      expect(digest).toHaveBeenCalledTimes(1);

      resolveFirst(new ArrayBuffer(32));
      const result = await pending;
      expect(result.ok).toBe(true);
      expect(digest).toHaveBeenCalledTimes(2);
    } finally {
      digest.mockRestore();
    }
  });

  it("resolves only the released explicit task partitions", () => {
    expect(isMinimaxH3Family(" MiniMax_H3 ")).toBe(true);
    expect(minimaxH3TaskForModel("minimax-h3-fl2va:official-bf16")).toBe(
      "fl2va",
    );
    expect(minimaxH3TaskForModel("minimax_h3_ref2va:comfy-pruned-int8")).toBe(
      "ref2va",
    );
    expect(minimaxH3TaskForModel(MINIMAX_H3_FL2VA_COMFY_NVFP4)).toBe("fl2va");
    expect(minimaxH3TaskForModel(MINIMAX_H3_REF2VA_COMFY_NVFP4)).toBe("ref2va");
    expect(minimaxH3TaskForModel("hf:opaque")).toBeNull();
    expect(isMinimaxH3Identity(null, "minimax_h3_ref2va")).toBe(true);
    expect(isMinimaxH3Identity(null, "hf:opaque")).toBe(false);
    expect(isMinimaxH3Identity(null, "minimax-h3-ref2va:future")).toBe(true);
    expect(canonicalMinimaxH3ModelName(" MiniMax_H3_Ref2VA ")).toBe(
      MINIMAX_H3_REF2VA_COMFY,
    );
    expect(canonicalMinimaxH3ModelName("minimax-h3-ref2va:future")).toBeNull();
  });

  it("normalizes an existing surface-picker image into the FL2VA first-frame authority", () => {
    const png = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";
    const result = setMinimaxH3PickedImageFirstFrame(null, {
      filename: "opening.png",
      base64: png,
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.state.firstFrame).toMatchObject({
      filename: "opening.png",
      mimeType: "image/png",
      width: 7,
      height: 4,
      data: png,
    });
  });

  it("normalizes a surface-picker image into the FL2VA closing boundary without touching the opening one", () => {
    const png = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";
    const opened = setMinimaxH3PickedImageBoundary(null, "firstFrame", {
      filename: "opening.png",
      base64: png,
    });
    expect(opened.ok).toBe(true);
    if (!opened.ok) return;

    const closed = setMinimaxH3PickedImageBoundary(opened.state, "lastFrame", {
      filename: "closing.png",
      base64: png,
    });
    expect(closed.ok).toBe(true);
    if (!closed.ok) return;
    expect(closed.state.firstFrame?.filename).toBe("opening.png");
    expect(closed.state.lastFrame).toMatchObject({
      filename: "closing.png",
      mimeType: "image/png",
      width: 7,
      height: 4,
      data: png,
    });
    expect(closed.reference).toBeNull();
  });

  it("reads a picked file into a boundary and refuses non-still media", async () => {
    const png = "iVBORw0KGgoAAAANSUhEUgAAAAcAAAAECAIAAAAmkwkpAAAAAElFTkSuQmCC";
    const bytes = Uint8Array.from(atob(png), (c) => c.charCodeAt(0));
    const attached = await setMinimaxH3BoundaryFile(
      null,
      "lastFrame",
      new File([bytes], "closing.png", { type: "image/png" }),
    );
    expect(attached.ok).toBe(true);
    if (!attached.ok) return;
    expect(attached.state.lastFrame).toMatchObject({
      filename: "closing.png",
      mimeType: "image/png",
      width: 7,
      height: 4,
      data: png,
    });

    const refused = await setMinimaxH3BoundaryFile(
      null,
      "firstFrame",
      new File(["mp4"], "clip.mp4", { type: "video/mp4" }),
    );
    expect(refused).toEqual({
      ok: false,
      error: "FL2VA endpoints must be still images.",
    });

    const undecodable = await setMinimaxH3BoundaryFile(
      null,
      "firstFrame",
      new File(["webp"], "still.webp", { type: "image/webp" }),
    );
    expect(undecodable).toEqual({
      ok: false,
      error: "Use a PNG or JPEG image for FL2VA endpoints.",
    });
  });

  it("rejects a non-image gallery pick for the closing boundary", () => {
    const result = setMinimaxH3GalleryImageBoundary(null, "lastFrame", {
      filename: "clip.mp4",
      mimeType: "video/mp4",
      width: 1280,
      height: 720,
      data: "VIDEO",
    });
    expect(result).toEqual({
      ok: false,
      error: "Only gallery images can be used as MiniMax H3 visual references.",
    });
  });

  it.each([
    ["MiniMax_H3", "minimax-h3-fl2va:comfy-pruned-int8"],
    ["MiniMax_H3_Ref2VA", "minimax-h3-ref2va:comfy-pruned-int8"],
    ["MiniMax_H3_FL2VA:OFFICIAL_BF16", "minimax-h3-fl2va:official-bf16"],
    [
      "minimax-h3-ref2va:comfy-pruned-int8",
      "minimax-h3-ref2va:comfy-pruned-int8",
    ],
  ])("canonicalizes %s inside the frozen H3 request", (model, expected) => {
    const request = serializeMinimaxH3Authoring(
      { model, frames: 124 },
      null,
      model,
      emptyMinimaxH3AuthoringState(),
    );
    expect(request.model).toBe(expected);
  });

  it("keeps an unknown H3 partition unchanged while failing its task projection closed", () => {
    const model = "minimax-h3-ref2va:future-layout";
    const request = serializeMinimaxH3Authoring(
      {
        model,
        frames: 124,
        source_image: "STALE",
        references: [image("stale.png")],
      },
      "minimax-h3",
      model,
      emptyMinimaxH3AuthoringState(),
    );
    expect(request.model).toBe(model);
    expect(request).not.toHaveProperty("source_image");
    expect(request).not.toHaveProperty("references");
    expect(minimaxH3AuthoringError(null, model, null)).toContain(
      "explicit FL2VA or Ref2VA",
    );
  });

  it("requires exactly one first endpoint for the reviewed compact profile", () => {
    const model = MINIMAX_H3_FL2VA_COMFY;
    expect(
      minimaxH3AuthoringError(
        "minimax-h3",
        model,
        emptyMinimaxH3AuthoringState(),
        true,
      ),
    ).toContain("requires a first frame");

    const state = emptyMinimaxH3AuthoringState();
    state.firstFrame = {
      filename: "first.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "AQ==",
    };
    expect(
      minimaxH3AuthoringError("minimax-h3", model, state, true),
    ).toBeNull();
    state.lastFrame = {
      filename: "last.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "Ag==",
    };
    expect(minimaxH3AuthoringError("minimax-h3", model, state, true)).toContain(
      "only one first-frame endpoint",
    );
  });

  it("appends a gallery image as the final ordered inline Ref2VA reference", async () => {
    const state = emptyMinimaxH3AuthoringState();
    const original = { reference: video("motion.mp4") };
    state.references = [original];

    const result = await appendMinimaxH3GalleryImageReference(state, {
      filename: " subject.png ",
      mimeType: "image/png; charset=binary",
      width: 1280,
      height: 720,
      data: "U1VCSkVDVA==",
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.reference).toBe(2);
    expect(result.state.references[0]).toBe(original);
    expect(result.state.references[1]?.reference).toEqual({
      kind: "image",
      media: { authority: "inline", data: "U1VCSkVDVA==" },
      provenance: {
        name: "subject.png",
        sha256:
          "5995c8078362c12933e619215e1bd732118ae7e7ae32ddbf89db02057cbdb4c5",
      },
      mime_type: "image/png",
      width: 1280,
      height: 720,
    });
  });

  it("enforces gallery reference limits and maps FL2VA source to its first frame", async () => {
    const images = Array.from({ length: 9 }, (_, index) => ({
      reference: image(`${index}.png`),
    }));
    const rejected = await appendMinimaxH3GalleryImageReference(
      { ...emptyMinimaxH3AuthoringState(), references: images },
      {
        filename: "overflow.png",
        mimeType: "image/png",
        width: 10,
        height: 10,
        data: "OVERFLOW",
      },
    );
    expect(rejected).toEqual({
      ok: false,
      error: "Use at most 9 image references.",
    });

    const boundary = setMinimaxH3GalleryImageFirstFrame(null, {
      filename: "opening.jpg",
      mimeType: "image/jpeg",
      width: 640,
      height: 480,
      data: "OPENING",
    });
    expect(boundary).toMatchObject({
      ok: true,
      reference: null,
      state: {
        firstFrame: {
          filename: "opening.jpg",
          mimeType: "image/jpeg",
          width: 640,
          height: 480,
          data: "OPENING",
        },
      },
    });
  });

  it("restores opening-frame gallery provenance as explicitly missing media", () => {
    const boundary = minimaxH3BoundaryFromSourceMetadata(
      "opening.png",
      "a".repeat(64),
    );
    expect(boundary).toMatchObject({
      filename: "opening.png",
      data: "",
      sha256: "a".repeat(64),
    });
    expect(
      minimaxH3AuthoringError("minimax-h3", "minimax-h3-fl2va", {
        ...emptyMinimaxH3AuthoringState(),
        firstFrame: boundary,
      }),
    ).toContain("Reattach first frame opening.png");
  });

  it("reports the compact step range, and a Turbo tier's exact count", () => {
    const base = minimaxH3AuthoringCapabilities({
      name: "minimax-h3-fl2va:comfy-pruned-int8",
      family: "minimax-h3",
    })!;
    expect(base.minSteps).toBe(2);
    expect(base.maxSteps).toBe(50);
    expect(base.minFrames).toBe(107);
    expect(base.maxFrames).toBe(345);

    // A distilled adapter's schedule length is not a preference.
    const turbo = minimaxH3AuthoringCapabilities({
      name: "minimax-h3-fl2va:comfy-pruned-int8-turbo-8step",
      family: "minimax-h3",
    })!;
    expect(turbo.minSteps).toBe(9);
    expect(turbo.maxSteps).toBe(9);
    // The frame grid is still the family's — only the step axis is the
    // adapter's property.
    expect(turbo.minFrames).toBe(107);
    expect(turbo.maxFrames).toBe(345);
  });

  it("keeps runtime and legal access as independent fail-closed authorities", () => {
    expect(
      minimaxH3AuthoringCapabilities({
        name: "minimax-h3-fl2va:official-bf16",
        family: "minimax-h3",
        runtime_available: false,
      })?.runtimeAvailable,
    ).toBe(false);
    expect(
      minimaxH3AuthoringCapabilities(
        {
          name: "minimax-h3-fl2va:official-bf16",
          family: "minimax-h3",
        },
        {
          model_access: {
            restrictions: [
              {
                code: "MINIMAX_H3_AUTHORIZATION_REQUIRED",
                family: "minimax-h3",
                message: "authorization required",
                license_url: "https://example.test/license",
                authorization_url: "https://example.test/authorize",
              },
            ],
          },
        },
      )?.runtimeAvailable,
    ).toBe(false);
  });

  it("derives every FL2VA endpoint mode from first/last presence", () => {
    const state = emptyMinimaxH3AuthoringState();
    expect(minimaxH3Mode("fl2va", state)).toBe("text-to-audio-video");
    state.firstFrame = {
      filename: "first.png",
      mimeType: "image/png",
      width: 1,
      height: 1,
      data: "FIRST",
    };
    expect(minimaxH3Mode("fl2va", state)).toBe("first-frame-to-audio-video");
    state.lastFrame = {
      ...state.firstFrame,
      filename: "last.png",
      data: "LAST",
    };
    expect(minimaxH3Mode("fl2va", state)).toBe(
      "first-and-last-frame-to-audio-video",
    );
    state.firstFrame = null;
    expect(minimaxH3Mode("fl2va", state)).toBe("last-frame-to-audio-video");
  });

  it("validates ordered heterogeneous counts, durations, soundtrack association, and audio-only sets", () => {
    const valid = minimaxH3ReferenceBudget([
      { reference: image("subject.png") },
      { reference: video("motion.mp4") },
      { reference: audio("voice.wav") },
    ]);
    expect(valid).toMatchObject({
      total: 3,
      images: 1,
      videos: 1,
      audios: 1,
      videoDurationMs: 4_000,
      audioDurationMs: 7_000,
      errors: [],
    });

    const invalid = minimaxH3ReferenceBudget([
      { reference: audio("voice.wav", 1_000) },
      { reference: video("long.mp4", 16_000) },
    ]);
    expect(invalid.errors.join(" ")).toContain("Reference 1 audio duration");
    expect(invalid.errors.join(" ")).toContain("Reference 2 video duration");
    expect(invalid.errors.join(" ")).toContain("Combined reference video");
    expect(invalid.errors.join(" ")).toContain("Combined standalone audio");
  });

  it("reorders without regrouping media kinds", () => {
    const mixed = [
      { reference: image("one.png") },
      { reference: audio("two.wav") },
      { reference: video("three.mp4") },
    ];
    expect(
      moveMinimaxH3Reference(mixed, 2, 0).map((entry) => entry.reference.kind),
    ).toEqual(["video", "image", "audio"]);
  });

  it("serializes H3 scalars and FL2VA first/last endpoints from one authority", () => {
    const state = emptyMinimaxH3AuthoringState();
    state.firstFrame = {
      filename: "first.png",
      mimeType: "image/png",
      width: 1280,
      height: 720,
      data: "FIRST",
    };
    state.lastFrame = {
      ...state.firstFrame,
      filename: "last.png",
      data: "LAST",
    };
    const request = serializeMinimaxH3Authoring(
      {
        prompt: "a new shot",
        frames: 130,
        guidance: 7,
        strength: 0.3,
        output_format: "gif",
        negative_prompt: "bad",
        loras: [{ path: "nope", scale: 1 }],
        references: [{ reference: image("stale.png") }],
      },
      "minimax-h3",
      "minimax-h3-fl2va:comfy-pruned-int8",
      state,
    );
    expect(request).toMatchObject({
      frames: 124,
      fps: MINIMAX_H3_FIXED_FPS,
      batch_size: 1,
      guidance: 0,
      strength: 1,
      output_format: "mp4",
      source_image: "FIRST",
      source_image_name: "first.png",
      keyframes: [{ frame: 123, image: "LAST", name: "last.png" }],
    });
    expect(request).not.toHaveProperty("negative_prompt");
    expect(request).not.toHaveProperty("loras");
    expect(request).not.toHaveProperty("references");
  });

  it("serializes Ref2VA in exact mixed order and strips edit/endpoint fields", () => {
    const state = emptyMinimaxH3AuthoringState();
    state.references = [
      { reference: video("one.mp4") },
      { reference: image("two.png") },
      { reference: audio("three.wav") },
    ];
    const request = serializeMinimaxH3Authoring(
      {
        prompt: "new synchronized shot",
        frames: 362,
        source_image: "STALE",
        edit_images: ["STALE"],
        keyframes: [{ frame: 0, image: "STALE" }],
      },
      null,
      "minimax_h3_ref2va:official-bf16",
      state,
    );
    expect(
      (
        request as typeof request & { references: GenerationReference[] }
      ).references.map((reference) => reference.provenance?.name),
    ).toEqual(["one.mp4", "two.png", "three.wav"]);
    expect(request).not.toHaveProperty("source_image");
    expect(request).not.toHaveProperty("edit_images");
    expect(request).not.toHaveProperty("keyframes");
  });

  it("restores ordered redacted gallery provenance without pretending bytes remain", () => {
    const drafts = minimaxH3ReferenceDraftsFromMetadata([
      {
        kind: "image",
        index: 1,
        name: "subject.png",
        sha256: "c".repeat(64),
        mime_type: "image/png",
        width: 1024,
        height: 768,
        visual_rows: undefined,
      } as never,
      {
        kind: "audio",
        index: 2,
        name: "voice.wav",
        sha256: "d".repeat(64),
        mime_type: "audio/wav",
        duration_ms: 3_000,
        sample_rate: 48_000,
        channels: 2,
        sample_count: 144_000,
      },
    ]);
    expect(drafts.map((draft) => draft.reference.kind)).toEqual([
      "image",
      "audio",
    ]);
    expect(
      drafts.every((draft) => draft.reference.media.authority === "descriptor"),
    ).toBe(true);
    expect(
      minimaxH3AuthoringError("minimax-h3", "minimax-h3-ref2va:official-bf16", {
        firstFrame: null,
        lastFrame: null,
        references: drafts,
      }),
    ).toContain("Reattach reference 1");
  });

  it("uses honest semantic-resynthesis wording", () => {
    expect(MINIMAX_H3_RESYNTHESIS_GUIDANCE).toContain("newly synthesized");
    expect(MINIMAX_H3_RESYNTHESIS_GUIDANCE).toContain("not pixel-aligned");
    expect(MINIMAX_H3_RESYNTHESIS_GUIDANCE).toContain("no denoise-strength");
  });

  it("restores only an actual final-frame keyframe when frame count is known", () => {
    expect(
      minimaxH3ClosingBoundaryFromMetadata(124, [
        { frame: 0, name: "opening.png", sha256: "a".repeat(64) },
      ]),
    ).toBeNull();
    expect(
      minimaxH3ClosingBoundaryFromMetadata(124, [
        { frame: 123, name: "closing.png", sha256: "b".repeat(64) },
      ]),
    ).toMatchObject({
      filename: "closing.png",
      data: "",
      sha256: "b".repeat(64),
    });
  });
});

describe("model-switch boundary bridge", () => {
  it("builds a boundary from a staged single-source image", () => {
    expect(
      minimaxH3BoundaryFromStagedImage({
        base64: "QUJD",
        filename: "pic.png",
        width: 1024,
        height: 576,
        mime: "image/png",
        draftId: "d1",
      }),
    ).toEqual({
      filename: "pic.png",
      mimeType: "image/png",
      width: 1024,
      height: 576,
      data: "QUJD",
      sha256: null,
      draftId: "d1",
    });
  });

  it("defaults missing staged metadata instead of failing", () => {
    expect(minimaxH3BoundaryFromStagedImage({ base64: "QUJD" })).toMatchObject({
      filename: "First frame",
      mimeType: "image/*",
      width: 0,
      height: 0,
      data: "QUJD",
    });
  });

  it("refuses to build a boundary without bytes", () => {
    expect(minimaxH3BoundaryFromStagedImage({ base64: "" })).toBeNull();
    expect(minimaxH3BoundaryFromStagedImage({ base64: "   " })).toBeNull();
    expect(minimaxH3BoundaryFromStagedImage(null)).toBeNull();
  });

  it("round-trips a boundary back into a staged image", () => {
    const staged = stagedImageFromMinimaxH3Boundary({
      filename: "first.png",
      mimeType: "image/png",
      width: 1344,
      height: 768,
      data: "REVG",
      sha256: "c".repeat(64),
      draftId: "d2",
    });
    expect(staged).toEqual({
      base64: "REVG",
      filename: "first.png",
      width: 1344,
      height: 768,
      mime: "image/png",
      sha256: "c".repeat(64),
      draftId: "d2",
    });
  });

  it("never promotes a bytes-less reattach descriptor into a source well", () => {
    expect(
      stagedImageFromMinimaxH3Boundary(
        minimaxH3BoundaryFromSourceMetadata("pic.png", "a".repeat(64)),
      ),
    ).toBeNull();
    expect(stagedImageFromMinimaxH3Boundary(null)).toBeNull();
  });
});

describe("reference crop", () => {
  const CROP = { x: 256, y: 0, width: 768, height: 768 };
  const cropped = (name: string) => ({ reference: image(name), crop: CROP });
  const fitImage = vi.fn(async () => "Q1JPUFBFRA==");

  it("re-encodes a cropped image at the crop's own size with a fresh digest and crop provenance", async () => {
    fitImage.mockClear();
    const state = emptyMinimaxH3AuthoringState();
    state.references = [
      cropped("subject.png"),
      { reference: video("motion.mp4") },
      { reference: image("plain.png") },
    ];
    const next = await applyMinimaxH3ReferenceCrops(state, { fitImage });

    expect(fitImage).toHaveBeenCalledTimes(1);
    expect(fitImage).toHaveBeenCalledWith("SUBJECT.PNG", {
      outputWidth: 768,
      outputHeight: 768,
      drawWidth: 1024,
      drawHeight: 768,
      offsetX: -256,
      offsetY: 0,
      maskPaddedPixels: false,
    });
    const first = next.references[0]!;
    expect(first.crop).toBeUndefined();
    expect(first.reference).toEqual({
      kind: "image",
      media: { authority: "inline", data: "Q1JPUFBFRA==" },
      mime_type: "image/png",
      width: 768,
      height: 768,
      provenance: {
        name: "subject.png",
        sha256: await sha256PaddedBase64("Q1JPUFBFRA=="),
        crop: {
          ...CROP,
          source_width: 1024,
          source_height: 768,
          source_sha256: "subject.png".repeat(64).slice(0, 64),
        },
      },
    });
    // Untouched neighbours are the same objects, in the same order.
    expect(next.references[1]!.reference).toBe(state.references[1]!.reference);
    expect(next.references[2]!.reference).toBe(state.references[2]!.reference);
    // The live state was never mutated.
    expect(state.references[0]!.reference).toMatchObject({ width: 1024 });
  });

  it("leaves an identity crop's bytes untouched and records no crop provenance", async () => {
    fitImage.mockClear();
    const state = emptyMinimaxH3AuthoringState();
    state.references = [
      {
        reference: image("whole.png"),
        crop: { x: 0, y: 0, width: 1024, height: 768 },
      },
    ];
    const next = await applyMinimaxH3ReferenceCrops(state, { fitImage });
    expect(fitImage).not.toHaveBeenCalled();
    expect(next.references[0]!.reference).toBe(state.references[0]!.reference);
    expect(next.references[0]!.crop).toBeUndefined();
  });

  it("refuses to crop a reference whose bytes are not attached, naming its position", async () => {
    const state = emptyMinimaxH3AuthoringState();
    state.references = [
      { reference: video("one.mp4") },
      {
        reference: {
          ...image("gone.png"),
          media: { authority: "descriptor" },
        },
        crop: CROP,
      },
    ];
    await expect(
      applyMinimaxH3ReferenceCrops(state, { fitImage }),
    ).rejects.toThrow(/Reference 2/);
  });

  it("serializes a pending crop into the projection so conditioning fingerprints stale on it", () => {
    const state = emptyMinimaxH3AuthoringState();
    state.references = [cropped("subject.png")];
    const request = serializeMinimaxH3Authoring(
      { frames: 124 },
      "minimax-h3",
      "minimax-h3-ref2va:comfy-pruned-int8",
      state,
    ) as unknown as { references: GenerationReference[] };
    expect(request.references[0]!.provenance?.crop).toEqual({
      ...CROP,
      source_width: 1024,
      source_height: 768,
      source_sha256: "subject.png".repeat(64).slice(0, 64),
    });
    // The projection's bytes are still the source's: only the applied crop
    // rewrites them, and the server refuses a size mismatch by name.
    expect(request.references[0]!.media).toEqual({
      authority: "inline",
      data: "SUBJECT.PNG",
    });
  });

  it("restores a saved crop as a pending draft crop over the uncropped source facts", () => {
    const [draft] = minimaxH3ReferenceDraftsFromMetadata([
      {
        kind: "image",
        index: 1,
        name: "subject.png",
        sha256: "c".repeat(64),
        mime_type: "image/png",
        width: 768,
        height: 768,
        crop: {
          ...CROP,
          source_width: 1024,
          source_height: 768,
          source_sha256: "d".repeat(64),
        },
      },
    ]);
    expect(draft!.crop).toEqual(CROP);
    expect(draft!.reference).toMatchObject({
      kind: "image",
      media: { authority: "descriptor" },
      width: 1024,
      height: 768,
      provenance: { name: "subject.png", sha256: "d".repeat(64) },
    });
    expect(draft!.reference.provenance).not.toHaveProperty("crop");
  });

  it("re-applies the crop only when the reattached original is byte-identical", () => {
    const state = emptyMinimaxH3AuthoringState();
    state.references = [
      {
        reference: {
          ...image("subject.png"),
          media: { authority: "descriptor" },
        },
        crop: CROP,
      },
    ];
    const same = reattachMinimaxH3Reference(state, 0, {
      reference: image("subject.png"),
    });
    expect(same.notice).toBeNull();
    expect(same.state.references[0]).toEqual({
      reference: image("subject.png"),
      crop: CROP,
    });

    const other = reattachMinimaxH3Reference(state, 0, {
      reference: image("other.png"),
    });
    expect(other.state.references[0]).toEqual({
      reference: image("other.png"),
    });
    expect(other.notice).toContain("crop");

    expect(() =>
      reattachMinimaxH3Reference(state, 0, { reference: video("clip.mp4") }),
    ).toThrow(/as image/);
  });

  it("sets and clears a draft crop, and the byte-free projection keeps it", () => {
    const state = emptyMinimaxH3AuthoringState();
    state.references = [{ reference: image("subject.png") }];
    const set = setMinimaxH3ReferenceCrop(state, 0, {
      x: 0.4,
      y: 0,
      width: 900.2,
      height: 768,
    });
    expect(set.references[0]!.crop).toEqual({
      x: 0,
      y: 0,
      width: 900,
      height: 768,
    });
    expect(stripMinimaxH3AuthoringMedia(set).references[0]).toMatchObject({
      crop: { x: 0, y: 0, width: 900, height: 768 },
      reference: { media: { authority: "descriptor" } },
    });
    const identity = setMinimaxH3ReferenceCrop(set, 0, {
      x: 0,
      y: 0,
      width: 1024,
      height: 768,
    });
    expect(identity.references[0]).not.toHaveProperty("crop");
    expect(
      setMinimaxH3ReferenceCrop(set, 0, null).references[0],
    ).not.toHaveProperty("crop");
  });
});
