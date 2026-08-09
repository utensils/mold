import { describe, expect, it } from "vitest";
import type { GenerationReference } from "./generationReferences";
import {
  MINIMAX_H3_FIXED_FPS,
  MINIMAX_H3_REF2VA_COMFY,
  MINIMAX_H3_RESYNTHESIS_GUIDANCE,
  appendMinimaxH3GalleryImageReference,
  canonicalMinimaxH3ModelName,
  emptyMinimaxH3AuthoringState,
  isMinimaxH3Family,
  isMinimaxH3Identity,
  minimaxH3AuthoringCapabilities,
  minimaxH3AuthoringError,
  minimaxH3BoundaryFromSourceMetadata,
  minimaxH3ClosingBoundaryFromMetadata,
  minimaxH3Mode,
  minimaxH3ReferenceBudget,
  minimaxH3ReferenceDraftsFromMetadata,
  minimaxH3TaskForModel,
  moveMinimaxH3Reference,
  serializeMinimaxH3Authoring,
  setMinimaxH3GalleryImageFirstFrame,
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
  it("resolves only the released explicit task partitions", () => {
    expect(isMinimaxH3Family(" MiniMax_H3 ")).toBe(true);
    expect(minimaxH3TaskForModel("minimax-h3-fl2va:official-bf16")).toBe(
      "fl2va",
    );
    expect(minimaxH3TaskForModel("minimax_h3_ref2va:comfy-pruned-int8")).toBe(
      "ref2va",
    );
    expect(minimaxH3TaskForModel("hf:opaque")).toBeNull();
    expect(isMinimaxH3Identity(null, "minimax_h3_ref2va")).toBe(true);
    expect(isMinimaxH3Identity(null, "hf:opaque")).toBe(false);
    expect(isMinimaxH3Identity(null, "minimax-h3-ref2va:future")).toBe(true);
    expect(canonicalMinimaxH3ModelName(" MiniMax_H3_Ref2VA ")).toBe(
      MINIMAX_H3_REF2VA_COMFY,
    );
    expect(canonicalMinimaxH3ModelName("minimax-h3-ref2va:future")).toBeNull();
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

  it("appends a gallery image as the final ordered inline Ref2VA reference", () => {
    const state = emptyMinimaxH3AuthoringState();
    const original = { reference: video("motion.mp4") };
    state.references = [original];

    const result = appendMinimaxH3GalleryImageReference(state, {
      filename: " subject.png ",
      mimeType: "image/png; charset=binary",
      width: 1280,
      height: 720,
      data: "SUBJECT",
    });

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.reference).toBe(2);
    expect(result.state.references[0]).toBe(original);
    expect(result.state.references[1]?.reference).toEqual({
      kind: "image",
      media: { authority: "inline", data: "SUBJECT" },
      provenance: { name: "subject.png" },
      mime_type: "image/png",
      width: 1280,
      height: 720,
    });
  });

  it("enforces gallery reference limits and maps FL2VA source to its first frame", () => {
    const images = Array.from({ length: 9 }, (_, index) => ({
      reference: image(`${index}.png`),
    }));
    const rejected = appendMinimaxH3GalleryImageReference(
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
