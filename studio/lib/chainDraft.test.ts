import { describe, expect, it } from "vitest";
import {
  buildClipsRequest,
  editClipOrder,
  chainInputRefusal,
} from "./chainDraft";

const base = {
  model: "test",
  prompt: "First",
  frames: 97,
  width: 384,
  height: 256,
  fps: 24,
  steps: 8,
  guidance: 1,
};
describe("authored video clips", () => {
  it("preserves ordered visible prompts and explicitly creates durable authored work", () => {
    const request = buildClipsRequest(
      base,
      [
        { id: "a", prompt: "First", frames: 97 },
        { id: "b", prompt: "Second", frames: 49 },
      ],
      17,
    );
    expect(request.stages.map((clip) => clip.prompt)).toEqual([
      "First",
      "Second",
    ]);
    expect(request).toMatchObject({
      output_mode: "sequence",
      motion_tail_frames: 17,
    });
    expect(request).not.toHaveProperty("ephemeral");
  });
  it("refuses unsafe numeric seeds before JSON can silently round them", () => {
    expect(
      chainInputRefusal({ ...base, seed: Number.MAX_SAFE_INTEGER + 1 }),
    ).toContain("seed");
    expect(chainInputRefusal({ ...base, seed: -1 })).toContain("seed");
    expect(
      chainInputRefusal({ ...base, seed: Number.MAX_SAFE_INTEGER }),
    ).toBeNull();
  });
  it("moves stable rows without losing authored fields", () => {
    const clips = [
      { id: "a", prompt: "First", frames: 97 },
      { id: "b", prompt: "Second", frames: 49 },
    ];
    expect(editClipOrder(clips, "b", -1)).toEqual([clips[1], clips[0]]);
    expect(editClipOrder(clips, "a", -1)).toEqual(clips);
  });
  it("keeps typed references on every independently conditioned clip and endpoints on their authored clip", () => {
    const references = [
      {
        kind: "image" as const,
        width: 32,
        height: 32,
        mime_type: "image/png",
        media: { authority: "inline" as const, data: "abc" },
      },
    ];
    const request = buildClipsRequest(
      { ...base, references },
      [
        { id: "a", prompt: "One", frames: 97 },
        { id: "b", prompt: "Two", frames: 97 },
      ],
      0,
    );
    expect(
      request.stages.every((clip) => clip.generation_references === references),
    ).toBe(true);
  });
  it("refuses fields the chain cannot carry instead of silently dropping them", () => {
    expect(chainInputRefusal({ ...base, mask_image: "mask" })).toContain(
      "mask_image",
    );
    expect(chainInputRefusal(base)).toBeNull();
  });
  it("puts the first frame on clip one and the final frame on the final clip's own boundary", () => {
    const result = buildClipsRequest(
      {
        ...base,
        keyframes: [
          { frame: 0, image: "first" },
          { frame: 96, image: "last" },
        ],
      },
      [
        { id: "a", prompt: "One", frames: 97 },
        { id: "b", prompt: "Two", frames: 49 },
      ],
      1,
    );
    expect(result.stages[0]?.keyframes).toEqual([{ frame: 0, image: "first" }]);
    expect(result.stages[1]?.keyframes).toEqual([{ frame: 48, image: "last" }]);
  });
  it("does not replace a first endpoint in a linked continuation", () => {
    expect(() =>
      buildClipsRequest(
        { ...base, keyframes: [{ frame: 0, image: "x" }] },
        [{ id: "a", prompt: "One", frames: 97 }],
        1,
        { job_id: "old", finalize_take: 2 },
      ),
    ).toThrow(/first frame/i);
  });
  it("rejects transient reference authorities and aggregate repeated media before stage expansion", () => {
    const references = [
      {
        kind: "image" as const,
        width: 32,
        height: 32,
        mime_type: "image/png",
        media: { authority: "upload" as const, handle: "transient" },
      },
    ];
    expect(() =>
      buildClipsRequest(
        { ...base, references },
        [{ id: "a", prompt: "One", frames: 97 }],
        0,
      ),
    ).toThrow(/inline|original/i);
    const large = "A".repeat(24 * 1024 * 1024);
    const inline = [
      {
        ...references[0],
        media: { authority: "inline" as const, data: large },
      },
    ];
    expect(() =>
      buildClipsRequest(
        { ...base, references: inline },
        [
          { id: "a", prompt: "One", frames: 97 },
          { id: "b", prompt: "Two", frames: 97 },
        ],
        0,
      ),
    ).toThrow(/32 MiB/);
  });
  it("preserves the source-fit policy with already fitted opening pixels", () => {
    const source_fit = { mode: "crop-fill" as const };
    const result = buildClipsRequest(
      { ...base, source_image: "AAAA", source_fit },
      [{ id: "a", prompt: "One", frames: 97 }],
      1,
    );
    expect(result.source_fit).toEqual(source_fit);
    expect(result.stages[0]?.source_image).toBe("AAAA");
  });
});

it("preserves restored per-clip controls and inputs while editing prompts and endpoint length", () => {
  const saved = {
    prompt: "old",
    frames: 97,
    transition: "fade" as const,
    fade_frames: 8,
    seed_offset: "18446744073709551615",
    loras: [{ path: "adapter", scale: 0.75 }],
    keyframes: [{ frame: 96, image: "YQ==" }],
    source_image: "Yg==",
  };
  const r = buildClipsRequest(
    { ...base, seed: "18446744073709551615" },
    [{ id: "saved", prompt: " literal\\n ", frames: 121, restored: saved }],
    17,
  );
  expect(r.stages[0]).toEqual({
    ...saved,
    prompt: " literal\\n ",
    frames: 121,
    keyframes: [{ frame: 120, image: "YQ==" }],
  });
  expect(r.seed).toBe("18446744073709551615");
  expect(() =>
    buildClipsRequest(
      base,
      [
        {
          id: "saved",
          prompt: "new",
          frames: 10,
          restored: { ...saved, keyframes: [{ frame: 20, image: "YQ==" }] },
        },
      ],
      17,
    ),
  ).toThrow(/keyframe/);
});
