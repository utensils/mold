import { describe, expect, it } from "vitest";
import { inferFormatFromName, mediaKind } from "./types";
import type { ChainJobDetail, ChainJobEvent } from "./types";

describe("inferFormatFromName", () => {
  it.each([
    ["cat.png", "png"],
    ["CAT.PNG", "png"],
    ["shot.jpg", "jpeg"],
    ["shot.jpeg", "jpeg"],
    ["loop.gif", "gif"],
    ["loop.apng", "apng"],
    ["loop.webp", "webp"],
    ["clip.mp4", "mp4"],
  ] as const)("maps %s → %s", (name, expected) => {
    expect(inferFormatFromName(name)).toBe(expected);
  });

  it("returns null for unrecognized extensions", () => {
    expect(inferFormatFromName("something.txt")).toBeNull();
    expect(inferFormatFromName("noext")).toBeNull();
    expect(inferFormatFromName("")).toBeNull();
  });
});

describe("mediaKind", () => {
  it("uses the explicit format when provided", () => {
    expect(mediaKind("mp4", "anything")).toBe("video");
    expect(mediaKind("gif", "anything")).toBe("animated");
    expect(mediaKind("png", "anything")).toBe("image");
  });

  it("falls back to the filename extension when format is null/undefined", () => {
    expect(mediaKind(null, "clip.mp4")).toBe("video");
    expect(mediaKind(undefined, "loop.webp")).toBe("animated");
    expect(mediaKind(null, "still.jpg")).toBe("image");
  });

  it("defaults to 'image' when nothing identifies the media type", () => {
    expect(mediaKind(null, "mystery")).toBe("image");
  });
});

function detailFixture(): ChainJobDetail {
  return {
    id: "job-1",
    state: "running",
    model: "ltx-2-19b-distilled:fp8",
    stage_count: 1,
    current_stage: 0,
    created_at_unix_ms: 1,
    updated_at_unix_ms: 2,
    error: null,
    ephemeral: false,
    stages: [
      {
        idx: 0,
        state: "pending",
        seed: "42",
        frames_emitted: null,
        generation_time_ms: null,
        has_preview: false,
        error: null,
      },
    ],
    finalizes: [],
    retakes: [],
    script: {
      schema: "mold.chain.v1",
      chain: { model: "ltx-2-19b-distilled:fp8" },
      stage: [{ prompt: "stage zero", frames: 97 }],
    },
  };
}

describe("ChainJobEvent serde-tag fixtures", () => {
  it("mirrors one fixture per Rust variant", () => {
    const events: ChainJobEvent[] = [
      { type: "snapshot", job: detailFixture() },
      { type: "stage_start", stage_idx: 2 },
      { type: "denoise_step", stage_idx: 2, step: 3, total: 8 },
      {
        type: "stage_done",
        stage_idx: 2,
        frames_emitted: 97,
        has_preview: true,
      },
      { type: "yielded", pending_small_jobs: 4 },
      { type: "finalizing", total_frames: 194 },
      { type: "finalized", output: "final/output-1.mp4", take: 1 },
      { type: "state_changed", state: "completed", error: null },
    ];

    expect(events.map((event) => event.type)).toEqual([
      "snapshot",
      "stage_start",
      "denoise_step",
      "stage_done",
      "yielded",
      "finalizing",
      "finalized",
      "state_changed",
    ]);
    expect(JSON.parse(JSON.stringify(events[3]))).toEqual({
      type: "stage_done",
      stage_idx: 2,
      frames_emitted: 97,
      has_preview: true,
    });
  });
});
