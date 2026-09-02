import { describe, expect, it } from "vitest";
import { isAudioCompletion } from "./ltx2Pipeline";
import { isMeshCompletion } from "./meshCompletion";

describe("isMeshCompletion", () => {
  it("keys on the vertex count, the server's own mesh marker", () => {
    expect(isMeshCompletion({ mesh_vertices: 24_576 })).toBe(true);
    expect(isMeshCompletion({ mesh_vertices: 0 })).toBe(true);
  });

  it("is false for raster, video, and audio completions", () => {
    expect(isMeshCompletion({})).toBe(false);
    expect(isMeshCompletion({ mesh_vertices: null })).toBe(false);
    expect(
      isMeshCompletion({ video_frames: 97 } as Record<string, unknown>),
    ).toBe(false);
    expect(
      isMeshCompletion({ audio_sample_rate: 48_000 } as Record<
        string,
        unknown
      >),
    ).toBe(false);
    expect(isMeshCompletion(null)).toBe(false);
    expect(isMeshCompletion(undefined)).toBe(false);
  });

  it("is disjoint from the audio probe, so probing mesh first is safe", () => {
    const mesh = { mesh_vertices: 12, audio_sample_rate: null };
    expect(isMeshCompletion(mesh)).toBe(true);
    expect(isAudioCompletion(mesh)).toBe(false);
  });
});
