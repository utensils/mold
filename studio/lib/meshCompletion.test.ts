import { describe, expect, it } from "vitest";
import { isAudioCompletion } from "./ltx2Pipeline";
import {
  isMeshArtifact,
  isMeshCompletion,
  type MeshArtifactProbe,
} from "./meshCompletion";

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

describe("isMeshArtifact", () => {
  it("accepts a live completion the server reported mesh facts for", () => {
    expect(isMeshArtifact({ mesh_vertices: 24_576, format: "glb" })).toBe(true);
  });

  it("accepts a durable completion that names only the glTF container", () => {
    // A durable batch child reports a filename and a container, never counts.
    expect(isMeshArtifact({ format: "glb" })).toBe(true);
    expect(isMeshArtifact({ format: "GLB" })).toBe(true);
  });

  it("leaves every raster, video and audio container on its own arm", () => {
    expect(isMeshArtifact({ format: "png" })).toBe(false);
    expect(
      isMeshArtifact({ format: "mp4", video_frames: 97 } as MeshArtifactProbe),
    ).toBe(false);
    expect(
      isMeshArtifact({
        format: "wav",
        audio_sample_rate: 48_000,
      } as MeshArtifactProbe),
    ).toBe(false);
    expect(isMeshArtifact({})).toBe(false);
    expect(isMeshArtifact(null)).toBe(false);
    expect(isMeshArtifact(undefined)).toBe(false);
  });
});
