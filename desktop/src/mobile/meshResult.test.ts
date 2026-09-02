import { describe, expect, it } from "vitest";
import type { CompleteEvent } from "../lib/api/types";
import {
  GLB_MIME_TYPE,
  isAnimatedMeshExportFormat,
  isMobileMeshResult,
  meshExportFilename,
  meshResultBlob,
} from "./meshResult";

function completion(overrides: Partial<CompleteEvent> = {}): CompleteEvent {
  return {
    image: "",
    format: "png",
    width: 1024,
    height: 1024,
    seed_used: 7,
    generation_time_ms: 100,
    model: "sdxl:test",
    ...overrides,
  };
}

describe("isMobileMeshResult", () => {
  it("recognizes an inline completion by its vertex count", () => {
    expect(
      isMobileMeshResult(completion({ format: "glb", mesh_vertices: 24_576, mesh_faces: 49_152 })),
    ).toBe(true);
  });

  /**
   * A durable completion is synthesized from the byte-free presentation the
   * phone persisted at submit time: it carries the request's container and
   * nothing else, so keying only on `mesh_vertices` would draw glTF bytes
   * into an `<img>`.
   */
  it("recognizes a durable completion by its container alone", () => {
    expect(isMobileMeshResult(completion({ format: "glb" }))).toBe(true);
  });

  it("leaves every raster, clip, and audio completion alone", () => {
    expect(isMobileMeshResult(completion())).toBe(false);
    expect(isMobileMeshResult(completion({ format: "mp4", video_frames: 97 }))).toBe(false);
    expect(isMobileMeshResult(completion({ format: "wav", audio_sample_rate: 48_000 }))).toBe(
      false,
    );
    expect(isMobileMeshResult(null)).toBe(false);
  });
});

describe("meshResultBlob", () => {
  it("decodes inline glTF bytes under the binary glTF media type", async () => {
    // "glTF" — the container's own magic.
    const blob = meshResultBlob(btoa("glTF"))!;
    expect(blob.type).toBe(GLB_MIME_TYPE);
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(
      Uint8Array.from([0x67, 0x6c, 0x54, 0x46]),
    );
  });

  it("answers null for a metadata-only completion and for corrupt bytes", () => {
    expect(meshResultBlob("")).toBeNull();
    expect(meshResultBlob(null)).toBeNull();
    expect(meshResultBlob("not base64 !!!")).toBeNull();
  });
});

describe("mesh export naming", () => {
  it("splits the advertised list into geometry files and turntables", () => {
    expect(["obj", "stl", "ply"].map(isAnimatedMeshExportFormat)).toEqual([false, false, false]);
    expect(["gif", "APNG", "webp"].map(isAnimatedMeshExportFormat)).toEqual([true, true, true]);
  });

  it("keeps the print's own stem and takes the requested extension", () => {
    expect(meshExportFilename("armchair 01.glb", "obj")).toBe("armchair 01.obj");
    expect(meshExportFilename("armchair 01.glb", "GIF")).toBe("armchair 01.gif");
    expect(meshExportFilename(".glb", "stl")).toBe("mold-mesh.stl");
  });
});
