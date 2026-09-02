import { describe, expect, it } from "vitest";
import {
  GLB_MIME_TYPE,
  isAnimatedMeshExport,
  meshExportFilename,
  splitMeshExportFormats,
} from "./meshExport";

/**
 * The one mesh export menu policy for web, desktop and the phone: the host's
 * advertised list is split into one-click geometry transcodes and the
 * animated turntables that share the video export sheet, and the stored GLB
 * is never offered as an "export" beside Download.
 */
describe("isAnimatedMeshExport", () => {
  it("names the turntable containers, whatever their case", () => {
    expect(["gif", "APNG", " webp "].map(isAnimatedMeshExport)).toEqual([
      true,
      true,
      true,
    ]);
  });

  it("treats every geometry container as a direct transcode", () => {
    expect(["obj", "stl", "ply", "usdz"].map(isAnimatedMeshExport)).toEqual([
      false,
      false,
      false,
      false,
    ]);
  });
});

describe("splitMeshExportFormats", () => {
  it("splits the host's list into geometry files and turntables, in its order", () => {
    expect(
      splitMeshExportFormats(["obj", "stl", "ply", "gif", "apng", "webp"]),
    ).toEqual({
      files: ["obj", "stl", "ply"],
      animations: ["gif", "apng", "webp"],
    });
  });

  // The server lists the stored container first so a client can see what it
  // holds; no menu should offer "Export as GLB" beside Download.
  it("drops glb, the stored form, from both halves", () => {
    expect(splitMeshExportFormats(["glb", "obj", "GLB", "gif"])).toEqual({
      files: ["obj"],
      animations: ["gif"],
    });
  });

  it("keeps a container this client has never heard of as a direct transcode", () => {
    expect(splitMeshExportFormats(["obj", "usdz"]).files).toEqual([
      "obj",
      "usdz",
    ]);
  });

  it("answers two empty lists for a host that advertises nothing", () => {
    expect(splitMeshExportFormats(undefined)).toEqual({
      files: [],
      animations: [],
    });
    expect(splitMeshExportFormats(null)).toEqual({
      files: [],
      animations: [],
    });
    expect(splitMeshExportFormats([])).toEqual({ files: [], animations: [] });
  });

  it("normalises the advertised spelling to lower case", () => {
    expect(splitMeshExportFormats(["OBJ", "Gif"])).toEqual({
      files: ["obj"],
      animations: ["gif"],
    });
  });
});

describe("meshExportFilename", () => {
  it("keeps the print's own stem and takes the requested extension", () => {
    expect(meshExportFilename("armchair 01.glb", "obj")).toBe(
      "armchair 01.obj",
    );
    expect(meshExportFilename("armchair__hunyuan3d__s7.glb", "stl")).toBe(
      "armchair__hunyuan3d__s7.stl",
    );
  });

  it("lower-cases the extension whatever the host's spelling", () => {
    expect(meshExportFilename("armchair 01.glb", "GIF")).toBe(
      "armchair 01.gif",
    );
  });

  it("falls back to a stem when the name has none", () => {
    expect(meshExportFilename("", "stl")).toBe("mold-mesh.stl");
    expect(meshExportFilename(".glb", "stl")).toBe("mold-mesh.stl");
  });
});

describe("GLB_MIME_TYPE", () => {
  it("is the binary glTF registration", () => {
    expect(GLB_MIME_TYPE).toBe("model/gltf-binary");
  });
});
