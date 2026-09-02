import { describe, expect, it } from "vitest";
import {
  meshAnimationExportFormats,
  meshExportFilename,
  meshFileExportFormats,
} from "./meshExport";

describe("mesh export menu", () => {
  it("splits the host's advertised list into transcodes and turntables", () => {
    const advertised = ["obj", "stl", "ply", "gif", "apng", "webp"];
    expect(meshFileExportFormats(advertised)).toEqual(["obj", "stl", "ply"]);
    expect(meshAnimationExportFormats(advertised)).toEqual(["gif", "apng", "webp"]);
  });

  it("never offers the stored glb as an export", () => {
    expect(meshFileExportFormats(["glb", "obj"])).toEqual(["obj"]);
    expect(meshAnimationExportFormats(["glb", "gif"])).toEqual(["gif"]);
  });

  it("keeps a container this client has never heard of as a direct transcode", () => {
    expect(meshFileExportFormats(["obj", "usdz"])).toEqual(["obj", "usdz"]);
  });

  it("offers nothing for a host that advertises no mesh exports", () => {
    expect(meshFileExportFormats(undefined)).toEqual([]);
    expect(meshAnimationExportFormats(null)).toEqual([]);
  });

  it("swaps the container on the print's own save name", () => {
    expect(meshExportFilename("armchair__hunyuan3d__s7.glb", "obj")).toBe(
      "armchair__hunyuan3d__s7.obj",
    );
    expect(meshExportFilename("", "stl")).toBe("mold-mesh.stl");
  });
});
