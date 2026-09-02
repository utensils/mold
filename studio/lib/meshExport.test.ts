import { describe, expect, it } from "vitest";
import {
  GLB_MIME_TYPE,
  isAnimatedMeshExport,
  meshExportDimensionsMm,
  meshExportFilename,
  meshExportRequest,
  meshExportSizeLabel,
  meshGeometryDefaults,
  splitMeshExportFormats,
  takesGeometryOptions,
  type MeshExportGeometryCapabilities,
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

// ── Geometry options ───────────────────────────────────────────────────────

const CAPABILITIES: MeshExportGeometryCapabilities = {
  size_mm: { min: 1, max: 1000, default: 100 },
  up_axes: ["y", "z"],
  origins: ["center", "floor"],
  defaults: {
    obj: { size_mm: null, up_axis: "y", origin: "floor" },
    stl: { size_mm: 100, up_axis: "z", origin: "floor" },
    ply: { size_mm: 100, up_axis: "z", origin: "floor" },
  },
};

/** A unit-ish box in the stored Y-up frame: 1 wide, 0.4286 tall, 0.6857 deep. */
const BOUNDS = { min: [-0.5, -0.2143, -0.3429], max: [0.5, 0.2143, 0.3428] };

describe("takesGeometryOptions", () => {
  it("names the geometry containers, whatever their case", () => {
    expect(["obj", "STL", " ply "].map(takesGeometryOptions)).toEqual([
      true,
      true,
      true,
    ]);
  });

  it("excludes the stored form and every turntable", () => {
    expect(
      ["glb", "GLB", "gif", "apng", "webp"].map(takesGeometryOptions),
    ).toEqual([false, false, false, false, false]);
  });

  // The host's own defaults table decides what it will actually scale; this
  // structural rule must not hide a container a future host adds.
  it("stays permissive about a container this client has never heard of", () => {
    expect(takesGeometryOptions("usdz")).toBe(true);
  });
});

describe("meshGeometryDefaults", () => {
  it("reads the host's per-format defaults", () => {
    expect(meshGeometryDefaults(CAPABILITIES, "stl")).toEqual({
      size_mm: 100,
      up_axis: "z",
      origin: "floor",
    });
    expect(meshGeometryDefaults(CAPABILITIES, "OBJ")).toEqual({
      size_mm: null,
      up_axis: "y",
      origin: "floor",
    });
  });

  // The presence of the block is the ONLY gate. An older server drops the
  // three keys instead of refusing them, so a client that guessed defaults
  // would promise a resize the host never performed.
  it("answers null for a host that advertises no geometry block", () => {
    expect(meshGeometryDefaults(null, "stl")).toBeNull();
    expect(meshGeometryDefaults(undefined, "stl")).toBeNull();
  });

  it("answers null for the stored form and the turntables", () => {
    expect(meshGeometryDefaults(CAPABILITIES, "glb")).toBeNull();
    expect(meshGeometryDefaults(CAPABILITIES, "gif")).toBeNull();
  });

  it("answers null for a container the host does not list", () => {
    expect(meshGeometryDefaults(CAPABILITIES, "usdz")).toBeNull();
  });

  it("clamps a default size into the host's own bounds", () => {
    const tight: MeshExportGeometryCapabilities = {
      ...CAPABILITIES,
      size_mm: { min: 10, max: 50, default: 50 },
    };
    expect(meshGeometryDefaults(tight, "stl")?.size_mm).toBe(50);
  });

  it("falls back to the first advertised axis and origin", () => {
    const narrow: MeshExportGeometryCapabilities = {
      ...CAPABILITIES,
      up_axes: ["y"],
      origins: ["center"],
    };
    expect(meshGeometryDefaults(narrow, "stl")).toEqual({
      size_mm: 100,
      up_axis: "y",
      origin: "center",
    });
  });
});

describe("meshExportDimensionsMm", () => {
  it("scales the longest stored extent to the requested size", () => {
    const dimensions = meshExportDimensionsMm(BOUNDS, 100, "y")!;
    expect(dimensions[0]).toBeCloseTo(100, 4);
    expect(dimensions[1]).toBeCloseTo(42.86, 1);
    expect(dimensions[2]).toBeCloseTo(68.57, 1);
  });

  // A Z-up file rotates (x, y, z) into (x, -z, y), so its own axes read
  // width, then depth, then height.
  it("reorders the extents for a z-up export", () => {
    const yUp = meshExportDimensionsMm(BOUNDS, 100, "y")!;
    const zUp = meshExportDimensionsMm(BOUNDS, 100, "z")!;
    expect(zUp[0]).toBeCloseTo(yUp[0], 6);
    expect(zUp[1]).toBeCloseTo(yUp[2], 6);
    expect(zUp[2]).toBeCloseTo(yUp[1], 6);
  });

  it("writes model units verbatim when no size is asked for", () => {
    const dimensions = meshExportDimensionsMm(BOUNDS, null, "y")!;
    expect(dimensions[0]).toBeCloseTo(1, 6);
    expect(dimensions[1]).toBeCloseTo(0.4286, 3);
  });

  it("answers null when the viewer has reported no box", () => {
    expect(meshExportDimensionsMm(null, 100, "z")).toBeNull();
    expect(meshExportDimensionsMm(undefined, 100, "z")).toBeNull();
    expect(
      meshExportDimensionsMm({ min: [0, 0], max: [1, 1] }, 100, "z"),
    ).toBeNull();
  });

  it("answers null for a degenerate box rather than dividing by zero", () => {
    expect(
      meshExportDimensionsMm({ min: [1, 1, 1], max: [1, 1, 1] }, 100, "z"),
    ).toBeNull();
  });
});

describe("meshExportSizeLabel", () => {
  it("names all three extents in millimetres", () => {
    expect(
      meshExportSizeLabel(BOUNDS, {
        size_mm: 100,
        up_axis: "z",
        origin: "floor",
      }),
    ).toBe("100.0 × 68.6 × 42.9 mm");
  });

  it("names the model units when the mesh is written as stored", () => {
    expect(
      meshExportSizeLabel(BOUNDS, {
        size_mm: null,
        up_axis: "z",
        origin: "floor",
      }),
    ).toBe("as stored (1.00 × 0.69 × 0.43)");
  });

  it("still names the knob when the viewer has reported no box", () => {
    expect(
      meshExportSizeLabel(null, {
        size_mm: 120,
        up_axis: "z",
        origin: "floor",
      }),
    ).toBe("longest side 120 mm");
    expect(
      meshExportSizeLabel(null, {
        size_mm: null,
        up_axis: "y",
        origin: "floor",
      }),
    ).toBe("as stored");
  });
});

describe("meshExportRequest", () => {
  it("sends the three keys the host advertised", () => {
    expect(
      meshExportRequest("stl", {
        size_mm: 120,
        up_axis: "y",
        origin: "center",
      }),
    ).toEqual({ format: "stl", size_mm: 120, up_axis: "y", origin: "center" });
  });

  // The wire has no way to ask a size-defaulting format to skip scaling, so
  // "as stored" is simply the absent key, and it is only ever offered for a
  // format whose own default is already null.
  it("omits size_mm for an as-stored export", () => {
    expect(
      meshExportRequest("obj", {
        size_mm: null,
        up_axis: "y",
        origin: "floor",
      }),
    ).toEqual({ format: "obj", up_axis: "y", origin: "floor" });
  });

  // The old-server shape, byte for byte: an older host drops unknown keys.
  it("sends the bare format when the host advertised no geometry block", () => {
    expect(meshExportRequest("stl", null)).toEqual({ format: "stl" });
    expect(Object.keys(meshExportRequest("stl", null))).toEqual(["format"]);
  });
});
