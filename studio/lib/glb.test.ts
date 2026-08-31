import { describe, expect, it } from "vitest";
import { GlbParseError, parseGlb, type ParsedMesh } from "./glb";
import {
  COMPONENT_UNSIGNED_BYTE,
  COMPONENT_UNSIGNED_INT,
  COMPONENT_UNSIGNED_SHORT,
  TRIANGLE,
  assemble,
  buildDocument,
  triangleGlb,
  type MeshSpec,
} from "./glbFixture";

function triple(data: Float32Array, index: number): [number, number, number] {
  return [
    data[index * 3] ?? NaN,
    data[index * 3 + 1] ?? NaN,
    data[index * 3 + 2] ?? NaN,
  ];
}

describe("parseGlb — a well-formed mold mesh", () => {
  let mesh: ParsedMesh;

  it("reads positions, indices, bounds and counts exactly", () => {
    mesh = parseGlb(triangleGlb());
    expect(Array.from(mesh.positions)).toEqual(TRIANGLE.positions);
    expect(Array.from(mesh.indices)).toEqual([0, 1, 2]);
    expect(mesh.bounds).toEqual({ min: [0, 0, -1], max: [2, 4, 0] });
    expect(mesh.vertexCount).toBe(3);
    expect(mesh.triangleCount).toBe(1);
  });

  it("reports absent optional attributes as null", () => {
    const parsed = parseGlb(triangleGlb());
    expect(parsed.uvs).toBeNull();
    expect(parsed.colors).toBeNull();
    expect(parsed.baseColorTexture).toBeNull();
  });

  it("reads UVs, vertex colors and supplied normals when present", () => {
    const { json, bin } = buildDocument({
      ...TRIANGLE,
      uvs: [0, 0, 1, 0, 0, 1],
      colors: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      normals: [0, 0, 1, 0, 0, 1, 0, 0, 1],
    });
    const parsed = parseGlb(assemble(json, bin));
    expect(Array.from(parsed.uvs ?? [])).toEqual([0, 0, 1, 0, 0, 1]);
    expect(Array.from(parsed.colors ?? [])).toEqual([
      1, 0, 0, 0, 1, 0, 0, 0, 1,
    ]);
    expect(Array.from(parsed.normals ?? [])).toEqual([
      0, 0, 1, 0, 0, 1, 0, 0, 1,
    ]);
  });

  it("surfaces an embedded baseColor image as a typed Blob", async () => {
    const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3]);
    const { json, bin } = buildDocument({
      ...TRIANGLE,
      uvs: [0, 0, 1, 0, 0, 1],
      png,
    });
    const texture = parseGlb(assemble(json, bin)).baseColorTexture;
    expect(texture).toBeInstanceOf(Blob);
    expect(texture?.type).toBe("image/png");
    const bytes = new Uint8Array(
      (await texture?.arrayBuffer()) ?? new ArrayBuffer(0),
    );
    expect(Array.from(bytes)).toEqual(Array.from(png));
  });
});

describe("parseGlb — malformed containers", () => {
  it("rejects a buffer too small to hold a header", () => {
    expect(() => parseGlb(new ArrayBuffer(8))).toThrow(/12-byte header/);
  });

  it("rejects bad magic, naming the magic", () => {
    expect(() => parseGlb(triangleGlb({ magic: "GLTF" }))).toThrow(
      /bad magic/i,
    );
  });

  it("rejects glTF 1, naming the version", () => {
    expect(() => parseGlb(triangleGlb({ version: 1 }))).toThrow(
      /unsupported GLB version 1/,
    );
  });

  it("rejects a declared length that disagrees with the buffer", () => {
    const buffer = triangleGlb();
    expect(() =>
      parseGlb(triangleGlb({ totalLength: buffer.byteLength + 4 })),
    ).toThrow(/length mismatch/);
  });

  it("rejects a BIN chunk that runs past the file", () => {
    expect(() => parseGlb(triangleGlb({ binChunkLength: 4096 }))).toThrow(
      /truncated GLB BIN chunk/,
    );
  });

  it("rejects an accessor that reads past the end of its bufferView", () => {
    const { json, bin } = buildDocument(TRIANGLE);
    const accessors = json["accessors"] as Record<string, unknown>[];
    const position = accessors[0];
    if (position) position["count"] = 999;
    expect(() => parseGlb(assemble(json, bin))).toThrow(
      /POSITION accessor reads .* past the end of the buffer/,
    );
  });

  it("rejects an index that points past the vertices", () => {
    const { json, bin } = buildDocument({
      positions: TRIANGLE.positions,
      indices: [0, 1, 7],
    });
    expect(() => parseGlb(assemble(json, bin))).toThrow(
      /index 7 at position 2 is past the 3-vertex/,
    );
  });

  it("rejects a JSON chunk that is not JSON", () => {
    const bin = new Uint8Array(0);
    const buffer = assemble({ asset: { version: "2.0" } }, bin);
    // Corrupt the first byte of the JSON payload, which starts at byte 20.
    new Uint8Array(buffer)[20] = 0x7b + 1;
    expect(() => parseGlb(buffer)).toThrow(/not valid JSON/);
  });

  it("throws GlbParseError, not a bare TypeError, on a gutted document", () => {
    const buffer = assemble({ asset: { version: "2.0" } }, new Uint8Array(0));
    expect(() => parseGlb(buffer)).toThrow(GlbParseError);
    expect(() => parseGlb(buffer)).toThrow(/"meshes" array/);
  });
});

describe("parseGlb — index component types", () => {
  it.each([
    ["UNSIGNED_BYTE", COMPONENT_UNSIGNED_BYTE],
    ["UNSIGNED_SHORT", COMPONENT_UNSIGNED_SHORT],
    ["UNSIGNED_INT", COMPONENT_UNSIGNED_INT],
  ])("reads %s indices as the same triangle list", (_name, componentType) => {
    const { json, bin } = buildDocument({
      positions: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
      indices: [0, 1, 2, 2, 1, 3],
      indexComponentType: componentType,
    });
    const parsed = parseGlb(assemble(json, bin));
    expect(Array.from(parsed.indices)).toEqual([0, 1, 2, 2, 1, 3]);
    expect(parsed.triangleCount).toBe(2);
  });
});

describe("parseGlb — generated normals", () => {
  // A regular tetrahedron centred on the origin: every vertex normal must end
  // up pointing straight away from the centre, so "outward" is checkable
  // without knowing anything about the smoothing weights.
  const TETRAHEDRON: MeshSpec = {
    positions: [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1],
    // Counter-clockwise seen from outside, the winding glTF calls front-facing.
    indices: [0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
  };

  it("generates unit-length normals that face outward", () => {
    const { json, bin } = buildDocument(TETRAHEDRON);
    const parsed = parseGlb(assemble(json, bin));
    const normals = parsed.normals;
    expect(normals).not.toBeNull();
    if (!normals) return;
    expect(normals.length).toBe(parsed.positions.length);

    for (let v = 0; v < parsed.vertexCount; v += 1) {
      const normal = triple(normals, v);
      const position = triple(parsed.positions, v);
      expect(Math.hypot(...normal)).toBeCloseTo(1, 5);
      // The outward direction at a tetrahedron corner IS the corner direction.
      const outward = Math.hypot(...position);
      const dot =
        (normal[0] * position[0] +
          normal[1] * position[1] +
          normal[2] * position[2]) /
        outward;
      expect(dot).toBeCloseTo(1, 5);
    }
  });

  it("prefers the file's own normals over generated ones", () => {
    const { json, bin } = buildDocument({
      ...TETRAHEDRON,
      normals: new Array(4).fill([0, 1, 0]).flat() as number[],
    });
    const parsed = parseGlb(assemble(json, bin));
    expect(triple(parsed.normals ?? new Float32Array(3), 0)).toEqual([0, 1, 0]);
  });
});
