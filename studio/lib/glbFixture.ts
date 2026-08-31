/**
 * GLB fixtures, BUILT rather than committed as binary blobs.
 *
 * The writer under test on the Rust side is
 * `crates/mold-inference/src/hunyuan3d/glb.rs`; assembling the container here
 * is the only way to exercise the checks a well-formed file can never trip —
 * bad magic, the wrong version, a header length that lies, a truncated chunk.
 * Test-only: nothing that ships imports this module.
 */

/*
 * GLB fixtures are BUILT here rather than committed as binaries: the writer
 * under test on the Rust side is `crates/mold-inference/src/hunyuan3d/glb.rs`,
 * and a hand-assembled container is the only way to test the container checks
 * (bad magic, wrong version, a length that lies, a truncated chunk) that a
 * well-formed fixture can never exercise.
 */

export const CHUNK_JSON = 0x4e4f534a;
const CHUNK_BIN = 0x004e4942;
export const COMPONENT_UNSIGNED_BYTE = 5121;
export const COMPONENT_UNSIGNED_SHORT = 5123;
export const COMPONENT_UNSIGNED_INT = 5125;
const COMPONENT_FLOAT = 5126;

const pad4 = (length: number): number => (4 - (length % 4)) % 4;

/** Appends 4-byte-aligned blobs, handing back each blob's bufferView fields. */
export class BinWriter {
  private readonly parts: Uint8Array[] = [];
  private length = 0;

  push(bytes: Uint8Array): { byteOffset: number; byteLength: number } {
    const byteOffset = this.length;
    this.parts.push(bytes);
    this.length += bytes.length;
    const padding = pad4(bytes.length);
    if (padding > 0) {
      this.parts.push(new Uint8Array(padding));
      this.length += padding;
    }
    return { byteOffset, byteLength: bytes.length };
  }

  floats(values: number[]): { byteOffset: number; byteLength: number } {
    return this.push(new Uint8Array(new Float32Array(values).buffer));
  }

  bytes(): Uint8Array {
    const out = new Uint8Array(this.length);
    let at = 0;
    for (const part of this.parts) {
      out.set(part, at);
      at += part.length;
    }
    return out;
  }
}

export interface ContainerOverrides {
  magic?: string;
  version?: number;
  /** The length written into the header, when it should disagree. */
  totalLength?: number;
  /** The length written into the BIN chunk header, when it should lie. */
  binChunkLength?: number;
}

export function assemble(
  json: unknown,
  bin: Uint8Array,
  overrides: ContainerOverrides = {},
): ArrayBuffer {
  const encoder = new TextEncoder();
  const jsonBytes = encoder.encode(JSON.stringify(json));
  const jsonPadded = new Uint8Array(jsonBytes.length + pad4(jsonBytes.length));
  jsonPadded.set(jsonBytes);
  jsonPadded.fill(0x20, jsonBytes.length); // JSON pads with spaces
  const binPadded = new Uint8Array(bin.length + pad4(bin.length));
  binPadded.set(bin);

  const total = 12 + 8 + jsonPadded.length + 8 + binPadded.length;
  const buffer = new ArrayBuffer(total);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  bytes.set(encoder.encode(overrides.magic ?? "glTF"), 0);
  view.setUint32(4, overrides.version ?? 2, true);
  view.setUint32(8, overrides.totalLength ?? total, true);
  view.setUint32(12, jsonPadded.length, true);
  view.setUint32(16, CHUNK_JSON, true);
  bytes.set(jsonPadded, 20);
  const binHeader = 20 + jsonPadded.length;
  view.setUint32(binHeader, overrides.binChunkLength ?? binPadded.length, true);
  view.setUint32(binHeader + 4, CHUNK_BIN, true);
  bytes.set(binPadded, binHeader + 8);
  return buffer;
}

export interface MeshSpec {
  positions: number[];
  indices: number[];
  indexComponentType?: number;
  normals?: number[];
  uvs?: number[];
  colors?: number[];
  /** Embedded baseColor image bytes (any bytes: the parser never decodes). */
  png?: Uint8Array;
}

/** The exact document shape `write_glb` emits, as JSON we can then mutate. */
export function buildDocument(spec: MeshSpec): {
  json: Record<string, unknown>;
  bin: Uint8Array;
} {
  const componentType = spec.indexComponentType ?? COMPONENT_UNSIGNED_INT;
  const bin = new BinWriter();
  const vertexCount = spec.positions.length / 3;

  const positionView = bin.floats(spec.positions);
  const indexArray =
    componentType === COMPONENT_UNSIGNED_BYTE
      ? new Uint8Array(spec.indices)
      : componentType === COMPONENT_UNSIGNED_SHORT
        ? new Uint16Array(spec.indices)
        : new Uint32Array(spec.indices);
  const indexView = bin.push(
    new Uint8Array(
      indexArray.buffer,
      indexArray.byteOffset,
      indexArray.byteLength,
    ),
  );

  const min: number[] = [Infinity, Infinity, Infinity];
  const max: number[] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < spec.positions.length; i += 1) {
    const axis = i % 3;
    const value = spec.positions[i] ?? 0;
    min[axis] = Math.min(min[axis] ?? Infinity, value);
    max[axis] = Math.max(max[axis] ?? -Infinity, value);
  }

  const bufferViews: Record<string, unknown>[] = [
    { buffer: 0, ...positionView, target: 34962 },
    { buffer: 0, ...indexView, target: 34963 },
  ];
  const accessors: Record<string, unknown>[] = [
    {
      bufferView: 0,
      byteOffset: 0,
      componentType: COMPONENT_FLOAT,
      count: vertexCount,
      type: "VEC3",
      min,
      max,
    },
    {
      bufferView: 1,
      byteOffset: 0,
      componentType,
      count: spec.indices.length,
      type: "SCALAR",
    },
  ];
  const attributes: Record<string, number> = { POSITION: 0 };

  const addAttribute = (
    name: string,
    values: number[] | undefined,
    type: string,
  ): void => {
    if (!values) return;
    bufferViews.push({ buffer: 0, ...bin.floats(values), target: 34962 });
    accessors.push({
      bufferView: bufferViews.length - 1,
      byteOffset: 0,
      componentType: COMPONENT_FLOAT,
      count: vertexCount,
      type,
    });
    attributes[name] = accessors.length - 1;
  };
  addAttribute("TEXCOORD_0", spec.uvs, "VEC2");
  addAttribute("COLOR_0", spec.colors, "VEC3");
  addAttribute("NORMAL", spec.normals, "VEC3");

  const json: Record<string, unknown> = {
    asset: { version: "2.0", generator: "mold" },
    buffers: [{ byteLength: 0 }],
    bufferViews,
    accessors,
    meshes: [
      {
        primitives: [{ attributes, indices: 1, mode: 4, material: 0 }],
      },
    ],
    nodes: [{ mesh: 0 }],
    scenes: [{ nodes: [0] }],
    scene: 0,
    materials: [
      {
        pbrMetallicRoughness: {
          baseColorFactor: [0.22, 0.22, 0.22, 1],
          metallicFactor: 0,
          roughnessFactor: 0.5,
        },
        doubleSided: true,
      },
    ],
  };

  if (spec.png) {
    bufferViews.push({ buffer: 0, ...bin.push(spec.png) });
    json["images"] = [
      { bufferView: bufferViews.length - 1, mimeType: "image/png" },
    ];
    json["samplers"] = [{ magFilter: 9729, minFilter: 9729 }];
    json["textures"] = [{ source: 0, sampler: 0 }];
    const material = (json["materials"] as Record<string, unknown>[])[0];
    const pbr = material?.["pbrMetallicRoughness"] as Record<string, unknown>;
    pbr["baseColorTexture"] = { index: 0, texCoord: 0 };
  }

  const bytes = bin.bytes();
  json["buffers"] = [{ byteLength: bytes.length }];
  return { json, bin: bytes };
}

/** The smallest mesh mold can write: one triangle, positions and indices. */
export const TRIANGLE: MeshSpec = {
  positions: [0, 0, 0, 2, 0, 0, 0, 4, -1],
  indices: [0, 1, 2],
};

/** A complete, valid one-triangle `.glb`, optionally corrupted on the way out. */
export function triangleGlb(overrides: ContainerOverrides = {}): ArrayBuffer {
  const { json, bin } = buildDocument(TRIANGLE);
  return assemble(json, bin, overrides);
}
