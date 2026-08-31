/**
 * Binary glTF reader for the meshes mold writes.
 *
 * The counterpart of `crates/mold-inference/src/hunyuan3d/glb.rs`: one mesh,
 * one triangle primitive, `POSITION` plus indices, optionally `NORMAL`,
 * `TEXCOORD_0`, `COLOR_0` and an embedded PNG baseColor texture. It is NOT a
 * general glTF loader — no scene graph, no animation, no Draco, no external
 * buffers, no sparse accessors — and it says so rather than half-reading a
 * file it does not understand.
 *
 * Every offset is checked against the chunk that owns it BEFORE any typed
 * array is created. A truncated or hostile `.glb` must throw
 * [`GlbParseError`], never read past the buffer and never loop forever: the
 * gallery hands this function bytes straight off the wire.
 */

/** A `.glb` that this reader will not or cannot read, with the reason why. */
export class GlbParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GlbParseError";
  }
}

export interface ParsedMesh {
  positions: Float32Array; // xyz triples
  normals: Float32Array | null;
  uvs: Float32Array | null;
  colors: Float32Array | null;
  indices: Uint32Array;
  baseColorTexture: Blob | null;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  vertexCount: number;
  triangleCount: number;
}

const MAGIC = 0x46546c67; // "glTF", read as one little-endian u32
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"
const MODE_TRIANGLES = 4;

const COMPONENT_UNSIGNED_BYTE = 5121;
const COMPONENT_UNSIGNED_SHORT = 5123;
const COMPONENT_UNSIGNED_INT = 5125;
const COMPONENT_FLOAT = 5126;

const COMPONENT_SIZE = new Map<number, number>([
  [5120, 1], // BYTE
  [COMPONENT_UNSIGNED_BYTE, 1],
  [5122, 2], // SHORT
  [COMPONENT_UNSIGNED_SHORT, 2],
  [COMPONENT_UNSIGNED_INT, 4],
  [COMPONENT_FLOAT, 4],
]);

const TYPE_COMPONENTS = new Map<string, number>([
  ["SCALAR", 1],
  ["VEC2", 2],
  ["VEC3", 3],
  ["VEC4", 4],
]);

/**
 * Typed arrays read in platform byte order; glTF is little-endian. Every
 * WebGL-capable platform is little-endian, so the fast zero-copy path is
 * taken there and the portable `DataView` loop covers the rest.
 */
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

// ── JSON navigation ────────────────────────────────────────────────────────
// The document is untrusted `unknown` until each field has been checked, so a
// missing or wrongly typed field is a message rather than a TypeError.

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function objectField(
  parent: Record<string, unknown>,
  key: string,
  what: string,
): Record<string, unknown> {
  const value = parent[key];
  if (!isObject(value)) {
    throw new GlbParseError(`GLB ${what} is missing the "${key}" object`);
  }
  return value;
}

function arrayField(
  parent: Record<string, unknown>,
  key: string,
  what: string,
): unknown[] {
  const value = parent[key];
  if (!Array.isArray(value)) {
    throw new GlbParseError(`GLB ${what} is missing the "${key}" array`);
  }
  return value;
}

function objectAt(
  array: unknown[],
  index: number,
  what: string,
): Record<string, unknown> {
  if (!Number.isInteger(index) || index < 0 || index >= array.length) {
    throw new GlbParseError(
      `GLB ${what} index ${index} is out of range (${array.length} entries)`,
    );
  }
  const value = array[index];
  if (!isObject(value)) {
    throw new GlbParseError(`GLB ${what} ${index} is not an object`);
  }
  return value;
}

/** A non-negative integer field, defaulted when absent (glTF's own rule). */
function intField(
  parent: Record<string, unknown>,
  key: string,
  what: string,
  fallback: number | null = null,
): number {
  const value = parent[key];
  if (value === undefined || value === null) {
    if (fallback !== null) return fallback;
    throw new GlbParseError(`GLB ${what} is missing "${key}"`);
  }
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) {
    throw new GlbParseError(
      `GLB ${what} has a non-integer "${key}": ${String(value)}`,
    );
  }
  return value;
}

function stringField(
  parent: Record<string, unknown>,
  key: string,
  what: string,
): string {
  const value = parent[key];
  if (typeof value !== "string") {
    throw new GlbParseError(`GLB ${what} is missing the "${key}" string`);
  }
  return value;
}

// ── Container ──────────────────────────────────────────────────────────────

interface Container {
  json: Record<string, unknown>;
  bin: Uint8Array;
}

function chunkName(type: number): string {
  if (type === CHUNK_JSON) return "JSON";
  if (type === CHUNK_BIN) return "BIN";
  return `0x${type.toString(16).padStart(8, "0")}`;
}

/** 12-byte header, then length-prefixed chunks padded to 4 bytes. */
function splitContainer(buffer: ArrayBuffer): Container {
  if (buffer.byteLength < 12) {
    throw new GlbParseError(
      `not a GLB: a 12-byte header needs 12 bytes, got ${buffer.byteLength}`,
    );
  }
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) {
    throw new GlbParseError(
      `not a GLB: bad magic 0x${magic.toString(16).padStart(8, "0")}, expected "glTF"`,
    );
  }
  const version = view.getUint32(4, true);
  if (version !== 2) {
    throw new GlbParseError(
      `unsupported GLB version ${version}: only glTF 2 binary is supported`,
    );
  }
  const declared = view.getUint32(8, true);
  if (declared !== buffer.byteLength) {
    throw new GlbParseError(
      `GLB length mismatch: the header declares ${declared} bytes but the buffer holds ${buffer.byteLength}`,
    );
  }

  let offset = 12;
  let jsonChunk: Uint8Array | null = null;
  let binChunk: Uint8Array | null = null;
  while (offset + 8 <= declared) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const start = offset + 8;
    if (length > declared - start) {
      throw new GlbParseError(
        `truncated GLB ${chunkName(type)} chunk: it declares ${length} bytes but only ${declared - start} remain`,
      );
    }
    if (type === CHUNK_JSON && jsonChunk === null) {
      jsonChunk = new Uint8Array(buffer, start, length);
    } else if (type === CHUNK_BIN && binChunk === null) {
      binChunk = new Uint8Array(buffer, start, length);
    }
    // Chunk payloads are 4-byte aligned; `start` always grows, so a zero-length
    // chunk cannot stall the walk.
    offset = start + length + ((4 - (length % 4)) % 4);
  }

  if (jsonChunk === null) {
    throw new GlbParseError("GLB has no JSON chunk");
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(new TextDecoder().decode(jsonChunk)) as unknown;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new GlbParseError(`GLB JSON chunk is not valid JSON: ${detail}`);
  }
  if (!isObject(parsed)) {
    throw new GlbParseError("GLB JSON chunk is not a glTF object");
  }
  return { json: parsed, bin: binChunk ?? new Uint8Array(0) };
}

// ── Accessors ──────────────────────────────────────────────────────────────

interface AccessorLayout {
  componentType: number;
  components: number;
  count: number;
  /** Byte offset of element 0 within the BIN chunk. */
  start: number;
  stride: number;
  componentSize: number;
}

function accessorLayout(
  doc: Record<string, unknown>,
  bin: Uint8Array,
  index: number,
  label: string,
): AccessorLayout {
  const accessor = objectAt(
    arrayField(doc, "accessors", "document"),
    index,
    `${label} accessor`,
  );
  if (accessor["sparse"] !== undefined) {
    throw new GlbParseError(
      `the ${label} accessor is sparse, which mold never writes`,
    );
  }
  const type = stringField(accessor, "type", `${label} accessor`);
  const components = TYPE_COMPONENTS.get(type);
  if (components === undefined) {
    throw new GlbParseError(
      `the ${label} accessor has unsupported type "${type}"`,
    );
  }
  const componentType = intField(
    accessor,
    "componentType",
    `${label} accessor`,
  );
  const componentSize = COMPONENT_SIZE.get(componentType);
  if (componentSize === undefined) {
    throw new GlbParseError(
      `the ${label} accessor has unsupported componentType ${componentType}`,
    );
  }
  const count = intField(accessor, "count", `${label} accessor`);
  const viewIndex = intField(accessor, "bufferView", `${label} accessor`);
  const view = objectAt(
    arrayField(doc, "bufferViews", "document"),
    viewIndex,
    `${label} bufferView`,
  );
  const buffer = intField(view, "buffer", `${label} bufferView`, 0);
  if (buffer !== 0) {
    throw new GlbParseError(
      `the ${label} bufferView points at buffer ${buffer}; only the embedded BIN chunk is supported`,
    );
  }
  const viewOffset = intField(view, "byteOffset", `${label} bufferView`, 0);
  const viewLength = intField(view, "byteLength", `${label} bufferView`);
  if (viewOffset + viewLength > bin.byteLength) {
    throw new GlbParseError(
      `the ${label} bufferView ends at byte ${viewOffset + viewLength}, past the end of the ${bin.byteLength}-byte BIN chunk`,
    );
  }

  const elementSize = components * componentSize;
  const stride = intField(
    view,
    "byteStride",
    `${label} bufferView`,
    elementSize,
  );
  if (stride < elementSize) {
    throw new GlbParseError(
      `the ${label} bufferView has byteStride ${stride}, smaller than its ${elementSize}-byte elements`,
    );
  }
  const accessorOffset = intField(
    accessor,
    "byteOffset",
    `${label} accessor`,
    0,
  );
  const needed =
    count === 0 ? 0 : accessorOffset + (count - 1) * stride + elementSize;
  if (needed > viewLength) {
    throw new GlbParseError(
      `the ${label} accessor reads ${needed} bytes from a ${viewLength}-byte bufferView, past the end of the buffer`,
    );
  }

  return {
    componentType,
    components,
    count,
    start: viewOffset + accessorOffset,
    stride,
    componentSize,
  };
}

function readFloats(
  bin: Uint8Array,
  layout: AccessorLayout,
  label: string,
): Float32Array {
  if (layout.componentType !== COMPONENT_FLOAT) {
    throw new GlbParseError(
      `the ${label} accessor is componentType ${layout.componentType}; mold writes float ${label}`,
    );
  }
  const total = layout.count * layout.components;
  const elementSize = layout.components * 4;
  const base = bin.byteOffset + layout.start;
  if (LITTLE_ENDIAN && layout.stride === elementSize && base % 4 === 0) {
    return new Float32Array(bin.buffer, base, total).slice();
  }
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const out = new Float32Array(total);
  for (let i = 0; i < layout.count; i += 1) {
    const row = layout.start + i * layout.stride;
    for (let c = 0; c < layout.components; c += 1) {
      out[i * layout.components + c] = view.getFloat32(row + c * 4, true);
    }
  }
  return out;
}

function readIndices(bin: Uint8Array, layout: AccessorLayout): Uint32Array {
  if (layout.components !== 1) {
    throw new GlbParseError("the index accessor must be SCALAR");
  }
  const view = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const out = new Uint32Array(layout.count);
  for (let i = 0; i < layout.count; i += 1) {
    const at = layout.start + i * layout.stride;
    switch (layout.componentType) {
      case COMPONENT_UNSIGNED_BYTE:
        out[i] = view.getUint8(at);
        break;
      case COMPONENT_UNSIGNED_SHORT:
        out[i] = view.getUint16(at, true);
        break;
      case COMPONENT_UNSIGNED_INT:
        out[i] = view.getUint32(at, true);
        break;
      default:
        throw new GlbParseError(
          `the index accessor has componentType ${layout.componentType}; expected an unsigned byte, short or int`,
        );
    }
  }
  return out;
}

// ── Geometry helpers ───────────────────────────────────────────────────────

function computeBounds(positions: Float32Array): ParsedMesh["bounds"] {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    const point: [number, number, number] = [
      positions[i] ?? 0,
      positions[i + 1] ?? 0,
      positions[i + 2] ?? 0,
    ];
    if (!point.every((value) => Number.isFinite(value))) {
      throw new GlbParseError("the POSITION accessor holds a non-finite value");
    }
    min[0] = Math.min(min[0], point[0]);
    min[1] = Math.min(min[1], point[1]);
    min[2] = Math.min(min[2], point[2]);
    max[0] = Math.max(max[0], point[0]);
    max[1] = Math.max(max[1], point[1]);
    max[2] = Math.max(max[2], point[2]);
  }
  return { min, max };
}

/**
 * Area-weighted smooth normals, for meshes written without `NORMAL`.
 *
 * glTF winds front faces counter-clockwise, so `(b - a) × (c - a)` already
 * points out of the surface; a mesh that arrives without normals therefore
 * shades the same way as one that carries them, instead of rendering black.
 */
export function generateNormals(
  positions: Float32Array,
  indices: Uint32Array,
): Float32Array {
  const normals = new Float32Array(positions.length);
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const ia = (indices[t] ?? 0) * 3;
    const ib = (indices[t + 1] ?? 0) * 3;
    const ic = (indices[t + 2] ?? 0) * 3;
    const ax = positions[ia] ?? 0;
    const ay = positions[ia + 1] ?? 0;
    const az = positions[ia + 2] ?? 0;
    const ux = (positions[ib] ?? 0) - ax;
    const uy = (positions[ib + 1] ?? 0) - ay;
    const uz = (positions[ib + 2] ?? 0) - az;
    const vx = (positions[ic] ?? 0) - ax;
    const vy = (positions[ic + 1] ?? 0) - ay;
    const vz = (positions[ic + 2] ?? 0) - az;
    // Left unnormalized on purpose: the cross product's length is twice the
    // triangle's area, which is the weight smooth shading wants.
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (const base of [ia, ib, ic]) {
      normals[base] = (normals[base] ?? 0) + nx;
      normals[base + 1] = (normals[base + 1] ?? 0) + ny;
      normals[base + 2] = (normals[base + 2] ?? 0) + nz;
    }
  }
  for (let i = 0; i < normals.length; i += 3) {
    const x = normals[i] ?? 0;
    const y = normals[i + 1] ?? 0;
    const z = normals[i + 2] ?? 0;
    const length = Math.hypot(x, y, z);
    if (length > 0) {
      normals[i] = x / length;
      normals[i + 1] = y / length;
      normals[i + 2] = z / length;
    } else {
      // An unreferenced or degenerate vertex: any unit vector beats NaN.
      normals[i + 1] = 1;
    }
  }
  return normals;
}

// ── Texture ────────────────────────────────────────────────────────────────

function readBaseColorTexture(
  doc: Record<string, unknown>,
  bin: Uint8Array,
  primitive: Record<string, unknown>,
): Blob | null {
  if (typeof Blob === "undefined") return null;
  const materialIndex = primitive["material"];
  if (typeof materialIndex !== "number") return null;
  const materials = doc["materials"];
  if (!Array.isArray(materials)) return null;
  const material = objectAt(materials, materialIndex, "material");
  const pbr = material["pbrMetallicRoughness"];
  if (!isObject(pbr)) return null;
  const reference = pbr["baseColorTexture"];
  if (!isObject(reference)) return null;
  const textureIndex = intField(reference, "index", "baseColorTexture");
  const texture = objectAt(
    arrayField(doc, "textures", "document"),
    textureIndex,
    "texture",
  );
  const sourceIndex = intField(texture, "source", "texture");
  const image = objectAt(
    arrayField(doc, "images", "document"),
    sourceIndex,
    "image",
  );
  // Only embedded images: a `uri` would be a second network fetch, and mold
  // never writes one.
  const viewIndex = intField(image, "bufferView", "image");
  const view = objectAt(
    arrayField(doc, "bufferViews", "document"),
    viewIndex,
    "image bufferView",
  );
  const offset = intField(view, "byteOffset", "image bufferView", 0);
  const length = intField(view, "byteLength", "image bufferView");
  if (offset + length > bin.byteLength) {
    throw new GlbParseError(
      `the baseColor image ends at byte ${offset + length}, past the end of the ${bin.byteLength}-byte BIN chunk`,
    );
  }
  const mimeType =
    typeof image["mimeType"] === "string" ? image["mimeType"] : "image/png";
  return new Blob([bin.slice(offset, offset + length)], { type: mimeType });
}

// ── Entry point ────────────────────────────────────────────────────────────

/**
 * Read the first triangle primitive of the first mesh.
 *
 * @throws {GlbParseError} on anything malformed, unsupported, or out of range.
 */
export function parseGlb(buffer: ArrayBuffer): ParsedMesh {
  const { json: doc, bin } = splitContainer(buffer);

  const meshes = arrayField(doc, "meshes", "document");
  if (meshes.length === 0) {
    throw new GlbParseError("GLB has no meshes");
  }
  const mesh = objectAt(meshes, 0, "mesh");
  const primitives = arrayField(mesh, "primitives", "mesh 0");
  if (primitives.length === 0) {
    throw new GlbParseError("GLB mesh 0 has no primitives");
  }
  const primitive = objectAt(primitives, 0, "primitive");
  const mode = intField(primitive, "mode", "primitive", MODE_TRIANGLES);
  if (mode !== MODE_TRIANGLES) {
    throw new GlbParseError(
      `GLB primitive mode ${mode} is not triangles; mold only writes triangle meshes`,
    );
  }

  const attributes = objectField(primitive, "attributes", "primitive");
  const positionIndex = intField(
    attributes,
    "POSITION",
    "primitive attributes",
  );
  const positionLayout = accessorLayout(doc, bin, positionIndex, "POSITION");
  if (positionLayout.components !== 3) {
    throw new GlbParseError("the POSITION accessor must be VEC3");
  }
  const positions = readFloats(bin, positionLayout, "POSITION");
  const vertexCount = positionLayout.count;
  if (vertexCount === 0) {
    throw new GlbParseError("GLB mesh has no vertices");
  }

  const indicesIndex = primitive["indices"];
  let indices: Uint32Array;
  if (indicesIndex === undefined || indicesIndex === null) {
    // Non-indexed geometry: glTF says draw the vertices in order.
    indices = new Uint32Array(vertexCount);
    for (let i = 0; i < vertexCount; i += 1) indices[i] = i;
  } else {
    if (typeof indicesIndex !== "number") {
      throw new GlbParseError('GLB primitive has a non-numeric "indices"');
    }
    indices = readIndices(bin, accessorLayout(doc, bin, indicesIndex, "index"));
  }
  if (indices.length % 3 !== 0) {
    throw new GlbParseError(
      `GLB index count ${indices.length} is not a whole number of triangles`,
    );
  }
  for (let i = 0; i < indices.length; i += 1) {
    const index = indices[i] ?? 0;
    if (index >= vertexCount) {
      throw new GlbParseError(
        `GLB index ${index} at position ${i} is past the ${vertexCount}-vertex POSITION accessor`,
      );
    }
  }

  const optional = (name: string, components: number): Float32Array | null => {
    const index = attributes[name];
    if (index === undefined || index === null) return null;
    if (typeof index !== "number") {
      throw new GlbParseError(`GLB attribute ${name} is not an accessor index`);
    }
    const layout = accessorLayout(doc, bin, index, name);
    if (layout.count !== vertexCount) {
      throw new GlbParseError(
        `the ${name} accessor has ${layout.count} elements but POSITION has ${vertexCount}`,
      );
    }
    if (layout.components < components) {
      throw new GlbParseError(
        `the ${name} accessor has ${layout.components} components, expected ${components}`,
      );
    }
    const data = readFloats(bin, layout, name);
    if (layout.components === components) return data;
    // COLOR_0 may be VEC4; the renderer wants RGB, so drop the alpha rather
    // than reject the file.
    const trimmed = new Float32Array(vertexCount * components);
    for (let i = 0; i < vertexCount; i += 1) {
      for (let c = 0; c < components; c += 1) {
        trimmed[i * components + c] = data[i * layout.components + c] ?? 0;
      }
    }
    return trimmed;
  };

  return {
    positions,
    normals: optional("NORMAL", 3) ?? generateNormals(positions, indices),
    uvs: optional("TEXCOORD_0", 2),
    colors: optional("COLOR_0", 3),
    indices,
    baseColorTexture: readBaseColorTexture(doc, bin, primitive),
    bounds: computeBounds(positions),
    vertexCount,
    triangleCount: indices.length / 3,
  };
}
