import Foundation

@testable import MoldClient

/// GLB fixtures, BUILT rather than committed as binary blobs.
///
/// Port of `studio/lib/glbFixture.ts`. The writer under test on the Rust side
/// is `crates/mold-inference/src/hunyuan3d/glb.rs`, and a hand-assembled
/// container is the only way to exercise the checks a well-formed file can
/// never trip — bad magic, the wrong version, a header length that lies, a
/// truncated chunk.
enum GLBFixture {
    static let chunkJSON: UInt32 = 0x4E4F_534A
    static let chunkBIN: UInt32 = 0x004E_4942
    static let componentFloat = 5126

    struct MeshSpec {
        var positions: [Float]
        var indices: [UInt32]
        var indexComponentType = GLBAccessor.unsignedInt
        var normals: [Float]?
        var uvs: [Float]?
        var colors: [Float]?
        /// Embedded baseColor image bytes (any bytes: the parser never decodes).
        var png: [UInt8]?
    }

    struct Overrides {
        var magic: String? = nil
        var version: UInt32? = nil
        /// The length written into the header, when it should disagree.
        var totalLength: Int? = nil
        /// The length written into the BIN chunk header, when it should lie.
        var binChunkLength: Int? = nil
    }

    /// The smallest mesh mold can write: one triangle, positions and indices.
    static let triangle = MeshSpec(positions: [0, 0, 0, 2, 0, 0, 0, 4, -1],
                                   indices: [0, 1, 2])

    /// A complete, valid one-triangle `.glb`, optionally corrupted on the way out.
    static func triangleGLB(_ overrides: Overrides = Overrides()) -> [UInt8] {
        let built = buildDocument(triangle)
        return assemble(built.json, built.bin, overrides)
    }

    static func assemble(_ json: [String: Any], _ bin: [UInt8],
                         _ overrides: Overrides = Overrides()) -> [UInt8] {
        let jsonBytes = [UInt8](try! JSONSerialization.data(withJSONObject: json))
        // JSON pads with spaces; BIN with zeros.
        var jsonPadded = jsonBytes
        jsonPadded.append(contentsOf: [UInt8](repeating: 0x20, count: pad4(jsonBytes.count)))
        var binPadded = bin
        binPadded.append(contentsOf: [UInt8](repeating: 0, count: pad4(bin.count)))

        let total = 12 + 8 + jsonPadded.count + 8 + binPadded.count
        var out: [UInt8] = []
        out.reserveCapacity(total)
        out.append(contentsOf: Array((overrides.magic ?? "glTF").utf8))
        out.append(contentsOf: le32(overrides.version ?? 2))
        out.append(contentsOf: le32(UInt32(overrides.totalLength ?? total)))
        out.append(contentsOf: le32(UInt32(jsonPadded.count)))
        out.append(contentsOf: le32(chunkJSON))
        out.append(contentsOf: jsonPadded)
        out.append(contentsOf: le32(UInt32(overrides.binChunkLength ?? binPadded.count)))
        out.append(contentsOf: le32(chunkBIN))
        out.append(contentsOf: binPadded)
        return out
    }

    /// The exact document shape `write_glb` emits, as JSON a test can mutate.
    static func buildDocument(_ spec: MeshSpec) -> (json: [String: Any], bin: [UInt8]) {
        var writer = BinWriter()
        let vertexCount = spec.positions.count / 3
        let positionView = writer.floats(spec.positions)
        let indexView = writer.push(indexBytes(spec))

        var minimum = [Double](repeating: .infinity, count: 3)
        var maximum = [Double](repeating: -.infinity, count: 3)
        for (offset, value) in spec.positions.enumerated() {
            minimum[offset % 3] = Swift.min(minimum[offset % 3], Double(value))
            maximum[offset % 3] = Swift.max(maximum[offset % 3], Double(value))
        }

        var bufferViews: [[String: Any]] = [
            ["buffer": 0, "byteOffset": positionView.offset,
             "byteLength": positionView.length, "target": 34962],
            ["buffer": 0, "byteOffset": indexView.offset,
             "byteLength": indexView.length, "target": 34963],
        ]
        var accessors: [[String: Any]] = [
            ["bufferView": 0, "byteOffset": 0, "componentType": componentFloat,
             "count": vertexCount, "type": "VEC3", "min": minimum, "max": maximum],
            ["bufferView": 1, "byteOffset": 0, "componentType": spec.indexComponentType,
             "count": spec.indices.count, "type": "SCALAR"],
        ]
        var attributes: [String: Any] = ["POSITION": 0]

        func add(_ name: String, _ values: [Float]?, _ type: String) {
            guard let values else { return }
            let view = writer.floats(values)
            bufferViews.append(["buffer": 0, "byteOffset": view.offset,
                                "byteLength": view.length, "target": 34962])
            accessors.append(["bufferView": bufferViews.count - 1, "byteOffset": 0,
                              "componentType": componentFloat, "count": vertexCount,
                              "type": type])
            attributes[name] = accessors.count - 1
        }
        add("TEXCOORD_0", spec.uvs, "VEC2")
        add("COLOR_0", spec.colors, "VEC3")
        add("NORMAL", spec.normals, "VEC3")

        var json: [String: Any] = [
            "asset": ["version": "2.0", "generator": "mold"],
            "meshes": [["primitives": [["attributes": attributes, "indices": 1,
                                        "mode": 4, "material": 0]]]],
            "nodes": [["mesh": 0]],
            "scenes": [["nodes": [0]]],
            "scene": 0,
            "materials": [["pbrMetallicRoughness": ["baseColorFactor": [0.22, 0.22, 0.22, 1],
                                                    "metallicFactor": 0,
                                                    "roughnessFactor": 0.5],
                           "doubleSided": true]],
        ]

        if let png = spec.png {
            let view = writer.push(png)
            bufferViews.append(["buffer": 0, "byteOffset": view.offset,
                                "byteLength": view.length])
            json["images"] = [["bufferView": bufferViews.count - 1, "mimeType": "image/png"]]
            json["samplers"] = [["magFilter": 9729, "minFilter": 9729]]
            json["textures"] = [["source": 0, "sampler": 0]]
            json["materials"] = [["pbrMetallicRoughness":
                                    ["baseColorFactor": [0.22, 0.22, 0.22, 1],
                                     "metallicFactor": 0, "roughnessFactor": 0.5,
                                     "baseColorTexture": ["index": 0, "texCoord": 0]],
                                  "doubleSided": true]]
        }

        let bytes = writer.bytes
        json["buffers"] = [["byteLength": bytes.count]]
        json["bufferViews"] = bufferViews
        json["accessors"] = accessors
        return (json, bytes)
    }

    private static func indexBytes(_ spec: MeshSpec) -> [UInt8] {
        var out: [UInt8] = []
        for index in spec.indices {
            switch spec.indexComponentType {
            case GLBAccessor.unsignedByte: out.append(UInt8(index))
            case GLBAccessor.unsignedShort:
                out.append(contentsOf: [UInt8(index & 0xFF), UInt8((index >> 8) & 0xFF)])
            default: out.append(contentsOf: le32(index))
            }
        }
        return out
    }

    static func pad4(_ length: Int) -> Int { (4 - (length % 4)) % 4 }

    static func le32(_ value: UInt32) -> [UInt8] {
        [UInt8(value & 0xFF), UInt8((value >> 8) & 0xFF),
         UInt8((value >> 16) & 0xFF), UInt8((value >> 24) & 0xFF)]
    }

    /// Appends 4-byte-aligned blobs, handing back each blob's bufferView fields.
    struct BinWriter {
        private(set) var bytes: [UInt8] = []

        mutating func push(_ blob: [UInt8]) -> (offset: Int, length: Int) {
            let offset = bytes.count
            bytes.append(contentsOf: blob)
            bytes.append(contentsOf: [UInt8](repeating: 0, count: pad4(blob.count)))
            return (offset, blob.count)
        }

        mutating func floats(_ values: [Float]) -> (offset: Int, length: Int) {
            var blob: [UInt8] = []
            blob.reserveCapacity(values.count * 4)
            for value in values { blob.append(contentsOf: le32(value.bitPattern)) }
            return push(blob)
        }
    }
}
