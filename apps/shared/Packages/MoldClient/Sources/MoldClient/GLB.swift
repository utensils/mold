import Foundation

/// Binary glTF reader for the meshes mold writes.
///
/// The Swift counterpart of `studio/lib/glb.ts` — itself the counterpart of
/// `crates/mold-inference/src/hunyuan3d/glb.rs`: one mesh, one triangle
/// primitive, `POSITION` plus indices, optionally `NORMAL`, `TEXCOORD_0`,
/// `COLOR_0` and an embedded PNG baseColor texture. It is NOT a general glTF
/// loader — no scene graph, no animation, no Draco, no external buffers, no
/// sparse accessors — and it SAYS SO rather than half-reading a file it does
/// not understand.
///
/// A truncated or hostile `.glb` must throw [`GLBParseError`], never read past
/// the buffer and never loop forever: the gallery hands these bytes straight
/// off the wire.
public enum GLB {
    /// A mesh larger than this is refused rather than allowed to wedge the
    /// app. `MeshViewer.vue:225`'s own cap.
    public static let maximumBytes = 256 * 1024 * 1024

    private static let modeTriangles = 4

    /// Reads the first triangle primitive of the first mesh.
    ///
    /// The cap is a parameter so a test can prove the refusal WITHOUT
    /// allocating a quarter of a gigabyte; nothing in the app passes one.
    public static func parse(_ data: Data, maximumBytes: Int = maximumBytes) throws -> ParsedMesh {
        guard data.count <= maximumBytes else {
            throw GLBParseError(
                "This mesh is \(data.count) bytes, past the \(maximumBytes)-byte "
                    + "limit this app will read.")
        }
        return try parse([UInt8](data))
    }

    static func parse(_ buffer: [UInt8]) throws -> ParsedMesh {
        let container = try GLBContainer.split(buffer)
        let document = container.json
        let bin = container.bin

        let primitive = try firstPrimitive(in: document)
        let attributes = try GLBDocument.object(primitive, "attributes", "primitive")
        let positionIndex = try GLBDocument.int(attributes, "POSITION", "primitive attributes")
        let positionLayout = try GLBAccessor.layout(in: document, bin: bin,
                                                    index: positionIndex, label: "POSITION")
        guard positionLayout.components == 3 else {
            throw GLBParseError("the POSITION accessor must be VEC3")
        }
        let positions = try positionLayout.floats(in: bin, label: "POSITION")
        let vertexCount = positionLayout.count
        guard vertexCount > 0 else { throw GLBParseError("GLB mesh has no vertices") }

        let indices = try readIndices(document: document, bin: bin, primitive: primitive,
                                      vertexCount: vertexCount)
        let optional = { (name: String, components: Int) throws -> [Float]? in
            try attribute(name, components: components, document: document, bin: bin,
                          attributes: attributes, vertexCount: vertexCount)
        }
        return ParsedMesh(
            positions: positions,
            normals: try optional("NORMAL", 3)
                ?? GLBGeometry.generatedNormals(positions: positions, indices: indices),
            uvs: try optional("TEXCOORD_0", 2),
            colors: try optional("COLOR_0", 3),
            indices: indices,
            baseColorTexture: try GLBTexture.baseColor(in: document, bin: bin,
                                                       primitive: primitive),
            bounds: try GLBGeometry.bounds(of: positions),
            vertexCount: vertexCount,
            triangleCount: indices.count / 3)
    }

    private static func firstPrimitive(in document: [String: Any]) throws -> [String: Any] {
        let meshes = try GLBDocument.array(document, "meshes", "document")
        guard !meshes.isEmpty else { throw GLBParseError("GLB has no meshes") }
        let mesh = try GLBDocument.objectAt(meshes, 0, "mesh")
        let primitives = try GLBDocument.array(mesh, "primitives", "mesh 0")
        guard !primitives.isEmpty else { throw GLBParseError("GLB mesh 0 has no primitives") }
        let primitive = try GLBDocument.objectAt(primitives, 0, "primitive")
        let mode = try GLBDocument.int(primitive, "mode", "primitive", fallback: modeTriangles)
        guard mode == modeTriangles else {
            throw GLBParseError(
                "GLB primitive mode \(mode) is not triangles; mold only writes triangle meshes")
        }
        return primitive
    }
}
