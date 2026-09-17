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

    /// Indices, widened to `u32` whatever the file stored them as, and every
    /// one of them proved to name a vertex that exists.
    private static func readIndices(document: [String: Any], bin: [UInt8],
                                    primitive: [String: Any],
                                    vertexCount: Int) throws -> [UInt32] {
        var indices: [UInt32]
        if let raw = GLBDocument.present(primitive["indices"]) {
            guard let index = GLBDocument.integer(raw) else {
                throw GLBParseError("GLB primitive has a non-numeric \"indices\"")
            }
            indices = try GLBAccessor.layout(in: document, bin: bin, index: index,
                                             label: "index").indices(in: bin)
        } else {
            // Non-indexed geometry: glTF says draw the vertices in order.
            indices = (0..<vertexCount).map(UInt32.init)
        }
        // A mesh with nothing to draw is not a mesh. The reference tolerates
        // it (`drawElements(0)` draws nothing); here it used to reach Metal as
        // a zero-length buffer built off a nil base address, which is API
        // misuse on an untrusted input.
        guard !indices.isEmpty else {
            throw GLBParseError("GLB mesh has no triangles")
        }
        guard indices.count % 3 == 0 else {
            throw GLBParseError(
                "GLB index count \(indices.count) is not a whole number of triangles")
        }
        for (position, index) in indices.enumerated() where Int(index) >= vertexCount {
            throw GLBParseError(
                "GLB index \(index) at position \(position) is past the "
                    + "\(vertexCount)-vertex POSITION accessor")
        }
        return indices
    }

    /// An optional vertex attribute, trimmed to the components the renderer
    /// wants: a VEC4 `COLOR_0` arrives with its alpha DROPPED rather than the
    /// file being refused.
    private static func attribute(_ name: String, components: Int,
                                  document: [String: Any], bin: [UInt8],
                                  attributes: [String: Any],
                                  vertexCount: Int) throws -> [Float]? {
        guard let raw = GLBDocument.present(attributes[name]) else { return nil }
        guard let index = GLBDocument.integer(raw) else {
            throw GLBParseError("GLB attribute \(name) is not an accessor index")
        }
        let layout = try GLBAccessor.layout(in: document, bin: bin, index: index, label: name)
        guard layout.count == vertexCount else {
            throw GLBParseError(
                "the \(name) accessor has \(layout.count) elements but POSITION has "
                    + "\(vertexCount)")
        }
        guard layout.components >= components else {
            throw GLBParseError(
                "the \(name) accessor has \(layout.components) components, expected "
                    + "\(components)")
        }
        let data = try layout.floats(in: bin, label: name)
        if layout.components == components { return data }
        var trimmed = [Float](repeating: 0, count: vertexCount * components)
        for vertex in 0..<vertexCount {
            for component in 0..<components {
                trimmed[vertex * components + component] =
                    data[vertex * layout.components + component]
            }
        }
        return trimmed
    }
}
