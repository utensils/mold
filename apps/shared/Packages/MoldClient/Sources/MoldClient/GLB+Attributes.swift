import Foundation

// The primitive's indices and its optional vertex attributes. Split from
// the entry point for size.
extension GLB {
    /// Indices, widened to `u32` whatever the file stored them as, and every
    /// one of them proved to name a vertex that exists.
    static func readIndices(document: [String: Any], bin: [UInt8],
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
    static func attribute(_ name: String, components: Int,
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
