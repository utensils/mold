import Foundation

/// A `.glb` this reader will not or cannot read, with the reason why.
///
/// Every refusal is a SENTENCE, because the viewer shows it: the gallery
/// hands these bytes straight off the wire, and "couldn't read it" with no
/// reason is what a black rectangle says.
public struct GLBParseError: Error, Equatable, Sendable, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

/// The bounding box of a parsed mesh, in model units.
///
/// Doubles, not floats: the centre every camera orbits is the midpoint of
/// this box, and the port it mirrors (`studio/lib/glb.ts:388-408`) computes
/// it in JavaScript numbers.
public struct MeshBounds: Hashable, Sendable {
    public let min: SIMD3<Double>
    public let max: SIMD3<Double>

    public init(min: SIMD3<Double>, max: SIMD3<Double>) {
        self.min = min
        self.max = max
    }

    /// The point a camera orbits: mold's writer centres a mesh on the query
    /// GRID rather than on the model, so this is never assumed to be zero.
    public var center: SIMD3<Double> { (min + max) / 2 }

    /// Half the box's diagonal, floored so a degenerate box still gives a
    /// usable depth range.
    public var radius: Double {
        Swift.max(((max - min) * 0.5).lengthSquared.squareRoot(), 1e-4)
    }
}

private extension SIMD3 where Scalar == Double {
    var lengthSquared: Double { x * x + y * y + z * z }
}

/// An embedded baseColor image, exactly as the file carries it.
///
/// Bytes rather than a decoded picture, because the decode belongs to
/// whichever surface is going to upload it to a GPU. `uri` images are never
/// followed: a second fetch is not something a mesh file gets to ask for.
public struct MeshTexture: Hashable, Sendable {
    public let data: Data
    public let mimeType: String

    public init(data: Data, mimeType: String) {
        self.data = data
        self.mimeType = mimeType
    }
}

/// The first triangle primitive of the first mesh in a `.glb`.
///
/// The Swift half of `studio/lib/glb.ts`'s `ParsedMesh`. `normals` is never
/// nil: a file without them gets area-weighted ones, so a mesh that arrives
/// without normals shades like one that carries them instead of rendering
/// black (`glb.ts:410-461`).
public struct ParsedMesh: Sendable {
    /// xyz triples, `vertexCount * 3` long.
    public let positions: [Float]
    /// xyz triples, supplied by the file or generated.
    public let normals: [Float]
    /// uv pairs, or nil.
    public let uvs: [Float]?
    /// rgb triples, or nil. A VEC4 COLOR_0 arrives here with its alpha dropped.
    public let colors: [Float]?
    public let indices: [UInt32]
    public let baseColorTexture: MeshTexture?
    public let bounds: MeshBounds
    public let vertexCount: Int
    public let triangleCount: Int

    public init(positions: [Float], normals: [Float], uvs: [Float]?, colors: [Float]?,
                indices: [UInt32], baseColorTexture: MeshTexture?, bounds: MeshBounds,
                vertexCount: Int, triangleCount: Int) {
        self.positions = positions
        self.normals = normals
        self.uvs = uvs
        self.colors = colors
        self.indices = indices
        self.baseColorTexture = baseColorTexture
        self.bounds = bounds
        self.vertexCount = vertexCount
        self.triangleCount = triangleCount
    }
}
