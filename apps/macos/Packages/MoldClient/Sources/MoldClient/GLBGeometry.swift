import Foundation

/// The two things a parsed mesh needs that the file may not carry: its box,
/// and normals when it has none.
///
/// Port of `computeBounds` and `generateNormals`, `studio/lib/glb.ts:386-461`.
enum GLBGeometry {

    /// The axis-aligned box, refusing a non-finite vertex rather than letting
    /// it poison every camera that reads the centre.
    static func bounds(of positions: [Float]) throws -> MeshBounds {
        var min = SIMD3<Double>(repeating: .infinity)
        var max = SIMD3<Double>(repeating: -.infinity)
        var index = 0
        while index + 2 < positions.count {
            let point = SIMD3<Double>(Double(positions[index]),
                                      Double(positions[index + 1]),
                                      Double(positions[index + 2]))
            guard point.x.isFinite, point.y.isFinite, point.z.isFinite else {
                throw GLBParseError("the POSITION accessor holds a non-finite value")
            }
            min = min.replacing(with: point, where: point .< min)
            max = max.replacing(with: point, where: point .> max)
            index += 3
        }
        return MeshBounds(min: min, max: max)
    }

    /// Area-weighted smooth normals, for meshes written without `NORMAL`.
    ///
    /// glTF winds front faces counter-clockwise, so `(b - a) × (c - a)`
    /// already points out of the surface. The cross product is left
    /// UNNORMALIZED on purpose: its length is twice the triangle's area,
    /// which is the weight smooth shading wants.
    ///
    /// Accumulated in single precision, because the reference accumulates
    /// into a `Float32Array` and rounds on every store.
    static func generatedNormals(positions: [Float], indices: [UInt32]) -> [Float] {
        var normals = [Float](repeating: 0, count: positions.count)
        var triangle = 0
        while triangle + 2 < indices.count {
            let a = Int(indices[triangle]) * 3
            let b = Int(indices[triangle + 1]) * 3
            let c = Int(indices[triangle + 2]) * 3
            triangle += 3
            guard let base = vertex(positions, a), let second = vertex(positions, b),
                  let third = vertex(positions, c)
            else { continue }
            let u = second - base
            let v = third - base
            let normal = SIMD3<Double>(u.y * v.z - u.z * v.y,
                                       u.z * v.x - u.x * v.z,
                                       u.x * v.y - u.y * v.x)
            for slot in [a, b, c] {
                normals[slot] = Float(Double(normals[slot]) + normal.x)
                normals[slot + 1] = Float(Double(normals[slot + 1]) + normal.y)
                normals[slot + 2] = Float(Double(normals[slot + 2]) + normal.z)
            }
        }
        var index = 0
        while index + 2 < normals.count {
            let x = Double(normals[index])
            let y = Double(normals[index + 1])
            let z = Double(normals[index + 2])
            let length = (x * x + y * y + z * z).squareRoot()
            if length > 0 {
                normals[index] = Float(x / length)
                normals[index + 1] = Float(y / length)
                normals[index + 2] = Float(z / length)
            } else {
                // An unreferenced or degenerate vertex: any unit vector beats NaN.
                normals[index + 1] = 1
            }
            index += 3
        }
        return normals
    }

    private static func vertex(_ positions: [Float], _ at: Int) -> SIMD3<Double>? {
        guard at >= 0, at + 2 < positions.count else { return nil }
        return SIMD3<Double>(Double(positions[at]), Double(positions[at + 1]),
                             Double(positions[at + 2]))
    }
}
