import Foundation

/// The arithmetic behind the mesh view's auto-rotation and wireframe overlay,
/// kept out of the view so it can be tested without a GPU.
///
/// Port of `studio/lib/meshViewerMath.ts`.
public enum MeshViewerMath {
    /// The tour's speed, radians a second.
    public static let autoRotateRadiansPerSecond = 0.25

    private static let tau = Double.pi * 2

    /// Folds any angle into `[-π, π)`.
    static func wrapAngle(_ angle: Double) -> Double {
        guard angle.isFinite else { return 0 }
        // `%` keeps the dividend's sign in both languages, so the `+ tau` and
        // the second fold are what make a negative angle come back positive.
        let shifted = (angle + .pi).truncatingRemainder(dividingBy: tau)
        return (shifted + tau).truncatingRemainder(dividingBy: tau) - .pi
    }

    /// The next yaw for an auto-rotating viewer, `elapsedMs` after the last
    /// frame.
    ///
    /// Always wrapped into `[-π, π)`: a gallery left open all day would
    /// otherwise accumulate an ever-larger angle whose float precision — and
    /// whose rotation matrix — quietly degrade.
    public static func advanceAutoRotate(
        yaw: Double, elapsedMs: Double,
        radiansPerSecond: Double = autoRotateRadiansPerSecond
    ) -> Double {
        guard elapsedMs.isFinite, elapsedMs > 0 else { return wrapAngle(yaw) }
        return wrapAngle(yaw + radiansPerSecond * elapsedMs / 1000)
    }

    /// The deduplicated undirected edge list of a triangle index buffer, ready
    /// to be drawn as lines.
    ///
    /// Each edge appears exactly once as an ordered `[min, max]` pair, in
    /// first-seen order, so a shared edge is drawn once rather than twice and
    /// the buffer is stable across calls. Degenerate edges (a vertex joined to
    /// itself) and a trailing partial triangle are dropped.
    ///
    /// One `Set` of packed `low * vertexCount + high` keys and one
    /// preallocated output: a two-million-face mesh used to allocate a set per
    /// vertex plus a growing array and freeze for seconds on the first
    /// wireframe toggle.
    public static func edgeIndices(_ indices: [UInt32]) -> [UInt32] {
        let triangles = indices.count - (indices.count % 3)
        if triangles == 0 { return [] }

        var maxIndex: UInt64 = 0
        for position in 0..<triangles where UInt64(indices[position]) > maxIndex {
            maxIndex = UInt64(indices[position])
        }
        let vertexCount = maxIndex + 1

        var seen = Set<UInt64>()
        seen.reserveCapacity(triangles)
        var out = [UInt32]()
        out.reserveCapacity(triangles * 2)

        func add(_ a: UInt32, _ b: UInt32) {
            if a == b { return }
            let low = Swift.min(a, b)
            let high = Swift.max(a, b)
            let key = UInt64(low) * vertexCount + UInt64(high)
            if seen.contains(key) { return }
            seen.insert(key)
            out.append(low)
            out.append(high)
        }

        var position = 0
        while position < triangles {
            let a = indices[position]
            let b = indices[position + 1]
            let c = indices[position + 2]
            add(a, b)
            add(b, c)
            add(a, c)
            position += 3
        }
        return out
    }

    /// Whether [`edgeIndices`] would emit anything at all: at least one
    /// complete triangle joining two distinct vertices. A linear scan with no
    /// allocation, so a view can decide whether to OFFER the wireframe toggle
    /// without paying for an edge list nobody may ask for.
    public static func meshHasEdges(_ indices: [UInt32]) -> Bool {
        let triangles = indices.count - (indices.count % 3)
        var position = 0
        while position < triangles {
            let a = indices[position]
            let b = indices[position + 1]
            let c = indices[position + 2]
            if a != b || b != c { return true }
            position += 3
        }
        return false
    }
}
