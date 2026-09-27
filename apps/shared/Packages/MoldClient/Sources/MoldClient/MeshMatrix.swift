import Foundation

/// Column-major 4×4 matrices, the order a GPU uniform wants them in.
///
/// Port of the matrix half of `studio/lib/meshViewerCamera.ts:169-259`. A flat
/// sixteen-float array rather than a `simd_float4x4` so the arithmetic is the
/// reference's, element for element, and a test can read one element without a
/// graphics framework — the app widens it at the uniform boundary.
public typealias Mat4 = [Float]

public enum MeshMatrix {

    public static func identity() -> Mat4 {
        var m = Mat4(repeating: 0, count: 16)
        m[0] = 1
        m[5] = 1
        m[10] = 1
        m[15] = 1
        return m
    }

    public static func multiply(_ a: Mat4, _ b: Mat4) -> Mat4 {
        var out = Mat4(repeating: 0, count: 16)
        guard a.count == 16, b.count == 16 else { return out }
        for column in 0..<4 {
            for row in 0..<4 {
                var sum = 0.0
                for k in 0..<4 {
                    sum += Double(a[k * 4 + row]) * Double(b[column * 4 + k])
                }
                out[column * 4 + row] = Float(sum)
            }
        }
        return out
    }

    public static func translation(_ x: Double, _ y: Double, _ z: Double) -> Mat4 {
        var m = identity()
        m[12] = Float(x)
        m[13] = Float(y)
        m[14] = Float(z)
        return m
    }

    public static func rotationX(_ angle: Double) -> Mat4 {
        var m = identity()
        m[5] = Float(cos(angle))
        m[6] = Float(sin(angle))
        m[9] = Float(-sin(angle))
        m[10] = Float(cos(angle))
        return m
    }

    public static func rotationY(_ angle: Double) -> Mat4 {
        var m = identity()
        m[0] = Float(cos(angle))
        m[2] = Float(-sin(angle))
        m[8] = Float(sin(angle))
        m[10] = Float(cos(angle))
        return m
    }

    /// A symmetric orthographic projection: the server renders the poster and
    /// every turntable frame orthographically, so the viewer must too or its
    /// home view would carry a perspective the thumbnail does not.
    public static func orthographic(halfWidth: Double, halfHeight: Double,
                                    near: Double, far: Double) -> Mat4 {
        var m = Mat4(repeating: 0, count: 16)
        m[0] = Float(1 / halfWidth)
        m[5] = Float(1 / halfHeight)
        m[10] = Float(-2 / (far - near))
        m[14] = Float(-(far + near) / (far - near))
        m[15] = 1
        return m
    }

    /// The upper-left 3×3, column-major. The camera only ROTATES, so this IS
    /// the normal matrix — no inverse-transpose is needed or wanted.
    public static func upper3x3(_ m: Mat4) -> [Float] {
        guard m.count == 16 else { return [Float](repeating: 0, count: 9) }
        return [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]
    }

    /// The model-view matrix the viewer draws with, at one camera:
    /// `T(0, 0, -3r) · RX(pitch) · RY(yaw) · T(-centre)`.
    ///
    /// `MeshViewer.vue:307-313`. The mesh is centred on the query GRID rather
    /// than on the model, so the orbit is about the bounding-box centre and
    /// never about the origin.
    public static func modelView(camera: ViewerCamera, center: SIMD3<Double>,
                                 distance: Double) -> Mat4 {
        multiply(
            multiply(translation(0, 0, -distance),
                     multiply(rotationX(camera.pitch), rotationY(camera.yaw))),
            translation(-center.x, -center.y, -center.z))
    }
}
