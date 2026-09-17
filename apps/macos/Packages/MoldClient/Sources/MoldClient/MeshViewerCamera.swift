import Foundation

/// The ONE camera convention shared by the server's poster, the server's
/// turntable, `studio/components/MeshViewer.vue`'s home view, and this app's
/// `MeshView`.
///
/// The four literals below MIRROR
/// `crates/mold-inference/src/hunyuan3d/poster.rs` (`POSTER_AZIMUTH_DEG`,
/// `POSTER_ELEVATION_DEG`, `POSTER_MARGIN`, `TURNTABLE_AZIMUTH_STEP_SIGN`),
/// exactly as `studio/lib/meshViewerCamera.ts` does. A Rust test there READS
/// THIS FILE and fails the build when they drift, which is why they keep the
/// Rust spelling instead of a Swift one — change them in every place or not
/// at all.
///
/// The conversion between the two frames lives here and nowhere else: the
/// server orbits the eye by `azimuth` about +Y with azimuth 0 on +Z; the
/// viewer rotates the MODEL by `yaw` about +Y, so `yaw = -azimuth` and
/// `pitch = elevation`.
public enum MeshViewerCamera {
    /// Orbit angle of the poster's eye about +Y, degrees. 0 places the eye on +Z.
    public static let POSTER_AZIMUTH_DEG = 30.0
    /// Angle of the poster's eye above the XZ plane, degrees.
    public static let POSTER_ELEVATION_DEG = 20.0
    /// Fraction of the frame left empty around the mesh's sweep extent.
    public static let POSTER_MARGIN = 0.08
    /// Sign of the turntable's per-frame azimuth step. Negative means the eye
    /// orbits toward -X, so the object spins to the RIGHT on screen — the way
    /// a rightward drag turns it here, and the way auto-rotate tours it.
    public static let TURNTABLE_AZIMUTH_STEP_SIGN = -1.0

    /// The camera the viewer opens on, and the one `0` returns it to: EXACTLY
    /// the server's poster camera, so the gallery thumbnail, the viewer's
    /// first frame and turntable frame 0 are the same picture.
    public static func homeCamera() -> ViewerCamera {
        ViewerCamera(yaw: -POSTER_AZIMUTH_DEG * .pi / 180,
                     pitch: POSTER_ELEVATION_DEG * .pi / 180,
                     zoom: 1)
    }

    /// The server-frame azimuth, in degrees, a viewer yaw is looking from.
    /// The one place the two conventions are converted, in either direction.
    public static func azimuthDegOfYaw(_ yaw: Double) -> Double {
        -yaw * 180 / .pi
    }

    /// The `(radial, |dy|)` pair per finite vertex, interleaved — everything
    /// [`sweepExtentOfProfile`] needs, and nothing else.
    ///
    /// Two thirds the size of the positions it replaces, and it lifts the
    /// centring, the hypotenuse and the finiteness check out of the
    /// per-elevation loop. Stored single-precision, which is the precision
    /// `sweep_fit_for` computes the same bound at.
    public static func sweepProfile(_ positions: [Float],
                                    center: SIMD3<Double>) -> [Float] {
        var out: [Float] = []
        out.reserveCapacity((positions.count / 3) * 2)
        var index = 0
        while index + 2 < positions.count {
            let delta = SIMD3<Double>(Double(positions[index]) - center.x,
                                      Double(positions[index + 1]) - center.y,
                                      Double(positions[index + 2]) - center.z)
            index += 3
            // A NaN or infinite vertex would poison the max for the whole mesh.
            guard delta.x.isFinite, delta.y.isFinite, delta.z.isFinite else { continue }
            out.append(Float((delta.x * delta.x + delta.z * delta.z).squareRoot()))
            out.append(Float(abs(delta.y)))
        }
        return out
    }

    /// The rotation-invariant half-extent that frames a prepared profile from
    /// EVERY azimuth at `elevationRad`.
    ///
    /// The closed form of the bounding cylinder about the bounding box's
    /// centre: for each vertex, the larger of its radial distance and its
    /// projected height `cos e · |dy| + sin e · radial`. Mirrors
    /// `sweep_fit_for` in `crates/mold-inference/src/hunyuan3d/raster.rs`;
    /// because it depends on neither the azimuth nor the frame count, the
    /// poster, a 36-frame turntable, a 72-frame one and this viewer all frame
    /// the mesh identically.
    ///
    /// The elevation is used by MAGNITUDE, so looking up at the mesh frames it
    /// exactly as looking down does.
    public static func sweepExtentOfProfile(_ profile: [Float],
                                            elevationRad: Double) -> Double {
        let sinE = abs(sin(elevationRad))
        let cosE = abs(cos(elevationRad))
        var extent = 0.0
        var index = 0
        while index + 1 < profile.count {
            let radial = Double(profile[index])
            let height = Double(profile[index + 1])
            index += 2
            let candidate = Swift.max(radial, cosE * height + sinE * radial)
            if candidate > extent { extent = candidate }
        }
        return extent
    }

    /// [`sweepExtentOfProfile`] straight from the positions. `0` when there is
    /// nothing finite to frame, which the caller reads as "draw nothing"
    /// rather than as a scale.
    public static func sweepExtent(_ positions: [Float], center: SIMD3<Double>,
                                   elevationRad: Double) -> Double {
        sweepExtentOfProfile(sweepProfile(positions, center: center),
                             elevationRad: elevationRad)
    }

    /// Pixels per world unit for a frame of `width` × `height` that leaves
    /// `margin` of itself empty around `extent`.
    ///
    /// `fit_scale`'s `FrameFit::Extent` arm in
    /// `crates/mold-inference/src/hunyuan3d/raster.rs`, margin clamp included.
    /// `0` — never infinity — for a mesh with no extent, so a caller that
    /// multiplies by it cannot produce a NaN projection matrix.
    public static func orthographicScale(extent: Double, width: Double, height: Double,
                                         margin: Double) -> Double {
        guard extent.isFinite, extent > 0 else { return 0 }
        let half = Swift.min(0.5 * width, 0.5 * height)
        let clamped = Swift.min(Swift.max(margin.isFinite ? margin : 0, 0), 0.9)
        let scale = (half / extent) * (1 - clamped)
        return scale.isFinite && scale > 0 ? scale : 0
    }
}

/// The viewer's own camera state: model rotation plus a zoom-out factor.
public struct ViewerCamera: Equatable, Sendable {
    /// Model rotation about +Y, radians. `yaw = -azimuth`.
    public var yaw: Double
    /// Model rotation about +X, radians. `pitch = elevation`.
    public var pitch: Double
    /// Multiplies the framed extent: 1 is the poster's own framing.
    public var zoom: Double

    public init(yaw: Double, pitch: Double, zoom: Double) {
        self.yaw = yaw
        self.pitch = pitch
        self.zoom = zoom
    }
}
