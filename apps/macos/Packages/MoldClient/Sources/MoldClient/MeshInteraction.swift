import Foundation

/// How a pointer, a wheel and the keyboard move the mesh camera.
///
/// Every constant is `studio/components/MeshViewer.vue`'s (`:871-973`), kept
/// out of the view so the gestures can be tested without a GPU: drag orbit at
/// `0.008 rad/px`, a pitch clamp just inside the poles, zoom 0.25…6, and NO
/// PAN -- the mesh is framed by the poster's own fit and a pan would show a
/// picture the thumbnail never had.
public enum MeshInteraction {
    public static let dragRadiansPerPixel = 0.008
    /// Just short of ±90°, so the orbit never degenerates at a pole.
    public static let pitchLimit = Double.pi / 2 - 0.01
    public static let minimumZoom = 0.25
    public static let maximumZoom = 6.0
    /// `exp(deltaY * 0.0015)` -- a wheel notch is a ratio, not a step.
    public static let wheelGain = 0.0015
    public static let keyRadians = 0.12
    public static let shiftKeyRadians = 0.30
    public static let keyZoomFactor = 1.15

    /// What a key press means here. The view maps the event; this decides.
    public enum Key: Hashable, Sendable {
        case orbitLeft, orbitRight, orbitUp, orbitDown
        case zoomIn, zoomOut
        case home
    }

    public static func orbit(_ camera: ViewerCamera, dx: Double, dy: Double) -> ViewerCamera {
        var next = camera
        next.yaw += dx
        next.pitch = Swift.min(pitchLimit, Swift.max(-pitchLimit, camera.pitch + dy))
        return next
    }

    /// A factor above 1 pulls BACK: `zoom` multiplies the framed extent, which
    /// is the sense the wheel, the pinch and `+`/`-` have always spoken.
    public static func zoom(_ camera: ViewerCamera, by factor: Double) -> ViewerCamera {
        guard factor.isFinite, factor > 0 else { return camera }
        var next = camera
        next.zoom = Swift.min(maximumZoom, Swift.max(minimumZoom, camera.zoom * factor))
        return next
    }

    public static func wheelFactor(deltaY: Double) -> Double {
        guard deltaY.isFinite else { return 1 }
        return exp(deltaY * wheelGain)
    }

    /// A pinch reports the distance between two fingers; the camera wants the
    /// ratio of the PREVIOUS distance to the current one, so spreading zooms in.
    public static func pinchFactor(previous: Double, current: Double) -> Double {
        guard previous > 0, current > 0 else { return 1 }
        return previous / current
    }

    public static func apply(_ key: Key, to camera: ViewerCamera,
                             shift: Bool) -> ViewerCamera {
        let step = shift ? shiftKeyRadians : keyRadians
        switch key {
        case .orbitLeft: return orbit(camera, dx: -step, dy: 0)
        case .orbitRight: return orbit(camera, dx: step, dy: 0)
        case .orbitUp: return orbit(camera, dx: 0, dy: -step)
        case .orbitDown: return orbit(camera, dx: 0, dy: step)
        case .zoomIn: return zoom(camera, by: 1 / keyZoomFactor)
        case .zoomOut: return zoom(camera, by: keyZoomFactor)
        case .home: return MeshViewerCamera.homeCamera()
        }
    }

    /// The characters `+`, `-` and `0` answer to; the arrows are matched by the
    /// view from the event's own special-key value, because they have no
    /// printable character.
    public static func key(forCharacters characters: String) -> Key? {
        switch characters {
        case "+", "=": return .zoomIn
        case "-", "_": return .zoomOut
        case "0": return .home
        default: return nil
        }
    }
}
