import AppKit
import MoldClient

// Drag to orbit, scroll or pinch to zoom, arrows and +/- and 0 from the
// keyboard. Every gesture's arithmetic is `MeshInteraction`'s, so what the
// mouse means is testable without a window.
extension MeshMetalView {

    private func move(_ camera: ViewerCamera) {
        renderer.setCamera(camera)
        needsDisplay = true
    }

    override func mouseDown(with event: NSEvent) {
        noteInteraction()
        window?.makeFirstResponder(self)
        // Double-click is the same reset as `0` and the Reset View control.
        if event.clickCount == 2 { move(MeshViewerCamera.homeCamera()) }
    }

    override func mouseDragged(with event: NSEvent) {
        noteInteraction()
        // AppKit's y grows UPWARD and the reference's pointer delta grows
        // downward, so the vertical delta is negated to keep "drag down tips
        // the top of the mesh away".
        move(MeshInteraction.orbit(
            renderer.camera,
            dx: Double(event.deltaX) * MeshInteraction.dragRadiansPerPixel,
            dy: Double(-event.deltaY) * MeshInteraction.dragRadiansPerPixel))
    }

    override func scrollWheel(with event: NSEvent) {
        noteInteraction()
        let delta = event.hasPreciseScrollingDeltas
            ? Double(event.scrollingDeltaY) : Double(event.scrollingDeltaY) * 10
        move(MeshInteraction.zoom(renderer.camera,
                                  by: MeshInteraction.wheelFactor(deltaY: delta)))
    }

    /// A trackpad pinch. `magnification` is a signed fraction, so spreading
    /// comes closer exactly as it does in every other viewer on this Mac.
    override func magnify(with event: NSEvent) {
        noteInteraction()
        let magnification = 1 + Double(event.magnification)
        move(MeshInteraction.zoom(
            renderer.camera,
            by: MeshInteraction.pinchFactor(previous: 1, current: magnification)))
    }

    override func keyDown(with event: NSEvent) {
        guard let key = Self.key(for: event) else {
            super.keyDown(with: event)
            return
        }
        noteInteraction()
        move(MeshInteraction.apply(key, to: renderer.camera,
                                   shift: event.modifierFlags.contains(.shift)))
    }

    /// The arrows carry no printable character, so they are matched by their
    /// special-key value; everything else is `MeshInteraction`'s own table.
    static func key(for event: NSEvent) -> MeshInteraction.Key? {
        switch event.specialKey {
        case .some(.leftArrow): return .orbitLeft
        case .some(.rightArrow): return .orbitRight
        case .some(.upArrow): return .orbitUp
        case .some(.downArrow): return .orbitDown
        default:
            return MeshInteraction.key(forCharacters: event.charactersIgnoringModifiers ?? "")
        }
    }

    /// Back to the poster's camera, from the control and from the menu.
    func resetView() {
        move(MeshViewerCamera.homeCamera())
    }
}
