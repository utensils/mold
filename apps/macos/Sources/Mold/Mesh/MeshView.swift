import MetalKit
import MoldClient
import SwiftUI

/// The mesh itself, as an AppKit view SwiftUI hosts.
///
/// Deliberately thin: the renderer owns the camera and the GPU state, the
/// `MTKView` subclass owns the events, and this only carries the tokens that
/// say "something changed, draw again". The canvas above it owns loading and
/// the failure sentence, so this view is never asked to draw nothing.
struct MeshView: NSViewRepresentable {
    let renderer: MeshRenderer
    let scene: MeshScene
    let autoRotate: Bool
    /// Bumped by the canvas when the wireframe is toggled or the camera is
    /// reset from a control, because neither is something the view can see.
    let redrawToken: Int
    let resetToken: Int
    let onAutoRotatingChange: (Bool) -> Void
    /// The first drag, key or wheel: the canvas stops offering the tour.
    let onInteraction: () -> Void

    final class Coordinator {
        var redrawToken = 0
        var resetToken = 0
        /// Which mesh is on the GPU. SwiftUI reuses this `NSView` across
        /// prints, so the NEXT mesh has to be installed here rather than in
        /// `makeNSView`, or the lightbox would keep drawing the first one.
        var installed: ObjectIdentifier?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> MeshMetalView {
        let view = MeshMetalView(renderer: renderer)
        view.onAutoRotateChange = { rotating in
            onAutoRotatingChange(rotating)
            if !rotating { onInteraction() }
        }
        renderer.install(scene)
        view.wantsAutoRotate = autoRotate
        context.coordinator.installed = ObjectIdentifier(scene)
        context.coordinator.redrawToken = redrawToken
        context.coordinator.resetToken = resetToken
        view.setAccessibilityLabel(Self.label(scene))
        view.setAccessibilityRole(.image)
        return view
    }

    func updateNSView(_ view: MeshMetalView, context: Context) {
        if context.coordinator.installed != ObjectIdentifier(scene) {
            context.coordinator.installed = ObjectIdentifier(scene)
            renderer.install(scene)
            view.resetView()
        }
        if view.wantsAutoRotate != autoRotate { view.wantsAutoRotate = autoRotate }
        if context.coordinator.resetToken != resetToken {
            context.coordinator.resetToken = resetToken
            view.resetView()
        }
        if context.coordinator.redrawToken != redrawToken {
            context.coordinator.redrawToken = redrawToken
            view.needsDisplay = true
        }
    }

    /// Releases the mesh's buffers and its texture the moment the view goes
    /// away: a lightbox opens and closes all session, and a Mac that keeps
    /// every mesh it has ever shown is one that eventually cannot show one.
    static func dismantleNSView(_ view: MeshMetalView, coordinator: Coordinator) {
        // The callback goes FIRST. `stopTour` fires it whenever the tour was
        // running, and it reports an INTERACTION -- so stepping from one mesh
        // to the next used to retire the tour for every mesh after it, with
        // nobody having touched anything, and wrote SwiftUI state during
        // teardown while it was at it.
        view.onAutoRotateChange = nil
        view.stopTour()
        view.delegate = nil
        view.renderer.release()
    }

    private static func label(_ scene: MeshScene) -> String {
        "Interactive 3-D view. Drag or use the arrow keys to turn it, "
            + "plus and minus to zoom, 0 to go back to the first view."
    }
}
